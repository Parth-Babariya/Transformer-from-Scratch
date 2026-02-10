import torch
import argparse
import spacy
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.metrics import bleu_score
from tqdm import tqdm

from train import (make_model, TranslationDataset,
                   token_transform, vocab_transform,
                   SRC_LANGUAGE, TGT_LANGUAGE,
                   SOS_IDX, EOS_IDX, PAD_IDX, config)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_test_mask(src):
    # src shape: [seq_len, batch_size]
    src_padding_mask = (src == PAD_IDX).transpose(0, 1) # [batch_size, seq_len]
    # Unsqueeze to add dimensions for multi-head attention
    src_padding_mask = src_padding_mask.unsqueeze(1).unsqueeze(2) # [batch_size, 1, 1, seq_len]
    return src_padding_mask.to(DEVICE)

# --- SINGLE-SENTENCE DECODING FUNCTIONS (for beam search and final example) ---

def greedy_decode(model, src, src_mask, max_len, start_symbol, min_len=5):
    model.eval()
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        tgt_seq_len = ys.shape[0]
        tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=DEVICE), diagonal=1).bool()
        out = model.decode(memory, ys, src_mask.squeeze(1).squeeze(1), tgt_mask)
        prob = model.generator(out[-1, :, :])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX and ys.shape[0] > min_len:
            break
    return ys

def beam_search_decode(model, src, src_mask, max_len, start_symbol, beam_width, min_len=5):
    model.eval()
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)
    beams = [(torch.ones(1, 1).fill_(start_symbol).long().to(DEVICE), 0.0)]
    for _ in range(max_len - 1):
        new_beams = []
        all_beams_ended = True
        for seq, score in beams:
            if seq[-1, 0].item() == EOS_IDX and seq.shape[0] > min_len:
                new_beams.append((seq, score))
                continue
            all_beams_ended = False
            tgt_seq_len = seq.shape[0]
            tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=DEVICE), diagonal=1).bool()
            out = model.decode(memory, seq, src_mask.squeeze(1).squeeze(1), tgt_mask)
            prob = torch.log_softmax(model.generator(out[-1, :, :]), dim=-1)
            top_k_probs, top_k_indices = torch.topk(prob, beam_width, dim=1)
            for i in range(beam_width):
                next_word_idx = top_k_indices[0, i].item()
                next_word_prob = top_k_probs[0, i].item()
                new_seq = torch.cat([seq, torch.tensor([[next_word_idx]], device=DEVICE)], dim=0)
                new_score = score + next_word_prob / (new_seq.shape[0] ** 0.6)
                new_beams.append((new_seq, new_score))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        if all_beams_ended: break
    best_seq, _ = max(beams, key=lambda x: x[1])
    return best_seq

def top_k_decode(model, src, src_mask, max_len, start_symbol, k, min_len=5):
    model.eval()
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        tgt_seq_len = ys.shape[0]
        tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=DEVICE), diagonal=1).bool()
        out = model.decode(memory, ys, src_mask.squeeze(1).squeeze(1), tgt_mask)
        logits = model.generator(out[-1, :, :])
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=1)
        dist = torch.distributions.Categorical(logits=top_k_logits)
        next_word_idx_in_k = dist.sample()
        next_word = top_k_indices.gather(1, next_word_idx_in_k.unsqueeze(1)).item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX and ys.shape[0] > min_len:
            break
    return ys

# --- BATCHED DECODING FUNCTIONS ---

def greedy_decode_batch(model, src, src_mask, max_len, start_symbol, min_len=5):
    model.eval()
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)
    batch_size = src.shape[1]
    ys = torch.ones(1, batch_size).fill_(start_symbol).type(torch.long).to(DEVICE)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)

    for _ in range(max_len - 1):
        tgt_seq_len = ys.shape[0]
        tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=DEVICE), diagonal=1).bool()
        out = model.decode(memory, ys, src_mask, tgt_mask)
        prob = model.generator(out[-1, :, :])
        _, next_word = torch.max(prob, dim=1)
        ys = torch.cat([ys, next_word.unsqueeze(0)], dim=0)
        finished = finished | (next_word == EOS_IDX)
        if finished.all() and ys.shape[0] > min_len:
            break
    return ys

def top_k_decode_batch(model, src, src_mask, max_len, start_symbol, k, min_len=5):
    model.eval()
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)
    batch_size = src.shape[1]
    ys = torch.ones(1, batch_size).fill_(start_symbol).type(torch.long).to(DEVICE)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)

    for _ in range(max_len - 1):
        tgt_seq_len = ys.shape[0]
        tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=DEVICE), diagonal=1).bool()
        out = model.decode(memory, ys, src_mask, tgt_mask)
        logits = model.generator(out[-1, :, :])
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=1)
        dist = torch.distributions.Categorical(logits=top_k_logits)
        next_word_idx_in_k = dist.sample()
        next_word = torch.gather(top_k_indices, 1, next_word_idx_in_k.unsqueeze(1)).squeeze(1)
        ys = torch.cat([ys, next_word.unsqueeze(0)], dim=0)
        finished = finished | (next_word == EOS_IDX)
        if finished.all() and ys.shape[0] > min_len:
            break
    return ys

# --- TRANSLATION & BATCHING ---

def collate_fn(batch):
    src_text_batch, tgt_text_batch, src_tensor_list = [], [], []
    for src_sample, tgt_sample in batch:
        src_text_batch.append(src_sample)
        tgt_text_batch.append(tgt_sample)
        src_tensor = torch.tensor([SOS_IDX] + [vocab_transform[SRC_LANGUAGE][token] for token in [t.text for t in token_transform[SRC_LANGUAGE](src_sample)]] + [EOS_IDX])
        src_tensor_list.append(src_tensor)
    
    src_padded = pad_sequence(src_tensor_list, padding_value=PAD_IDX)
    return src_text_batch, tgt_text_batch, src_padded

def translate(model, src_sentence, strategy, beam_width=5, k=10):
    src_tokens = [token.text for token in token_transform[SRC_LANGUAGE](src_sentence)]
    src_tensor = torch.tensor([SOS_IDX] + [vocab_transform[SRC_LANGUAGE][token] for token in src_tokens] + [EOS_IDX]).unsqueeze(1)
    src_mask = create_test_mask(src_tensor)
    
    if strategy == 'greedy':
        tgt_tokens = greedy_decode(model, src_tensor, src_mask, max_len=100, start_symbol=SOS_IDX).flatten()
    elif strategy == 'beam':
        tgt_tokens = beam_search_decode(model, src_tensor, src_mask, max_len=100, start_symbol=SOS_IDX, beam_width=beam_width).flatten()
    elif strategy == 'top_k':
        tgt_tokens = top_k_decode(model, src_tensor, src_mask, max_len=100, start_symbol=SOS_IDX, k=k).flatten()
    else:
        raise ValueError("Unknown or unsupported strategy for single sentence translation.")
        
    tgt_vocab_itos = vocab_transform[TGT_LANGUAGE].get_itos()
    translation = " ".join([tgt_vocab_itos[tok] for tok in tgt_tokens])
    return translation.replace("<sos>", "").replace("<eos>", "").replace("<pad>", "").strip()

def translate_batch(model, src_batch_tensor, strategy, k=10):
    src_mask = create_test_mask(src_batch_tensor)
    if strategy == 'greedy':
        tgt_tokens_batch = greedy_decode_batch(model, src_batch_tensor, src_mask, max_len=100, start_symbol=SOS_IDX)
    elif strategy == 'top_k':
        tgt_tokens_batch = top_k_decode_batch(model, src_batch_tensor, src_mask, max_len=100, start_symbol=SOS_IDX, k=k)
    else:
        raise ValueError("This batch function only supports 'greedy' and 'top_k' strategies.")

    tgt_vocab_itos = vocab_transform[TGT_LANGUAGE].get_itos()
    translations = []
    for i in range(tgt_tokens_batch.shape[1]):
        tokens = tgt_tokens_batch[:, i]
        eos_idx = (tokens == EOS_IDX).nonzero(as_tuple=True)[0]
        if len(eos_idx) > 0:
            tokens = tokens[1:eos_idx[0]]  # Exclude SOS and anything after EOS
        else:
            tokens = tokens[1:] # Exclude SOS
        
        translation = " ".join([tgt_vocab_itos[tok] for tok in tokens]).replace("<pad>", "").strip()
        translations.append(translation)
    return translations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a Transformer model.')
    parser.add_argument('--strategy', type=str, required=True, choices=['greedy', 'beam', 'top_k'], help='Decoding strategy.')
    parser.add_argument('--beam_width', type=int, default=5, help='Beam width for beam search')
    parser.add_argument('--k', type=int, default=10, help='k for top-k sampling')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--pe_strategy', type=str, default=config['pe_strategy'], help='PE strategy of the model to load')
    args = parser.parse_args()

    model_path = f"model_epoch_{config['epochs']}_{args.pe_strategy}.pt"
    model = make_model(len(vocab_transform[SRC_LANGUAGE]), len(vocab_transform[TGT_LANGUAGE]),
                       N=config['num_layers'], d_model=config['d_model'], d_ff=config['d_ff'],
                       h=config['num_heads'], dropout=config['dropout'], pe_strategy=args.pe_strategy)
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    print(f"Model from epoch {checkpoint['epoch']} loaded from {model_path}")

    torch.manual_seed(42)
    full_dataset = TranslationDataset('EUbookshop.fi', 'EUbookshop.en')
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    _, _, test_dataset_full = random_split(full_dataset, [train_size, val_size, len(full_dataset) - train_size - val_size])
    
    test_subset_size = 1000
    if len(test_dataset_full) > test_subset_size:
        test_dataset, _ = random_split(test_dataset_full, [test_subset_size, len(test_dataset_full) - test_subset_size],
                                       generator=torch.Generator().manual_seed(42))
    else:
        test_dataset = test_dataset_full

    print(f"\nEvaluating with {args.strategy} decoding on {len(test_dataset)} test samples...")
    
    candidate_corpus, references_corpus = [], []

    if args.strategy == 'beam':
        print("WARNING: Beam search is not implemented in batch mode and will be slow. Processing one by one.")
        for src_sent, tgt_sent in tqdm(test_dataset, desc="Translating Sentences"):
            translation = translate(model, src_sent, args.strategy, beam_width=args.beam_width)
            print(f"SRC:  {src_sent}\nTGT:  {tgt_sent}\nPRED: {translation}\n---")
            candidate_corpus.append([token.text for token in token_transform[TGT_LANGUAGE](translation)])
            references_corpus.append([[token.text for token in token_transform[TGT_LANGUAGE](tgt_sent)]])
    else: # Batch processing for greedy and top_k
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
        for src_batch, tgt_batch, src_padded in tqdm(test_dataloader, desc="Translating Batches"):
            translations = translate_batch(model, src_padded, args.strategy, k=args.k)
            for i in range(len(translations)):
                print(f"SRC:  {src_batch[i]}\nTGT:  {tgt_batch[i]}\nPRED: {translations[i]}\n---")
                candidate_corpus.append([token.text for token in token_transform[TGT_LANGUAGE](translations[i])])
                references_corpus.append([[token.text for token in token_transform[TGT_LANGUAGE](tgt_batch[i])]])

    bleu = bleu_score(candidate_corpus, references_corpus)
    print(f"\nFinal BLEU score on test set: {bleu*100:.2f}")

    example_src = "Ryhmä ihmisiä seisoo iglun edessä."
    print(f"\n--- Example Translation ---")
    print(f"Example Source: {example_src}")
    translation = translate(model, example_src, args.strategy, beam_width=args.beam_width, k=args.k)
    print(f"Translated Output ({args.strategy}): {translation}")