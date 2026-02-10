import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import spacy
from torchtext.vocab import build_vocab_from_iterator
import copy
import time
import matplotlib.pyplot as plt
import random
import datetime

from utils import Transformer, Generator, MultiHeadAttention, PositionwiseFeedForward
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer

# --- 1. Configuration ---
random.seed(42)
torch.manual_seed(42)

config = {
    "pe_strategy": "rope",
    "epochs": 10,
    "d_model": 512,
    "num_heads": 8,
    "num_layers": 6,
    "d_ff": 2048,
    "dropout": 0.1,
    "batch_size": 32,
    "lr": 0.0001,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 2. Data Loading and Preprocessing ---
SRC_LANGUAGE = 'fi'
TGT_LANGUAGE = 'en'
SRC_FILE_PATH = 'EUbookshop.fi'
TGT_FILE_PATH = 'EUbookshop.en'

token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = spacy.load('fi_core_news_sm').tokenizer
token_transform[TGT_LANGUAGE] = spacy.load('en_core_web_sm').tokenizer

def yield_tokens(file_path, language):
    with open(file_path, mode='rt', encoding='utf-8') as f:
        for line in f:
            yield [token.text for token in token_transform[language](line.strip())]

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

for ln, path in [(SRC_LANGUAGE, SRC_FILE_PATH), (TGT_LANGUAGE, TGT_FILE_PATH)]:
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(path, ln),
                                                    min_freq=2,
                                                    specials=special_symbols,
                                                    special_first=True)
    vocab_transform[ln].set_default_index(UNK_IDX)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
print(f"Source (fi) vocab size: {SRC_VOCAB_SIZE}")
print(f"Target (en) vocab size: {TGT_VOCAB_SIZE}")

class TranslationDataset(Dataset):
    def __init__(self, src_path, tgt_path):
        with open(src_path, 'r', encoding='utf-8') as f: self.src_sentences = f.readlines()
        with open(tgt_path, 'r', encoding='utf-8') as f: self.tgt_sentences = f.readlines()
        assert len(self.src_sentences) == len(self.tgt_sentences)
    def __len__(self): return len(self.src_sentences)
    def __getitem__(self, idx): return self.src_sentences[idx].strip(), self.tgt_sentences[idx].strip()

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_tokens = [token.text for token in token_transform[SRC_LANGUAGE](src_sample)]
        tgt_tokens = [token.text for token in token_transform[TGT_LANGUAGE](tgt_sample)]
        src_tensor = torch.tensor([SOS_IDX] + [vocab_transform[SRC_LANGUAGE][token] for token in src_tokens] + [EOS_IDX], dtype=torch.long)
        tgt_tensor = torch.tensor([SOS_IDX] + [vocab_transform[TGT_LANGUAGE][token] for token in tgt_tokens] + [EOS_IDX], dtype=torch.long)
        src_batch.append(src_tensor)
        tgt_batch.append(tgt_tensor)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

full_dataset = TranslationDataset(SRC_FILE_PATH, TGT_FILE_PATH)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, _ = random_split(full_dataset, [train_size, val_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=collate_fn, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], collate_fn=collate_fn)

# --- 3. Model Definition ---
def make_model(src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout, pe_strategy):
    encoder_layer = EncoderLayer(d_model, h, d_ff, dropout, pe_strategy)
    decoder_layer = DecoderLayer(d_model, h, d_ff, dropout, pe_strategy)
    model = Transformer(
        Encoder(encoder_layer, N, pe_strategy),
        Decoder(decoder_layer, N, pe_strategy),
        nn.Sequential(nn.Embedding(src_vocab, d_model), nn.Dropout(dropout)),
        nn.Sequential(nn.Embedding(tgt_vocab, d_model), nn.Dropout(dropout)),
        Generator(d_model, tgt_vocab)
    )
    for p in model.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
    return model

# --- 4. Training and Evaluation Loops ---
def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    batch_size = src.shape[1]

    # Causal mask: True where invalid (future)
    causal_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=DEVICE), diagonal=1).bool()

    # Pad masks: True where invalid (pad)
    src_padding_mask = (src == PAD_IDX).transpose(0, 1).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, src_seq_len)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, tgt_seq_len)

    # For tgt_mask: combine causal and tgt_pad
    tgt_mask = causal_mask.unsqueeze(0).unsqueeze(0) | tgt_padding_mask  # (B, 1, tgt_seq_len, tgt_seq_len) but broadcast from (1,1,T,T) | (B,1,1,T)

    # Expand causal to (1,1,T,T), tgt_pad to (B,1,1,T), | works since broadcast
    tgt_mask = causal_mask[None, None, :, :] | tgt_padding_mask

    return src_padding_mask, tgt_mask

def train_epoch(model, optimizer, criterion, dataloader):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask)
        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, criterion, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask = create_mask(src, tgt_input)
            logits = model(src, tgt_input, src_mask, tgt_mask)
            tgt_out = tgt[1:, :]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

# --- 5. Main Execution ---
if __name__ == "__main__":
    print(f"Training with PE strategy: {config['pe_strategy']}")
    model = make_model(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, N=config['num_layers'], d_model=config['d_model'],
                       d_ff=config['d_ff'], h=config['num_heads'], dropout=config['dropout'], 
                       pe_strategy=config['pe_strategy'])
    model.to(DEVICE)

    # DEFINITIVE FIX: Add label_smoothing to the loss function to combat overfitting to Teacher Forcing.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)

    train_losses, val_losses = [], []
    total_epochs = config['epochs']
    training_start_time = time.time()

    for epoch in range(1, total_epochs + 1):
        epoch_start_time = time.time()
        train_loss = train_epoch(model, optimizer, criterion, train_dataloader)
        val_loss = evaluate(model, criterion, val_dataloader)
        epoch_end_time = time.time()
        train_losses.append(train_loss); val_losses.append(val_loss)
        epoch_duration = epoch_end_time - epoch_start_time
        total_elapsed_time = time.time() - training_start_time
        avg_time_per_epoch = total_elapsed_time / epoch
        remaining_epochs = total_epochs - epoch
        estimated_remaining_time = remaining_epochs * avg_time_per_epoch
        epoch_duration_str = str(datetime.timedelta(seconds=int(epoch_duration)))
        eta_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))

        print(f"Epoch: {epoch:02d}/{total_epochs}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, "
              f"Epoch Time: {epoch_duration_str}, ETA: {eta_str}")
        
        checkpoint_path = f'model_epoch_{epoch}_{config["pe_strategy"]}.pt'
        torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'loss': val_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}\n")

    print("--- Training finished ---")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config['epochs'] + 1), train_losses, label='Training Loss')
    plt.plot(range(1, config['epochs'] + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss (PE: {config["pe_strategy"]})')
    plt.legend(); plt.grid(True); plt.savefig(f'loss_curve_{config["pe_strategy"]}.png'); plt.show()