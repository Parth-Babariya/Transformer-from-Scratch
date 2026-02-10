# Assignment 1: Transformer from Scratch

This project contains a from-scratch implementation of the Transformer model for Finnish-to-English machine translation, as part of the Advanced NLP course.

## Project Structure

-   `utils.py`: Contains core building blocks like Multi-Head Attention, Positional Encodings (RoPE, Relative Bias), and the final Transformer model class.
-   `encoder.py`: Implements the Encoder part of the Transformer.
-   `decoder.py`: Implements the Decoder part of the Transformer.
-   `train.py`: The main script for training the model. It handles data loading, training loops, validation, and saving checkpoints.
-   `test.py`: A script to load a trained model and evaluate it on the test set using different decoding strategies.
-   `report.pdf`: The final report with analysis and observations.

## How to Run on Google Colab

Follow these steps to set up and run the project in a Google Colab notebook.

### 1. Upload Files

Upload the following files to your Colab environment's root directory:
* `utils.py`
* `encoder.py`
* `decoder.py`
* `train.py`
* `test.py`
* Your dataset files: `EUbookshop.fi` and `EUbookshop.en`

### 2. Install Dependencies

Run the following cell in your Colab notebook to install the necessary libraries and language models for **Finnish** and **English**.

```python
!pip install spacy==3.7.2
!python -m spacy download en_core_web_sm
!python -m spacy download fi_core_news_sm
```

3.  **Train the Model**: You can configure the training run by modifying the `config` dictionary in `train.py`. [cite_start]To switch between positional encodings, change the `pe_strategy` value to either `'rope'` or `'relative_bias'`[cite: 35].

    Run the training script from a Colab cell:
    ```bash
    !python train.py
    ```
    This will train the model, save `transformer_model.pt`, and plot the loss curves.

4.  **Test the Model**: Use the `test.py` script to evaluate the saved model. [cite_start]You can switch between decoding strategies using a command-line argument[cite: 45].

    Run the testing script for each strategy:
    ```bash
    # Greedy Decoding
    !python test.py --strategy greedy

    # Beam Search Decoding
    !python test.py --strategy beam --beam_width 5

    # Top-k Sampling
    !python test.py --strategy top_k --k 10
    ```

## Pre-trained Model

[**IMPORTANT**] You must upload your trained `.pt` file to a service like Google Drive or Dropbox and paste the public link here.
