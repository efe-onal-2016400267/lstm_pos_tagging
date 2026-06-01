# Part-of-Speech (POS) Tagging with LSTMs 🏷️🧠

A PyTorch-based implementation of a **Part-of-Speech (POS) Tagger** using **Long Short-Term Memory (LSTM)** networks. This project demonstrates how to build, train, and evaluate an LSTM-based sequence labeling model from scratch, using both a toy dataset and the standard **NLTK Brown Corpus** with the universal tagset.

---

## 🚀 Features

* **Custom LSTM Tagger Architecture (`LSTMTagger`)**:
  * Flexible embedding layer supporting both randomly initialized embeddings and pre-trained embedding tensors.
  * Recurrent sequence processing using PyTorch's `nn.LSTM`.
  * Fully connected linear layer mapping hidden states to tag space with `log_softmax` activation.
* **Robust Training & Evaluation Pipeline**:
  * Custom training loop (`train_tagger`) with Stochastic Gradient Descent (SGD) and learning rate scheduling.
  * Evaluation function (`evaluate_tagger`) computing average sentence-level Negative Log-Likelihood (NLL) loss and token-level accuracy.
  * Support for validation tracking at each epoch.
  * Interactive progress bars powered by `tqdm`.
* **Dataset Splits & Preprocessing**:
  * Custom train-validation splitting function with tag separation.
  * Vocabulary and tagset indexing for sequence preparation.
* **Visualizations**:
  * Line plots for training and validation loss/accuracy curves over epochs.

---

## 📂 Project Structure

* **`LSTM_POS_tagging.ipynb`**: The main Jupyter Notebook containing the complete implementation, including:
  * Helper functions for plotting and dataset splitting.
  * Toy dataset experiments (200 epochs) to verify model convergence.
  * Large-scale training on a 10% sample of the NLTK Brown Corpus (50 epochs).
  * Loss and accuracy visualization cells.

---

## 🧠 Model Architecture

The sequence labeling model processes sentences token-by-token:

1. **Embedding Layer**: Maps each word index to a dense vector of size `EMBEDDING_DIM`.
2. **LSTM Layer**: Processes the sequence of word embeddings, capturing left-to-right contextual information across hidden states of size `HIDDEN_DIM`.
3. **Linear Layer**: Maps each LSTM hidden state output to the size of the tagset.
4. **Softmax Activation**: Applies `log_softmax` to obtain log-probabilities for each POS tag.

---

## 🛠️ Getting Started

### 1. Prerequisites
Make sure you have Python 3.8+ installed.

### 2. Installation
To run the notebook and its dependencies, install the required packages via `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. NLTK Data Download
The notebook automatically downloads the required NLTK corpora upon execution:
```python
import nltk
nltk.download('brown')
nltk.download('universal_tagset')
```

---

## 💻 Usage

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `LSTM_POS_tagging.ipynb`.
3. Run all cells to:
   * Train the toy tagger and inspect its predictions.
   * Load and preprocess the NLTK Brown Corpus.
   * Train the full model on the Brown Corpus sample.
   * Plot the training and validation loss/accuracy curves.
