# Neural Machine Translation with Attention Mechanism

> CSC 383/483 – Applied Deep Learning

An end-to-end English → French NMT pipeline built in **Keras / TensorFlow** using an **Encoder–Decoder LSTM** architecture with **Bahdanau (additive) attention**.

---

## Model Architecture

```
English sentence
    └── Encoder (Embedding → LSTM)
          ├── encoder_outputs  (batch, src_len, 128)   ← all timesteps
          ├── state_h          (batch, 128)             ← final hidden
          └── state_c          (batch, 128)             ← final cell

For each decoder timestep t:
    └── DecoderStep
          ├── BahdanauAttention(encoder_outputs, h_t)  → context (batch, 128)
          ├── Embedding(token_t)                        → (batch, 64)
          ├── Concat([embedding, context])              → (batch, 192)
          ├── LSTMCell                                  → new h, c
          └── Dense(fra_vocab_size, softmax)            → next-token probs
```

### Bahdanau Attention

```
e_{t,i} = V( tanh( W1·h_i^enc + W2·s_{t-1}^dec ) )
α_{t,i} = softmax(e_{t,i})          over source dim
c_t     = Σ α_{t,i} · h_i^enc
```

---

## Project Structure

```
nmt-attention/
├── NMT_Attention.ipynb      # Complete notebook — all 8 tasks solved
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.8+
- Google Colab T4 GPU recommended

### Install
```bash
pip install -r requirements.txt
```

### Dataset
Download `english_french.csv` from [Kaggle](https://www.kaggle.com/datasets/rajpulapakura/english-to-french-small-dataset) and upload to `/content/` in Colab.

---

## Usage

```
Runtime → Change runtime type → T4 GPU
Runtime → Run all
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR_USERNAME>/nmt-attention/blob/main/NMT_Attention.ipynb)

> Replace `<YOUR_USERNAME>` with your GitHub handle.

---

## Key Concepts

**Teacher Forcing** — During training the ground-truth previous token is fed to the decoder instead of its own prediction, stabilising gradients.

**Padding Mask** — Padding tokens (id = 0) are excluded from the loss so the model is not penalised for ignoring them.

**Bahdanau Attention** — Allows the decoder to dynamically focus on different parts of the source sentence at each generation step, overcoming the fixed-vector bottleneck.

---

## License
Academic use — CSC 383/483.
