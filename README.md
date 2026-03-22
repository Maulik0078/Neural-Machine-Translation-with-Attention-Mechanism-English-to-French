# Neural Machine Translation with Attention Mechanism

> CSC 483 – Applied Deep Learning

A full English-to-French neural machine translation pipeline combining **Encoder–Decoder LSTMs** and **Bahdanau (additive) attention** — allowing the model to dynamically focus on relevant source words at each decoding step.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Key Concepts](#key-concepts)
- [License](#license)

---

## Overview

Neural Machine Translation is the task of automatically translating text from one language to another using neural networks. A standard Encoder–Decoder compresses the entire source sentence into a single fixed-size vector — a bottleneck for longer sentences.

This project solves that with **Bahdanau attention**, which lets the decoder dynamically attend to all encoder hidden states at every generation step, producing significantly better translations on longer inputs.

---

## Results

| Metric | Value (60 epochs) |
|--------|:-----------------:|
| Training Loss | tracked via custom `GradientTape` loop |
| Translation Quality | verified against ground-truth French |
| Attention Heatmap | diagonal pattern confirms correct alignment |

> Translations are printed side-by-side with actual French labels.  
> Attention weights are visualized as **heatmaps** showing which source tokens the model focused on.

---

## Project Structure

```
nmt-attention/
├── NMT_Attention.ipynb   # Main notebook (all tasks)
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- A **GPU runtime** is strongly recommended (e.g., Google Colab T4)

### Install dependencies

```bash
pip install -r requirements.txt
```

Or let the notebook's first cell handle it automatically.

### Dataset

Download `english_french.csv` from [Kaggle — English to French Small Dataset](https://www.kaggle.com/datasets/rajpulapakura/english-to-french-small-dataset) and upload it to `/content/` in your Colab environment.

---

## Usage

### Google Colab (recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR_USERNAME>/nmt-attention/blob/main/NMT_Attention.ipynb)

> Replace `<YOUR_USERNAME>` with your GitHub handle after uploading.

Make sure to switch to a **GPU runtime**: `Runtime → Change runtime type → T4 GPU`

### Local

```bash
jupyter notebook NMT_Attention.ipynb
```

---
## Model Architecture

```
English sentence
  └── Encoder
        ├── Embedding(eng_vocab_size, 64, mask_zero=True)
        └── LSTM(128, return_sequences=True, return_state=True)
              ├── encoder_outputs  (batch, src_len, 128)  ← for attention
              ├── state_h          (batch, 128)            ← decoder init
              └── state_c          (batch, 128)            ← decoder init

For each decoder timestep t (teacher forcing during training):
  └── DecoderStep
        ├── BahdanauAttention(encoder_outputs, h_t) → context (batch, 128)
        ├── Embedding(token_t)                       → (batch, 64)
        ├── Concat([embedding, context])             → (batch, 192)
        ├── LSTMCell(128)                            → new_h, new_c
        └── Dense(fra_vocab_size, softmax)           → next-token probs
```

**Why a custom training loop?**  
`model.fit()` does not natively support teacher forcing or per-token padding masks. A `tf.GradientTape` loop gives full control over both.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle — English to French Small Dataset](https://www.kaggle.com/datasets/rajpulapakura/english-to-french-small-dataset) |
| Total pairs | ~80,000 sentence pairs |
| Used for training | 40,000 (sampled with `random_state=42`) |
| Format | CSV with `English` and `French` columns |
| Labels | Raw sentence strings, cleaned and tokenised |

**Splits used:**

| Split | Size |
|-------|-----:|
| Train | 32,000 |
| Test  | 8,000 |

---

## Key Concepts

### Bahdanau (Additive) Attention
At each decoder step, attention computes a score between the current decoder hidden state and every encoder hidden state. These scores are softmax-normalised into weights, and a weighted sum of encoder outputs forms the **context vector** fed into the decoder — allowing the model to focus on the most relevant source tokens dynamically.

### Teacher Forcing
During training, the ground-truth previous token is fed as input to the decoder instead of its own prediction. This stabilises gradients and speeds up convergence, at the cost of a gap between training and inference behaviour.

### Padding-Aware Loss
Target sequences are padded to a fixed length with token id `0`. These positions are masked out of the loss computation so the model is not penalised for ignoring them, ensuring only real tokens contribute to gradient updates.

### Greedy Decoding
At inference time, the decoder starts with a `<start>` token and repeatedly picks the highest-probability next token until it predicts `<end>` or reaches the maximum sequence length.

---

## License

This project is for academic purposes (CSC 483). Feel free to use the code for learning.
