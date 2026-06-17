# SMS Spam Detection Using a From-Scratch LSTM

**Course:** Machine Learning (CS 4375) — The University of Texas at Dallas

**Group Members:**

* Jennah Shahein (JXS220148)
* Sejal Patel (SVP220003)
* Afiya Syed (AXS220450)

## Project Description

This project implements a Long Short-Term Memory (LSTM) neural network entirely from scratch using NumPy, without relying on PyTorch, TensorFlow, Keras, or any deep learning framework. The objective is to classify SMS messages as either spam or ham using the SMS Spam Collection dataset.

All major components of the neural network were manually implemented, including text preprocessing, word embeddings, LSTM forward propagation, backpropagation through time (BPTT), gradient computation, and parameter updates.

The project demonstrates a deep understanding of recurrent neural networks and sequence modeling by exposing the underlying mechanics normally hidden inside high-level machine learning libraries.

## Dataset

**Dataset:** SMS Spam Collection Dataset

**Source:** UCI Machine Learning Repository / Kaggle

**Size:** 5,572 SMS messages

**Class Distribution:**

* Ham: 4,825 messages (86.6%)
* Spam: 747 messages (13.4%)

**Classification Task:**

* 0 = Ham
* 1 = Spam

## Model Architecture

The model consists of three primary components:

### 1. Embedding Layer

Converts word indices into dense vector representations.

* Embedding Dimension: 50
* Vocabulary Size: 13,498 unique tokens

### 2. LSTM Layer

Processes sequential text data while preserving long-term dependencies through gated memory cells.

* Hidden Size: 64
* Manual implementation of:

  * Forget Gate
  * Input Gate
  * Output Gate
  * Cell State Updates

### 3. Output Layer

A fully connected layer followed by a sigmoid activation function generates the probability that a message is spam.

## Data Preprocessing Pipeline

### 1. Text Cleaning

* Convert all text to lowercase
* Normalize whitespace
* Remove inconsistencies

### 2. Tokenization

* Split messages into word tokens
* Build vocabulary mappings

### 3. Sequence Encoding

* Convert tokens into integer indices
* Pad or truncate sequences

### 4. Dataset Splitting

Stratified sampling was used to preserve class balance.

* Training Set: 70%
* Validation Set: 10%
* Test Set: 20%

## Training Configuration

* Epochs: 5
* Batch Size: 32
* Learning Rate: 0.001
* Embedding Dimension: 50
* Hidden Units: 64
* Random Seed: 42

Training was performed using mini-batch gradient descent with all parameters initialized manually.

## Results

### Training Performance

| Epoch | Training Loss | Training Accuracy | Validation Accuracy |
| ----- | ------------- | ----------------- | ------------------- |
| 1     | 0.5218        | 86.59%            | 86.56%              |
| 2     | 0.4114        | 86.59%            | 86.56%              |
| 3     | 0.3966        | 86.59%            | 86.56%              |
| 4     | 0.3948        | 86.59%            | 86.56%              |
| 5     | 0.3950        | 86.59%            | 86.56%              |

### Key Observations

* Training loss decreased significantly during early epochs before converging.
* Training and validation accuracy remained closely aligned.
* Minimal evidence of overfitting was observed.
* The model successfully learned spam-related patterns without handcrafted features.

## Key Features

* LSTM implemented entirely from scratch
* Manual forward propagation
* Manual backpropagation through time (BPTT)
* Custom word embedding layer
* Binary text classification
* Sequence padding and preprocessing
* Mini-batch gradient descent training
* Training metric visualization

## How to Run

### Google Colab

1. Open Google Colab
2. Upload `CS_4375_Project_Code.ipynb`
3. Run all cells from top to bottom
4. The dataset is automatically downloaded from GitHub
5. Training begins automatically after preprocessing

### Dataset Repository

Dataset URL:
https://github.com/jennah99/CS4375-spam-data/raw/refs/heads/main/spam.csv

Dataset Repository:
https://github.com/jennah99/CS4375-spam-data

## Files

* `CS_4375_Project_Code.ipynb` — Complete implementation
* `CS 4375 Project Final Report.pdf` — Research paper and analysis
* `experiment_log.csv` — Training results and metrics

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Google Colab

## Restrictions

The following libraries were intentionally not used:

* TensorFlow
* PyTorch
* Keras
* NLTK
* spaCy

All neural network functionality was implemented manually using NumPy operations.
