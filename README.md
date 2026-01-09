# Fake News Detection using LSTM

## Overview

This project implements a **Fake News Detection** system using **Deep Learning (LSTM)** with **TensorFlow/Keras**. The goal is to classify news articles as **real or fake** based on their textual content.

The model uses word embeddings followed by an LSTM network to capture sequential and contextual information from text data.

---

## Dataset

* **Total samples:** 23,059
* **Features:**

  * `text` – News article text
  * `category` – Label (0 = Fake, 1 = Real)

### Dataset Structure

```
<class 'pandas.core.frame.DataFrame'>
Index: 23059 entries
Columns:
- text (object)
- category (int64)
```

---

## Model Architecture

The model is built using `Sequential` API in TensorFlow.

### Architecture Details

* **Embedding Layer**

  * Vocabulary Size: `vocab_size`
  * Embedding Dimension: 64
  * Input Length: 500

* **LSTM Layer**

  * Units: 128
  * Dropout: 0.2
  * Recurrent Dropout: 0.2

* **Dense Output Layer**

  * Units: 1
  * Activation: Sigmoid

### Model Summary

```
Embedding (None, 500, 64)   -> 640,000 params
LSTM (None, 128)           -> 98,816 params
Dense (None, 1)            -> 129 params
Total Parameters           -> ~739K
```

---

## Model Compilation

* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam
* **Metric:** Accuracy

```python
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

---

## Training Details

* **Epochs:** 10
* **Batch Processing:** 577 steps per epoch
* **Training Time:** ~6 minutes per epoch

### Training & Validation Performance

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
| ----- | -------------- | ------------ | ---------- | -------- |
| 1     | 0.741          | 0.754        | 0.573      | 0.558    |
| 5     | 0.755          | 0.754        | 0.557      | 0.558    |
| 10    | 0.757          | 0.754        | 0.555      | 0.558    |

The model converges early and remains stable across epochs.

---

## Evaluation Results

### Classification Report

```
              precision    recall  f1-score   support

Fake (0)       0.00       0.00      0.00      1133
Real (1)       0.75       1.00      0.86      3479

Accuracy                           0.75      4612
Macro Avg       0.38       0.50      0.43
Weighted Avg    0.57       0.75      0.65
```

### Key Observations

* Overall accuracy: **75%**
* Model strongly predicts **Real News**
* Poor performance on **Fake News class** due to **class imbalance**

---

## Visualization

* Accuracy and Loss curves show **stable learning**
* No severe overfitting observed
* Validation accuracy remains constant

---

## Limitations

* Severe class imbalance (Fake news underrepresented)
* Low recall for Fake News (class 0)
* Binary accuracy is misleading without balanced metrics

---

## Possible Improvements

* Apply **class weighting** or **SMOTE**
* Use **Bidirectional LSTM**
* Increase embedding dimension
* Try **Transformer-based models (BERT)**
* Optimize threshold instead of default 0.5

---

## Tech Stack

* Python
* TensorFlow / Keras
* Pandas
* NumPy
* Matplotlib

---

## Use Case

* Social media content filtering
* News credibility analysis
* Automated misinformation detection systems

---

## Conclusion

This project demonstrates a baseline deep learning approach for fake news detection using LSTM. While the model achieves reasonable accuracy, further improvements are required to handle class imbalance and improve fake news recall.

---

## Author

Developed as a Deep Learning / NLP project for Fake News Classification.
