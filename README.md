# 🧠 Mental Health Detection using BERT + Graph Attention Networks (GAT)

## 🚀 Overview

This project builds an advanced **Natural Language Processing (NLP) system** to detect mental health conditions from social media text using:

* Transformer-based models (BERTweet)
* Deep Learning (BiLSTM + Attention)
* Graph Neural Networks (Graph Attention Networks - GAT)

The system captures both **contextual language understanding** and **inter-sample relationships** to improve classification performance.

---

## 🎯 Problem Statement

To develop a robust NLP model that can classify user-generated text into mental health categories, enabling early detection and support using AI-driven insights.

---

## ⚡ Key Features

✔ Transformer-based Text Representation

* BERTweet for domain-specific embeddings

✔ Sequential Modeling

* BiLSTM for capturing contextual dependencies

✔ Attention Mechanism

* Focuses on important words in the sentence

✔ Graph Neural Network (GAT)

* Models relationships between samples in a batch

✔ Class Imbalance Handling

* Weighted loss function

✔ Scalable Training Pipeline

* Efficient batching and GPU support

---

## 🛠 Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Scikit-learn
* Pandas / NumPy

---

## 📊 Model Architecture

1. Input Text → Tokenization (BERT tokenizer)
2. BERT → Contextual embeddings
3. BiLSTM → Sequential feature extraction
4. Attention Pooling → Important token selection
5. Graph Attention Network (GAT) → Relational learning
6. Fully Connected Layer → Classification

---

## 📈 Results

* Achieved strong classification performance on mental health dataset
* Improved results using attention + graph-based learning
* Handles imbalanced datasets effectively

*(Add accuracy, precision, confusion matrix here if available)*

---

## 📸 Sample Outputs

* Training Accuracy per Epoch
* Confusion Matrix
* Classification Report

*(Add screenshots for better presentation)*

---

## ⚡ Quick Demo

```bash id="a8p9x1"
python bert_gat_mental_health.py
```

---

## 📁 Project Structure

```id="n2w9xz"
mental-health-nlp-gat/
│
├── bert_gat_mental_health.py
├── requirements.txt
├── README.md
└── data/
    └── Mental-Health-Twitter.csv
```

---

## 💡 Why This Project Matters

* Combines **Transformers + Graph Neural Networks** (rare skillset)
* Applies AI to **real-world mental health challenges**
* Demonstrates advanced NLP techniques beyond basic classification
* Useful for research, healthcare AI, and social media analysis

---

## 📌 Future Improvements

* Real-time inference system
* Deployment using Streamlit or Flask
* Multi-label classification
* Integration with clinical datasets

---

## ⚠ Disclaimer

This project is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis.

---

## 👨‍💻 Author

**Nishanth M**
🔗 GitHub: https://github.com/Nishanthx7

---
