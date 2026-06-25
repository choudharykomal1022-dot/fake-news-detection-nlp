# Fake News Detection Engine using NLP & Machine Learning

A robust, end-to-end Machine Learning and Natural Language Processing (NLP) pipeline engineered to classify raw news headlines into "Real" or "Fake" categories with high diagnostic precision.

## 🚀 Key Highlights & Performance
* **Optimized Infrastructure:** Resolved Google Colab RAM crashes by converting dense text vectors into memory-efficient **SciPy Sparse Matrices**, reducing memory overhead by over 90%.
* **Predictive Accuracy:** Tuned Logistic Regression model achieved an **95.02% Accuracy** score and a **0.94 Recall score** on unseen validation text data.
* **Production-Ready Inference:** Built a clean local inference pipeline capable of processing and cleaning standalone, raw text headlines for live simulation testing.

## 🛠️ Tech Stack & Skills
* **Languages:** Python
* **Machine Learning:** Scikit-Learn (`sklearn`), Logistic Regression, Multinomial Naive Bayes
* **Natural Language Processing:** `NLTK`, Regular Expressions (`re`), Tokenization, Stopword Removal, Porter Stemming
* **Data & Optimization:** SciPy (Sparse Matrices), Pandas, NumPy, Seaborn, Matplotlib

## 📊 Evaluation & Results
We benchmarked two classification architectures across a dataset of 44,000+ text instances using an 80/20 train-test split:

| Model Framework | Accuracy | Precision | Recall |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (Tuned, C=0.8)** | **95.02%** | **0.96** | **0.94** |
| Multinomial Naive Bayes | 93.36% | 0.93 | 0.95 |

*Logistic Regression was selected as the final production engine due to its superior overall precision and balanced generalization curve.*

## ⚙️ How the Pipeline Works (The Theory)
1. **Text Normalization:** Cleans noise using regex, forces lowercasing, and eliminates non-predictive English stopwords.
2. **Stemming:** Leverages the Porter Stemmer algorithm to reduce words to their base grammatical roots (e.g., "running" to "run").
3. **Vectorization:** Implements `CountVectorizer` with an `ngram_range=(1,3)` to capture multi-word structural phrases (unigrams, bigrams, trigrams) across a 5,000-word feature threshold.
4. **Classification:** Feeds sparse matrix features into the trained models, outputting a binary classification verdict via mapped weights.
