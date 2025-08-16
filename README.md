# Biomedical-Text-Classification-ML-
A machine learning pipeline for biomedical text classification using TF-IDF, Chi-square feature selection, and multiple classifiers with ensemble methods. Includes preprocessing, EDA, model training, and evaluation to deliver accurate categorization of clinical and biomedical text data.
## Project Overview
This project focuses on classifying biomedical literature and text data into meaningful categories using advanced machine learning techniques. It aims to support researchers by automating the organization and analysis of biomedical text.

## Features
- Preprocessing and cleaning of biomedical text.
- Feature extraction using TF-IDF and word embeddings.
- Implementation of multiple machine learning algorithms (Naive Bayes, SVM, Random Forest).
- Evaluation using Accuracy, Precision, Recall, and F1-Score.
- Easy-to-use pipeline for training and predicting new text data.

📂 Dataset

The dataset consists of biomedical clinical text samples labeled into multiple categories.

File: Clinical Text Data.csv

Columns:

Text → Biomedical/clinical text content.

Label → Category of the text.

⚠ The dataset used is not publicly provided here. You may replace it with any biomedical text dataset with similar structure.


📈 Results

The models were evaluated using multiple metrics.
Model	Accuracy	Precision (Macro Avg)	Recall (Macro Avg)	F1-Score (Macro Avg)
Multinomial Naive Bayes	0.9274	0.93	0.93	0.93
Logistic Regression	0.9460	0.95	0.95	0.95
Linear SVM	0.9674	0.97	0.97	0.97
Random Forest	0.9907	0.99	0.99	0.99
Artificial Neural Network (ANN)	0.9674	0.97	0.97	0.97
Voting Classifier	0.9394	0.94	0.94	0.94
Stacking Classifier	1.0000	1.00	1.00	1.00


📧 Contact

For queries, contact: [Mahin] – [Siam.mahin2001@gmail.com]
