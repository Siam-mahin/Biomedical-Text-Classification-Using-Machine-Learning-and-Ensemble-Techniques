# streamlit_app.py
import streamlit as st
import pandas as pd
import string
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# ================================
# Load dataset
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("Clinical Text Data.csv",encoding='ISO-8859-1')  # Change to your CSV filename
    return df

df = load_data()

# ================================
# Text Preprocessing Function
# ================================
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

df['clean_text'] = df['Text'].apply(clean_text)

# ================================
# Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['Label'], test_size=0.2, random_state=42)

# ================================
# Pipeline: TF-IDF + Logistic Regression
# ================================
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=500))
])

model_pipeline.fit(X_train, y_train)

# ================================
# Accuracy
# ================================
y_pred = model_pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ================================
# Streamlit UI
# ================================
st.title("Text Label Prediction App")
st.write(f"Model Accuracy: **{acc*100:.2f}%**")

user_input = st.text_area("Enter text to predict label:")

if st.button("Predict"):
    if user_input.strip() != "":
        cleaned_input = clean_text(user_input)
        prediction = model_pipeline.predict([cleaned_input])[0]
        st.success(f"Predicted Label: **{prediction}**")
    else:
        st.warning("Please enter some text before predicting.")

# Show dataset
if st.checkbox("Show sample data"):
    st.dataframe(df.head())
