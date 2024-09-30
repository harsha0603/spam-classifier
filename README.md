# Spam Classifier ML Project
This project is a **Spam Classification** system that detects whether an email is spam or not based on its content. It uses machine learning and natural language processing (NLP) techniques to preprocess and classify email data.
## Features
- *Text Preprocessing*: Includes tokenization, stemming, and vectorization using TF-IDF.
- *Classification Model*: Implements the Multinomial Naive Bayes algorithm to classify emails.
- *Web Application*: A simple Flask app allows users to input text and receive predictions.
- *Model Persistence*: The trained model and vectorizer are saved using joblib and dill for reuse.
