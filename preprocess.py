from faq_data import faq_data  # Import the faq_data variable from faq_data.py
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

def preprocess_data(faq_data):
    questions = [item[0] for item in faq_data]
    answers = [item[1] for item in faq_data]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions)

    return X, questions, answers, vectorizer

X, questions, answers, vectorizer = preprocess_data(faq_data)
