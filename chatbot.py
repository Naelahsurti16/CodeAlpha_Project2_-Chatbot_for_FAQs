import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import X, questions, answers, vectorizer

nlp = spacy.load("en_core_web_sm")

def find_answer(user_question):
    user_question_vec = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_question_vec, X)
    closest = np.argmax(similarities)
    return answers[closest]

def chatbot():
    print("Welcome to the FAQ chatbot! Type 'exit' to end the conversation.")
    while True:
        user_question = input("You: ")
        if user_question.lower() == "exit":
            print("Goodbye!")
            break
        answer = find_answer(user_question)
        print("Bot:", answer)

if __name__ == "__main__":
    chatbot()
