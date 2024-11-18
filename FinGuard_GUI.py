import streamlit as st
import nltk
import string
import random
import json
import pickle
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model

# Download necessary NLTK data
nltk.download('punkt')  # Download the tokenizer models
nltk.download('wordnet')  # Download the lemmatizer models

# Load intents and model
# Read intents from a JSON file
intents = json.loads(open('intents.json').read())

# Load word and class data from pickled files
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the pre-trained chatbot model
model = load_model('chatbot_model.h5')

# Function to load FAQs from a text file
def load_faqs(filename):
    questions = []
    answers = []
    with open(filename, 'r', encoding='latin-1') as file:
        for line in file:
            if ',' in line:
                # Split each line into question and answer
                question, answer = line.strip().split(',', 1)
                questions.append(question)
                answers.append(answer)
    return questions, answers

# Load FAQs and convert them to lowercase for consistency
sent_tokens, answer_tokens = load_faqs("BankFAQs.doc")
sent_tokens = [q.lower() for q in sent_tokens]

# Preprocessing functions for text data
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    """Lemmatize the list of tokens"""
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    """Normalize the text by removing punctuation and lemmatizing"""
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the sentence"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Convert the sentence into a bag of words representation"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predict the class of the given sentence using the model"""
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    """Get a response from the intents based on the predicted class"""
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def faq_response(user_response):
    """Get a response based on the FAQ dataset using TF-IDF"""
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens + [user_response])
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])

    idx = vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]

    if req_tfidf == 0:
        return "I am sorry! I don't understand you."
    else:
        return answer_tokens[idx]

# Streamlit UI
st.title("Finance chatbot - NBC Bank")  # Set the title of the Streamlit app
st.write("Hello! I am Finbot.🤖")
st.write("Ask me any banking-related questions or just say 'hi' to start!")  # Welcome message

# Initialize session state to store chat history and user input
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Function to handle user input and generate a response
def handle_input():
    user_response = st.session_state.user_input.lower()  # Get and convert user input to lowercase

    # Determine response based on user input
    if user_response in ['bye', 'thanks', 'thank you']:
        if user_response == 'bye':
            response = "Goodbye!"
        else:
            response = "You are welcome!"
    else:
        ints = predict_class(user_response)
        if ints:
            response = get_response(ints, intents)
        else:
            response = faq_response(user_response)

    # Add user message and bot response to chat history
    st.session_state.chat_history.append(f"You: {st.session_state.user_input}")
    st.session_state.chat_history.append(f"Finbot: {response}")

    # Clear the input field
    st.session_state.user_input = ""

# User input field with callback to handle input and response
st.text_input("You: ", value=st.session_state.user_input, key="user_input", on_change=handle_input, placeholder="Type your message here...")

# Display chat history in reverse order
for chat in reversed(st.session_state.chat_history):
    st.write(chat)  # Write each message in the chat history to the Streamlit app