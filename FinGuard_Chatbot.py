


# Import necessary libraries
import nltk #Natural Language Toolkit for processing human language data.
import string #Used for string operations.
import random #To generate random responses.
import json # For handling JSON data.
import pickle #For serializing and de-serializing Python objects.
import numpy as np #For numerical operations.
import tensorflow as tf #For building and training neural network models.

from sklearn.feature_extraction.text import TfidfVectorizer  #For machine learning functions like TF-IDF and cosine similarity.
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model # High-level API for building and training models.


# Download necessary NLTK data for tokenization and lemmatization
nltk.download('punkt')  # Download the tokenizer data
nltk.download('wordnet')  # Download the lemmatizer data


# Define preprocessing functions for FAQ handling
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    """
    Lemmatizes each token in the list.
    Uses NLTK's WordNetLemmatizer to reduce words to their base forms.
    """
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    """
    Normalizes the text by converting to lowercase, removing punctuation, and lemmatizing.
    Prepares the text for further processing by tokenizing, removing punctuation, and applying lemmatization.
    """
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Load FAQs from a text file
def load_faqs(filename):
    """
    Loads FAQs from a file and splits them into questions and answers.
    Assumes the file has a format where each line contains a question and answer separated by a comma.
    """
    questions = []
    answers = []
    with open(filename, 'r', encoding='latin-1') as file:
        for line in file:
            if ',' in line:
                question, answer = line.strip().split(',', 1)
                questions.append(question)
                answers.append(answer)
    return questions, answers

# Load FAQs and normalize questions
sent_tokens, answer_tokens = load_faqs("BankFAQs.doc")
sent_tokens = [q.lower() for q in sent_tokens]  # Convert questions to lowercase



#Load intents for greeting and intent classification
intents = json.loads(open('intents.json').read())
# Load the intents JSON file containing patterns and responses for different intents.


# Preprocess intents data for training the model
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']  # Characters to ignore during processing

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and prepare lists of unique words and classes
words = [lemmer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save processed words and classes for later use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


# Prepare training data for the intent classification model
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

# Shuffle and convert to numpy array for training
random.shuffle(training)
training = np.array(training)

train_x = training[:, :len(words)]  # Feature set
train_y = training[:, len(words):]  # Label set
# Build and train the intent classification model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

# Compile and train the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')  # Save the trained model
print('Intent model training complete')



#  Load the trained model
model = load_model('chatbot_model.h5')



# Define utility functions for handling chatbot interactions
def clean_up_sentence(sentence):
    """
    Tokenizes and lemmatizes the sentence for prediction.
    Prepares the input sentence to be compatible with the model.
    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """
    Creates a bag of words representation for the input sentence.
    Transforms the sentence into a fixed-size vector based on the vocabulary.
    """
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """
    Predicts the class of the sentence using the trained model.
    Returns a list of intent predictions with associated probabilities.
    """
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
    """
    Retrieves a response based on the predicted intent.
    Matches the predicted intent with the responses in the intents JSON.
    """
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def faq_response(user_response):
    """
    Generates a response based on the FAQ dataset using cosine similarity.
    Matches the user query to the most similar question in the FAQ dataset.
    """
    robo_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens + [user_response])
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])

    idx = vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]

    if req_tfidf == 0:
        robo_response = "I am sorry! I don't understand you."
    else:
        robo_response = answer_tokens[idx]

    return robo_response


# Cell 11: Chatbot interaction loop
print("Hello! I am Finbot. Start typing your text to talk to me. For ending the conversation type 'bye'!")

while True:
    user_response = input("").lower()

    if user_response == 'bye':
        print('Finbot: Goodbye!')
        break
    elif user_response in ['thanks', 'thank you']:
        print('Finbot: You are welcome!')
        break
    else:
        ints = predict_class(user_response)
        if ints:
            response = get_response(ints, intents)
        else:
            response = faq_response(user_response)
        print(f'Finbot: {response}')
