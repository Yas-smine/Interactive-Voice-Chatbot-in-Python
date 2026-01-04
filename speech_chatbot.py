# Importing necessary libraries
import nltk # For NLP tasks (tokenization)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer  # Text vectorization
import string
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import speech_recognition as sr      # For converting speech to text

# Loading and Preprocessing Data:
# Load the text file and preprocess the data
with open('chatbot.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')
# Tokenize the text into sentences
sentences = sent_tokenize(data)

# Define a function to preprocess each sentence
def preprocess(text):
    text = text.lower() #Converts text to lowercase and removes punctuation.
    return text.translate(str.maketrans("", "", string.punctuation))

# Chatbot response function
# -----------------------------
def chatbot_response(user_input):
    """
    Generates a chatbot response using cosine similarity
    between the user input and the corpus sentences.
    """
    
    # Preprocess user input
    user_input = preprocess(user_input)
    
    # Temporarily add user input to sentence list
    sentences.append(user_input)

    # Convert text data into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)

    # Compute cosine similarity between user input and all sentences
    similarity = cosine_similarity(tfidf[-1], tfidf)
    
    # Get index of the most similar sentence (excluding user input)
    index = similarity.argsort()[0][-2]
    score = similarity[0][index]

    # Remove user input from sentence list
    sentences.pop()

    # If similarity score is zero, chatbot doesn't understand
    if score == 0:
        return "Sorry, I didn't understand that."
    else:
        return sentences[index]

# -----------------------------
# Speech-to-text function
# -----------------------------
def speech_to_text():
    """
    Captures audio from microphone and converts it into text.
    """
    recognizer = sr.Recognizer()
    
    # Use microphone as audio source
    with sr.Microphone() as source:
        st.info("🎤 Listening... Please speak")
        audio = recognizer.listen(source)

    try:
        # Convert speech to text using Google Speech API
        text = recognizer.recognize_google(audio)
        return text

    except sr.UnknownValueError:
        # When speech is not understood
        return "Could not understand the audio"

    except sr.RequestError:
        # When speech service is unavailable
        return "Speech recognition service error"

# -----------------------------
# Streamlit User Interface
# -----------------------------

# App title
st.title("🎙️ Speech Enabled Chatbot")

# Allow user to choose input type
input_type = st.radio("Choose input type:", ("Text", "Speech"))

# -------- Text Input Mode --------
if input_type == "Text":
    user_text = st.text_input("Enter your message:")
    
    if st.button("Send"):
        if user_text:
            response = chatbot_response(user_text)
            st.success(f"🤖 Chatbot: {response}")

# -------- Speech Input Mode --------
else:
    if st.button("Speak"):
        speech_text = speech_to_text()
        st.write(f"🗣️ You said: {speech_text}")
        
        response = chatbot_response(speech_text)
        st.success(f"🤖 Chatbot: {response}")