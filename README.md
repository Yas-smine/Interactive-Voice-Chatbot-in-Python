# 🎙️ Speech-Enabled Chatbot using NLP and Streamlit

## 📌 Project Title
**Deep Learning & Neural Networks Checkpoint – Speech Enabled Chatbot**

---

## 🎯 Objective
The objective of this project is to build a **speech-enabled chatbot** that can interact with users using **text or voice input**.  
The chatbot:
- Converts **speech to text** using a speech recognition algorithm  
- Processes user input using **Natural Language Processing (NLP)**  
- Generates responses using **TF-IDF and cosine similarity**  
- Is deployed using a **Streamlit web interface**

---

## 🧠 Technologies & Libraries Used
- **Python**
- **NLTK** – text preprocessing and tokenization
- **Scikit-learn** – TF-IDF vectorization and cosine similarity
- **SpeechRecognition** – speech-to-text conversion
- **Streamlit** – interactive web application

---

## 🖥️ How the Application Works

1. The user selects **Text** or **Speech** input.

2. **Text Input Mode**:
   - The input is directly processed by the chatbot.

3. **Speech Input Mode**:
   - The user's voice is recorded via microphone.
   - Speech is transcribed into text.

4. The chatbot then:
   - Preprocesses the input.
   - Computes TF-IDF vectors.
   - Uses cosine similarity to find the most appropriate response.

5. The chatbot response is displayed on the screen.

