import streamlit as st
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.api.models import load_model 
import webbrowser
import datetime


class Chatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    
        self.words = pickle.load(open("words.pkl", "rb"))
        self.classes = pickle.load(open("classes.pkl", "rb"))
        self.model = load_model("medical_chatbot_with_sentiment.h5")

    def clean_sentence(self, sentence):
        """Clean and tokenize the sentence."""
        return [self.lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence)]

    def bag_o_words(self, sentence):
        sentence_words = self.clean_sentence(sentence)
        bag = [1 if word in sentence_words else 0 for word in self.words]
        return np.array(bag)


    def predict_class(self, sentence):
        bow = self.bag_o_words(sentence)
        bow = np.reshape(bow, (1, 72479)) 
        res = self.model.predict(bow)[0]  

        ERROR_THRESHOLD = 0.5
   
        results = [[i, r] for i, r in enumerate(res) if (np.array(r) > ERROR_THRESHOLD).any()]
        results.sort(key=lambda x: x[1], reverse=True)

        return [{"intent": self.classes[r[0]], "probability": str(r[1])} for r in results]


    def get_response(self, intents_list):
        """Return a response based on the predicted intent."""
        if not intents_list:
            return "I'm not sure, please rewrite."
        tag = intents_list[0]["intent"]
        return f"Bot response based on intent: {tag}"
       

    def get_bot_response(self, user_message):
        """Generate a bot response based on user input."""
        if user_message.lower() in ["exit", "quit", "bye"]:
            return "Goodbye!"
        elif user_message.lower().startswith("search "):
            query = user_message[7:]
            webbrowser.open(f"https://www.google.com/search?q={query}")
            return f"I have opened the web search for '{query}'"
        elif user_message.lower() == "time":
            return f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}."
        else:
            ints = self.predict_class(user_message)
            return self.get_response(ints)

chatbot = Chatbot()

            
st.title("Medquad-bot")

user_message = st.text_input("You:", "")

if user_message:
    bot_response = chatbot.get_bot_response(user_message)
    
    st.text_area("Conversation", value=f"You: {user_message}\nBot: {bot_response}", height=300)

if st.button("Help"):
    help_text = """Welcome to the bot!
    Type your questions or type 'exit' or 'quit' to end the conversation.
    You can also search the web by typing 'search <your query>'.
    """
    st.info(help_text)


if st.button("Save Chat"):
  
    filename = f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w") as f:
        f.write(f"You: {user_message}\n")
        f.write(f"Bot: {bot_response}\n")
    st.success(f"Chat saved to {filename}")
