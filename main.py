import random
import pickle
import numpy as np
import tensorflow as tf
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob  # For sentiment analysis

lemmatizer = WordNetLemmatizer()

print("Loading the CSV file...")
df = pd.read_csv(r'medqaudsmall.csv')
print("CSV file loaded.")


words = []
classes = []
documents = []
ignoreletters = ["?", ",", ".", "@", ";","(",')']


sentiments = ["positive", "neutral", "negative"]

print("Processing the CSV data...")
for index, row in df.iterrows():
    question = row['question']
    answer = row['answer']
    
    if not isinstance(question, str):
        question = str(question)

  
    sentiment = TextBlob(question).sentiment.polarity  
    
   
    if sentiment > 0.1:
        sentiment_label = "positive"
    elif sentiment < -0.1:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

   
    wordlist = nltk.word_tokenize(question)
    words.extend(wordlist)
    documents.append((wordlist, answer, sentiment_label))  
    
   
    if answer not in classes:
        classes.append(answer)

print(f"Processed {len(df)} rows from the CSV.")


print("Lemmatizing words and cleaning data...")
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreletters]


print("Saving words and classes to pickle files...")
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))
print("Words and classes saved.")


print("Preparing the training data...")
training = []
outputempty = [0] * len(classes)
sentiment_empty = [0] * len(sentiments)

for document in documents:
    bag = []
    question = document[0]
    question = [lemmatizer.lemmatize(word.lower()) for word in question]
    
   
    for word in words:
        bag.append(1 if word in question else 0)
    
    outputrow = list(outputempty)
    outputrow[classes.index(document[1])] = 1
    
    sentiment_row = list(sentiment_empty)
    sentiment_row[sentiments.index(document[2])] = 1
    
    training.append(bag + outputrow + sentiment_row)

print(f"Prepared {len(training)} training examples.")

print("Shuffling and converting the training data to numpy arrays...")
random.shuffle(training)
training = np.array(training)

print("Splitting the training data into features and labels...")
trainx = training[:, :len(words)]
trainy = training[:, len(words):(len(words) + len(classes))]
train_sentiment = training[:, (len(words) + len(classes)):]

print("Training data prepared.")

print("Defining the neural network model...")
input_layer = tf.keras.layers.Input(shape=(len(trainx[0]),))

x = tf.keras.layers.Dense(128, activation="relu")(input_layer)
x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)

intent_output = tf.keras.layers.Dense(len(trainy[0]), activation="softmax", name="intent_output")(x)

sentiment_output = tf.keras.layers.Dense(len(train_sentiment[0]), activation="softmax", name="sentiment_output")(x)

model = tf.keras.models.Model(inputs=input_layer, outputs=[intent_output, sentiment_output])

print("Compiling the model...")
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(
    loss={'intent_output': 'categorical_crossentropy', 'sentiment_output': 'categorical_crossentropy'},
    optimizer=sgd,
    #metrics=['accuracy']
)
print("Model compiled.")

print("Training the model...")
hist = model.fit(
    trainx, 
    {'intent_output': trainy, 'sentiment_output': train_sentiment},  # Labels for both outputs
    epochs=250, 
    batch_size=50, 
    verbose=1
)

print("Saving the trained model...")
model.save("medical_chatbot_with_sentiment.h5")
print("Model saved.")

print("done")
