from pydoc import doc
from xml.dom.minidom import Document
import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
nltk.download('stopwords')

df=pd.read_csv("IMDB Dataset.csv")
document=df['review'].values
stopWordsEngilish=set(stopwords.words('english'))
labels = df['sentiment'].map({'positive': 1, 'negative': 0}).values
def cleanText(document):
    document=document.lower()
    document=re.sub(r"[^a-z\s]","",document)
    textList=document.split()
    filterWord=[word for word in textList if word not in stopWordsEngilish]
    return " ".join(filterWord)
sentences = [cleanText(sentence) for sentence in document]

 
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

word_index = tokenizer.word_index
max_len = 200

padded = pad_sequences(sequences, padding='post',maxlen=max_len)
print(padded)
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=200))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))  # Çünkü binary sınıflama

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(padded, np.array(labels), epochs=13,batch_size=256,validation_split=0.2)

test_sentences = [
    "I absolutely loved this movie",
    "The film was fantastic and uplifting",
    "Great acting and brilliant story",
    "A masterpiece of modern cinema",
    "This movie exceeded my expectations",
    "Everything about this film was perfect",
    "I hated every minute of this movie",
    "Terrible acting and a boring plot",
    "Painfully slow and badly directed",
    "A total disaster, don’t bother watching"
]
test_sentences = [cleanText(sentence) for sentence in test_sentences]

test_seq = tokenizer.texts_to_sequences(test_sentences)
test_pad = pad_sequences(test_seq, maxlen=padded.shape[1], padding='post')

predictions = model.predict(test_pad)

for i, sentence in enumerate(test_sentences):
    print(f"{sentence} -> {'Pozitif' if predictions[i] > 0.5 else 'Negatif'}")
