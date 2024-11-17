from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import numpy as np
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# Download NLTK resources first
# nltk.download('punkt')  
# nltk.download('stopwords')  
# nltk.download('wordnet')  
# nltk.download('words') 
# nltk.download('averaged_perceptron_tagger_eng')

# Load your TensorFlow model
model = tf.keras.models.load_model('model')
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.BinaryCrossentropy())

app = Flask(__name__)

#Singluar Predict
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        preprocessed_text = preprocess_text(text)
        prediction = model.predict([preprocessed_text])
        sentiment = format_prediction(prediction[0][0])

        return jsonify({'prediction': sentiment, 'preprocessing': preprocessed_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


#Multiple Predict
@app.route('/klasifikasi-multiple')
def klasifikasi_multiple():
    return render_template('klasifikasi-multiple.html')


@app.route('/multiple-predict', methods=['POST'])
def multiple_predict():
    # Implement multiple prediction logic here.  You'll likely need to
    # receive a list of texts from the POST request.
    return jsonify({'message': 'Multiple prediction not yet implemented'}), 501


@app.route('/model-evaluation', methods=['POST'])
def model_evaluation():
    return render_template('model-evaluation.html')

# Universal Def for pre-processing and post-processing       
def preprocess_text(text):
    text = text.lower()
    text = ''.join(char for char in text if ord(char) < 128)
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    negation_words = {"not", "no", "dont", "didnt", "doesnt", "cant", "couldnt", "wont", "wouldnt", "shouldnt", "wasnt", "werent", "havent", "hasnt", "hadnt", "aint"}
    stop_words.difference_update(negation_words)
    tokens = word_tokenize(text)
    text = ' '.join([word for word in tokens if word not in stop_words])
    wnl = WordNetLemmatizer()

    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    lemmas = [wnl.lemmatize(token, pos=get_wordnet_pos(token)) for token in tokens]
    return ' '.join(lemmas)


def format_prediction(prediction):
    if prediction >= 0.5:
        return "Positive"
    else:
        return "Negative"


if __name__ == '__main__':
    app.run(debug=True)