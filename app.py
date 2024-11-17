from flask import Flask, request, jsonify, render_template
import tensorflow as tf

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import os
import numpy as np
import pandas as pd
import uuid

from collections import Counter
import matplotlib.pyplot as plt
import base64
import io
import logging
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# Download NLTK resources first
# nltk.download('punkt_tab')
# nltk.download('punkt')  
# nltk.download('stopwords')  
# nltk.download('wordnet')  
# nltk.download('words') 
# nltk.download('averaged_perceptron_tagger_eng')

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Load TensorFlow model
model = tf.keras.models.load_model('model')
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.BinaryCrossentropy())

# Temp File Dirr              
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

# prediction format
def format_prediction(prediction):
    if prediction >= 0.5:
        return "Positive"
    else:
        return "Negative"

# top 10 word usage for multiple classification
def get_top_n_words(text, n=10):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    words = [word for word in tokens if word.isalnum() and word not in stop_words]
    word_counts = Counter(words)
    return word_counts.most_common(n)        


#Singular Predict
@app.route('/')
def index():
    return render_template('index.html')

#Singular Predict Return
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

#Multiple Predict Return
@app.route('/multiple-predict', methods=['POST'])
def multiple_predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        file_uuid = str(uuid.uuid4())  #unique filename
        filename = file_uuid + ".xlsx" #save as excel
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        logging.info(f"File '{filename}' received and saved to '{file_path}'.")

        try:
            df = pd.read_excel(file_path)
            texts = df['content'].tolist()

            predictions = []
            preprocessed_texts = []
            positive_words = []
            negative_words = []

            for x in texts:
                try:
                    preprocessed_text = preprocess_text(x)
                    prediction = model.predict([preprocessed_text])
                    sentiment = format_prediction(prediction) #fixed this part
                    predictions.append(sentiment)
                    preprocessed_texts.append(preprocessed_text)

                    if sentiment == "Positive": #fixed this part
                        positive_words.extend(get_top_n_words(x))
                    elif sentiment == "Negative": #fixed this part
                        negative_words.extend(get_top_n_words(x))

                except Exception as e:
                    logging.error(f"Error during prediction for text '{x}': {e}")  # Log the error
                    predictions.append("Error")
                    preprocessed_texts.append("Error")

            positive_word_counts = Counter(word for word, count in positive_words)
            negative_word_counts = Counter(word for word, count in negative_words)

            result = {
                "predictions": predictions,
                "preprocessed_texts": preprocessed_texts,
                "top_positive_words": positive_word_counts.most_common(10),
                "top_negative_words": negative_word_counts.most_common(10)
            }

            logging.info(f"Prediction results: {result}")

            return jsonify(result)

        except Exception as e:
            logging.error(f"Error reading Excel file or during prediction: {e}")
            return jsonify({"error": f"Error reading Excel file or during prediction: {e}"}), 500

        finally:
            os.remove(file_path)

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/model-evaluation')
def model_evaluation():
    return render_template('model-evaluation.html')

if __name__ == '__main__':
    app.run(debug=True)