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
import logging
import io

# Download NLTK resources first
# nltk.download('punkt_tab')
# nltk.download('punkt')  
# nltk.download('stopwords')  
# nltk.download('wordnet')  
# nltk.download('words') 
# nltk.download('averaged_perceptron_tagger_eng')

app = Flask(__name__)

# Load TensorFlow model
model = tf.keras.models.load_model('model')
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.BinaryCrossentropy())

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')              

# Temp File Dirr              
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Static File Dirr              
STATIC_FOLDER = './static'
app.config['STATIC_FOLDER'] = STATIC_FOLDER
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Universal Def for pre-processing
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

#Def for every pre-processing return
def evaluation_preprocess_text(text):
    # 1. Case Folding
    text_lower = text.lower()

    # 2. Removing Non-ASCII Characters
    text_ascii = ''.join(char for char in text_lower if ord(char) < 128)

    # 3. Removing Punctuation
    text_no_punct = text_ascii.translate(str.maketrans('', '', string.punctuation))

    # 4. Stop Word Removal (excluding negation words)
    stop_words = set(stopwords.words('english'))
    negation_words = {"not", "no", "dont", "didnt", "doesnt", "cant", "couldnt", "wont", "wouldnt", "shouldnt", "wasnt", "werent", "havent", "hasnt", "hadnt", "aint"}
    stop_words.difference_update(negation_words)
    tokens = word_tokenize(text_no_punct)
    text_no_stopwords = ' '.join([word for word in tokens if word not in stop_words])

    # 5. Lemmatization
    wnl = WordNetLemmatizer()

    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    # Lemmatize the text_no_stopwords
    tokens = word_tokenize(text_no_stopwords)
    lemmas = [wnl.lemmatize(token, pos=get_wordnet_pos(token)) for token in tokens]
    text_lemmatized = ' '.join(lemmas)

    return [text_lower, text_ascii, text_no_punct, text_no_stopwords, text_lemmatized]   

# universal Def for Prediction Format
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

# Accuracy Calculation
def calculate_accuracy(cm):
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100
    return accuracy    


#Main Page
@app.route('/')
def index():
    return render_template('index.html')

#Preprocessing Page
@app.route('/preprocessing')
def preprocessing():
    return render_template('preprocessing.html')

#Preprocessing Page Return
@app.route('/preprocess', methods=['POST'])
def preprocess():
    if 'file' not in request.files:
        logging.error("No file part")
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    try:
        file_uuid = str(uuid.uuid4())
        filename = file_uuid + ".xlsx"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logging.info(f"File '{filename}' saved to '{file_path}'.")

        try:
            df = pd.read_excel(file_path)

            if 'content' not in df.columns:
                logging.error("Excel file must have a 'content' column")
                return jsonify({"error": "Excel file must have a 'content' column"}), 400

            texts = df['content'].tolist()
            review = []
            preprocessed_texts = []

            for index, x in enumerate(texts):
                try:
                    preprocessed_text = evaluation_preprocess_text(x)
                    if preprocessed_text[0] == "Error":
                        df = df.drop(index) 
                        continue
                    
                    preprocessed_texts.append(preprocessed_text)

                except Exception as e:
                    logging.error(f"Error during prediction for review {index + 1}: {e}")
                    predictions.append("Error")


            result = {
            "preprocessed_texts": preprocessed_texts,
            "reviews": texts,
            }

            logging.info(f"Evaluation results: {result}")

            return jsonify(result)

        except Exception as e:
            logging.exception(f"Error reading Excel file or during prediction: {e}")
            return jsonify({"error": f"Error reading Excel file or during prediction: {e}"}), 500

    except Exception as e:
        logging.exception("An unexpected error occurred: %s", str(e))
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(file_path)
            logging.info(f"Removed temporary file: {file_path}")
        except Exception as e:
            logging.error(f"Error deleting temporary file: {e}")

#Singular Predict Page
@app.route('/single')
def singular():
    return render_template('klasifikasi-single.html')

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


#Multiple Predict Page
@app.route('/klasifikasi-multiple')
def klasifikasi_multiple():
    return render_template('klasifikasi-multiple.html')

#Multiple Predict Return
@app.route('/multiple-predict', methods=['POST'])
def multiple_predict():
    try:
        # Check the request file
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        # Store as temp file
        file_uuid = str(uuid.uuid4())  #unique filename
        filename = file_uuid + ".xlsx" #save as excel
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Now use the multiple_predict logic 
        try:
            df = pd.read_excel(file_path)

            if 'content' not in df.columns:
                logging.error("Excel file must have a 'content' column")
                return jsonify({"error": "Excel file must have a 'content' column"}), 400

            texts = df['content'].tolist()

            review = []
            predictions = []
            positive_words = []
            negative_words = []
            positive_count = 0
            negative_count = 0

            for x in texts:
                try:
                    # Predict
                    preprocessed_text = preprocess_text(x)
                    prediction = model.predict([preprocessed_text])
                    sentiment = format_prediction(prediction)
                    # Store the review and the predict result
                    review.append(x)
                    predictions.append(sentiment)
                    # Count the words    
                    if sentiment == "Positive":
                        positive_count += 1
                        positive_words.extend(get_top_n_words(x))
                    elif sentiment == "Negative":
                        negative_count += 1
                        negative_words.extend(get_top_n_words(x))

                except Exception as e:
                    print(f"Error during prediction for text '{x}': {e}")
                    predictions.append("Error")
                    preprocessed_texts.append("Error")

            positive_word_counts = Counter(word for word, count in positive_words)
            negative_word_counts = Counter(word for word, count in negative_words)

            result = {
                "predictions": predictions,
                "reviews": review,
                "top_positive_words": positive_word_counts.most_common(10),
                "top_negative_words": negative_word_counts.most_common(10),
                "total_positive": positive_count,
                "total_negative": negative_count
            }

            logging.info(f"Prediction successful: {result}")

            return jsonify(result)

        except Exception as e:
            return jsonify({"error": f"Error reading Excel file or during prediction: {e}"}), 500

        finally: #remove temp file
            os.remove(file_path)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

#Scraping Predict Page
@app.route('/klasifikasi-scraping')
def klasifikasi_scraping():
    return render_template('klasifikasi-scraping.html')

#Scraping Predict Return
@app.route('/scraping-predict', methods=['POST'])
def scraping_predict():
    try:
        num_reviews = int(request.form.get('num_reviews', 100))
        logging.info(f"Scraping {num_reviews} reviews")

        try:
            from google_play_scraper import reviews, Sort

            results, _ = reviews(
                'com.discord',  # Replace with the desired app ID
                lang='en',
                country='id',
                sort=Sort.NEWEST,
                count=num_reviews,
            )
            logging.info(f"Scraping completed successfully. Received {len(results)} reviews.")
            
            reviews_data = [r['content'] for r in results]
            
            logging.info(f"Extracted reviews: {reviews_data}")

            predictions = []
            positive_words = []
            negative_words = []
            positive_count = 0
            negative_count = 0

            for x in reviews_data:
                try:
                    preprocessed_text = preprocess_text(x)
                    prediction = model.predict([preprocessed_text])
                    sentiment = format_prediction(prediction)

                    predictions.append(sentiment)

                    if sentiment == "Positive":
                        positive_count += 1
                        positive_words.extend(get_top_n_words(x))
                    elif sentiment == "Negative":
                        negative_count += 1
                        negative_words.extend(get_top_n_words(x))

                except Exception as e:
                    logging.error(f"Error during prediction for review '{x}': {e}")
                    return jsonify({"error": f"Error processing or predicting reviews: {e}"}), 500
                    predictions.append("Error")

            positive_word_counts = Counter(word for word, count in positive_words)
            negative_word_counts = Counter(word for word, count in negative_words)

            result = {
                "predictions": predictions,
                "reviews": reviews_data,
                "top_positive_words": positive_word_counts.most_common(10),
                "top_negative_words": negative_word_counts.most_common(10),
                "total_positive": positive_count,
                "total_negative": negative_count
            }

            logging.info(f"Prediction successful: {result}")

            return jsonify(result)

        except Exception as e:
            logging.error(f"Error processing or predicting reviews: {e}")
            return jsonify({"error": f"Error processing or predicting reviews: {e}"}), 500


        except Exception as e:
            logging.error(f"Error during scraping: {e}")
            return jsonify({"error": f"Error during scraping: {e}"}), 500


    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)