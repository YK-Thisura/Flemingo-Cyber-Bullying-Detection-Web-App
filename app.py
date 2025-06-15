# pip install flask joblib numpy tensorflow nltk scikit-learn

from flask import Flask, request, render_template
import joblib
import numpy as np
import re
import tensorflow as tf
from nltk.corpus import stopwords
import nltk

# Download once at the start
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Load model and other files
model = tf.keras.models.load_model('cyberbullying_model.h5')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

app = Flask(__name__)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\d+", "", text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    # --> Initialize variables for the template
    prediction_text = None
    prediction_class = None

    if request.method == 'POST':
        tweet = request.form['tweet']
        cleaned = clean_text(tweet)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect.toarray())
        predicted_label = label_encoder.inverse_transform([np.argmax(pred)])[0]
        
        # --> Set the text to be displayed.
        # We also format it to be more readable (e.g., "not_cyberbullying" becomes "Not Cyberbullying")
        prediction_text = predicted_label.replace('_', ' ').title()
        
        # If the prediction is NOT cyberbullying, we use the 'positive' class (green).
        # For any type of cyberbullying, we use the 'negative' class (red).
        if predicted_label == 'not_cyberbullying':
            prediction_class = 'positive'
        else:
            prediction_class = 'negative'

    # --> Pass the new variables to the template
    return render_template('index.html', 
                           prediction_text=prediction_text, 
                           prediction_class=prediction_class)

if __name__ == '__main__':
    app.run(debug=True)
      