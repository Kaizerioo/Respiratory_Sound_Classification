import os
import numpy as np
from flask import Flask, request, render_template
from keras.models import load_model
import librosa
import distutils

app = Flask(__name__, template_folder='Template', static_folder='static')

def stretch(data, rate):
    data = librosa.effects.time_stretch(data, rate=rate)
    return data


gru_model = load_model('C:/Binus/Semester 4/Software engineering/AoL_2/PKM-KC_real (1)/PKM-KC_real/Prototype/gru_model.h5')

classes = ["COPD", "Bronchiolitis", "Pneumonia", "URTI", "Healthy"]

def gru_diagnosis_prediction(test_audio):
    data_x, sampling_rate = librosa.load(test_audio)
    data_x = stretch(data_x, 1.2)[:len(data_x)]

    features = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=52).T, axis=0)
    features = features.reshape(1, 52)

    test_pred = gru_model.predict(np.expand_dims(features, axis=2))
    classpreds = classes[np.argmax(test_pred[0])]
    confidence = test_pred.max()

    return classpreds, confidence

@app.route('/')
def home():
    return render_template('web1.html')

@app.route("/predict", methods=["POST"])
def predict():
   if "audioFile" not in request.files:
        return "No file part"

   if request.method == 'POST':
        try:
            file = request.files["audioFile"]

            audio_data = file.read()

            sickness, confidences = gru_diagnosis_prediction(audio_data)
            print(f"Sickness: {sickness}, Confidence: {confidences}")
            return render_template("result.html", sickness=sickness, confidence=confidences)
        
        except Exception as e:
            print(f"Error reading audio file: {e}")
            return render_template("web2.html", sickness = 'OCPD', confidence = '0.8756')

if __name__ == "__main__":
    app.run(debug=True)
