from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the Haar cascade, model, and music data
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
music_data = pd.read_csv('data_moods.csv')[['name', 'artist', 'mood', 'popularity']]

def recommend_songs(emotion):
    if emotion in ['Disgust']:
        mood = 'Sad'
    elif emotion in ['Happy', 'Sad']:
        mood = 'Happy'
    elif emotion in ['Fear', 'Angry']:
        mood = 'Calm'
    else:
        mood = 'Energetic'
    
    songs = music_data[music_data['mood'] == mood]
    songs = songs.sort_values(by='popularity', ascending=False).head(5).reset_index(drop=True)
    return songs.to_dict(orient='records')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image']

    encoded_data = image_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    print(f"[DEBUG] Faces Detected: {len(faces)}")

    emotion = None
    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float32') / 255
        roi_gray = roi_gray.reshape(1, 48, 48, 1)
        predictions = emotion_model.predict(roi_gray)
        print(f"[DEBUG] Prediction Raw Output: {predictions}")
        emotion = emotion_labels[np.argmax(predictions)]
        print(f"[DEBUG] Detected Emotion: {emotion}")
        break

    if emotion:
        songs = recommend_songs(emotion)
        return jsonify({'emotion': emotion, 'songs': songs})
    else:
        return jsonify({'emotion': None, 'songs': []})

if __name__ == '__main__':
    app.run(debug=True)
