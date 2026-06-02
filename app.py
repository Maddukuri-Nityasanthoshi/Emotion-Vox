import streamlit as st
import librosa
import numpy as np
import json

# Dummy emotion detection function (Replace with actual model)
def predict_emotion(features):
    emotions = ["happy", "sad", "angry", "neutral"]
    return np.random.choice(emotions)  # Random prediction for now

st.title("Speech Emotion Recognition")

def analyze_audio(audio_file):
    try:
        # Load audio file
        y, sr = librosa.load(audio_file, sr=None)

        # Extract audio features (MFCC)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        feature_vector = np.mean(mfccs, axis=1)  # Convert to a single feature vector

        # Predict emotion
        emotion = predict_emotion(feature_vector)
        return emotion

    except Exception as e:
        return f"Error processing audio: {str(e)}"

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file:
    # Analyze the uploaded audio
    emotion = analyze_audio(uploaded_file)
    
    # Display result
    st.json({"By analysing your audio,predicted emotion is": emotion})

