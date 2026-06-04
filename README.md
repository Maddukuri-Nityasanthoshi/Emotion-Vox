# 🎙️ EmotionVox -  Real-time speech emotion recognition using Deep learning

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)](https://keras.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A real-time speech emotion recognition system using Convolutional Neural Networks (CNN) and Support Vector Machines (SVM), trained on 1,400+ audio samples with MFCC feature extraction — enabling machines to understand human emotional state from voice alone.**

---

## 📌 The Problem

Human emotion is embedded in how we speak — not just what we say. Tone, pitch, rhythm, and energy carry emotional signals that traditional NLP misses entirely. EmotionVox solves this by analysing raw audio and classifying the speaker's emotional state in real time.

**Applications:** Mental health monitoring · Call centre quality analysis · Human-computer interaction · Accessibility tools for emotion-aware AI

---

## 🎯 Emotions Classified

| Emotion | Label | Training Samples |
|---------|-------|-----------------|
| Happy | 😊 | ~200 |
| Sad | 😢 | ~200 |
| Angry | 😠 | ~200 |
| Fearful | 😨 | ~200 |
| Neutral | 😐 | ~200 |
| Disgusted | 🤢 | ~200 |
| Surprised | 😲 | ~200 |

**Total dataset: 1,400+ audio samples**

---

## 🏗️ ML Pipeline

```
Raw Audio (.wav)
      │
      ▼
Audio Preprocessing
(noise reduction · normalization · segmentation)
      │
      ▼
Feature Extraction — MFCC
(Mel-Frequency Cepstral Coefficients)
40 MFCC features per audio sample
      │
      ▼
      ├──► CNN Model (Deep Learning)
      │    Conv1D layers → MaxPooling → Dropout → Dense → Softmax
      │
      └──► SVM Classifier (Traditional ML)
           RBF kernel · GridSearchCV hyperparameter tuning
      │
      ▼
Emotion Prediction + Confidence Score
```

---

## 🧠 Model Architecture

### CNN Architecture
```
Input Layer       → MFCC features (40 × time_steps)
Conv1D (64)       → ReLU activation
MaxPooling1D      → Dimensionality reduction
Conv1D (128)      → ReLU activation
MaxPooling1D      → Dimensionality reduction
Dropout (0.3)     → Overfitting prevention
Flatten
Dense (128)       → ReLU activation
Dropout (0.3)
Output Dense (7)  → Softmax (7 emotion classes)
```

### SVM Classifier
```
Features: Flattened MFCC array
Kernel: RBF (Radial Basis Function)
Tuning: GridSearchCV (C, gamma parameters)
Scaling: StandardScaler preprocessing
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Deep Learning | TensorFlow 2.x, Keras |
| Classical ML | scikit-learn |
| Audio Processing | librosa |
| Feature Extraction | MFCC (Mel-Frequency Cepstral Coefficients) |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Model Saving | joblib (SVM), .h5 / .keras (CNN) |

---

## 📁 Project Structure

```
EmotionVox/
├── data/
│   ├── raw/                # Original .wav audio files
│   └── processed/          # Preprocessed features
├── src/
│   ├── preprocessing.py    # Audio loading and normalization
│   ├── feature_extraction.py  # MFCC extraction
│   ├── cnn_model.py        # CNN architecture and training
│   ├── svm_model.py        # SVM classifier and tuning
│   ├── evaluate.py         # Model evaluation and metrics
│   └── predict.py          # Real-time prediction interface
├── models/
│   ├── cnn_emotionvox.keras   # Trained CNN model
│   └── svm_emotionvox.pkl     # Trained SVM model
├── notebooks/
│   └── EmotionVox_EDA.ipynb   # Exploratory data analysis
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/Maddukuri-Nityasanthoshi/EmotionVox.git
cd EmotionVox
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare dataset
```bash
# Place your .wav audio files in data/raw/
# File naming convention: emotion_speakerID_index.wav
# Example: happy_01_001.wav
```

### 4. Extract features
```bash
python src/feature_extraction.py
```

### 5. Train models
```bash
# Train CNN
python src/cnn_model.py

# Train SVM
python src/svm_model.py
```

### 6. Run real-time prediction
```bash
python src/predict.py --audio path/to/audio.wav
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| CNN | ~85% | ~84% | ~85% | ~84% |
| SVM | ~78% | ~77% | ~78% | ~77% |
| Ensemble | ~87% | ~86% | ~87% | ~86% |

*Evaluated on 20% held-out test split*

---

## 📈 Key Results

- ✅ CNN outperforms SVM by ~7% accuracy on balanced dataset
- ✅ MFCC features with 40 coefficients gave optimal performance
- ✅ Dropout layers (0.3) significantly reduced overfitting
- ✅ Happy and Angry emotions achieve highest classification accuracy
- ✅ Neutral and Sad emotions show most overlap — known challenge in SER literature

---

## 🔬 Feature Extraction — MFCC Explained

MFCC (Mel-Frequency Cepstral Coefficients) mimics how the human ear processes sound — by mapping audio frequencies to the mel scale and computing spectral features. This makes MFCCs ideal for capturing the tonal and rhythmic patterns that distinguish emotional speech.

Each audio sample → 40 MFCC coefficients × time frames → input matrix for CNN/SVM

---

## 🚧 Future Improvements

- [ ] Add real-time microphone input with streaming prediction
- [ ] Expand dataset with RAVDESS + CREMA-D + TESS datasets
- [ ] Experiment with Transformer-based audio models (Wav2Vec 2.0)
- [ ] Build a web interface using Streamlit for live demos
- [ ] Add multilingual support for Hindi and Telugu speech

---

## 📚 Dataset

Trained on a curated dataset of 1,400+ labeled `.wav` audio recordings covering 7 emotion categories. Audio samples are sampled at 22,050 Hz with a minimum duration of 2 seconds per sample.

> If using RAVDESS or similar public datasets, please cite the original authors.

---

## 👩‍💻 Author

**Nitya Santhoshi Maddukuri**  
B.Tech Computer Science Engineering · Mallareddy Engineering College For Women· 2026  
Published Researcher — ICAECT 2026 (Explainable AI / ML)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/nitya-maddukuri)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/Maddukuri-Nityasanthoshi)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
