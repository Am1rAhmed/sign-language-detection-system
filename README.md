# HandTalk – Deep Learning-Based Sign Language Recognition

## Overview

HandTalk is a real-time sign language recognition system that uses MediaPipe landmark extraction and an LSTM-based deep learning model to identify sign language gestures from live webcam input. The application provides real-time predictions through a Flask web interface and achieves **93.28% test accuracy**.

---

## Features

* Real-time sign language recognition using webcam input
* MediaPipe-based pose and hand landmark extraction
* LSTM neural network for sequence-based gesture classification
* Prediction stabilization using confidence thresholds and majority voting
* Flask-powered web application with responsive UI
* Camera start/stop functionality for user control
* Achieved **93.28% test accuracy**

---

## Tech Stack

* Python
* TensorFlow / Keras
* MediaPipe
* OpenCV
* Flask
* NumPy
* Scikit-learn

---

## Model Performance

| Metric        | Value      |
| ------------- | ---------- |
| Test Accuracy | **93.28%** |

---

## Project Structure

```text
handtalk-sign-language-detection/
│
├── app.py
├── utils.py
├── requirements.txt
├── sign_language_model.h5
├── ver_1.ipynb
├── ver_2.ipynb
│
├── training/
│   ├── collect_data.py
│   ├── load_dataset.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── MP_Data/
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── Logs/
```
## Dataset Access

The `MP_Data` folder is not included in this repository due to its large size.

You can download the dataset from Google Drive:

**Google Drive Link:** https://drive.google.com/file/d/1L9MUH5e5Ig9NYiKnITYZEyD4oT-i5syO/view?usp=sharing

After downloading, extract the `MP_Data` folder and place it in the project root directory:

```text
sign-language-detection-system/
│
├── MP_Data/
├── app.py
├── utils.py
└── ...
```

The application and training scripts expect the `MP_Data` directory to be located in the project root.

---
## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/sign-language-detection-system.git

cd sign-language-detection-system
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
python app.py
```

## Dataset Preparation

1. Run the data collection script:

```bash
python collect_data.py
```

2. Train the model:

```bash
python train_model.py
```

3. Evaluate the trained model:

```bash
python evaluate_model.py
```

---

## How It Works

1. MediaPipe extracts pose and hand landmarks from webcam frames.
2. Landmark coordinates are converted into numerical feature vectors.
3. A sequence of landmark frames is fed into an LSTM network.
4. The model predicts the most likely gesture.
5. Confidence thresholds and majority voting are applied to improve prediction stability.
6. Predictions are displayed in real time through the Flask web interface.

---

## Future Improvements

* Expand the gesture vocabulary
* Add sentence-level sign recognition
* Support dynamic sign language translation
* Deploy as a cloud-hosted web application
* Improve model robustness under varying lighting conditions

---

## License

This project is licensed under the MIT License.

---

## Author

Amir Ahmed Siddiqui
