# Real-Time ASL Alphabet Translator 🤟

This project is a real-time American Sign Language (ASL) alphabet translator. It uses a live webcam feed to identify hand gestures and translate them into text, allowing users to form words and sentences.

The application is built with a two-stage computer vision pipeline:
1.  **Hand Detection:** A YOLOv8s model detects and isolates the user's hand in the frame.
2.  **Sign Classification:** The cropped hand image is fed into a custom-trained MobileNetV2 model to classify the ASL sign.

## ✨ Features

* **Real-Time Performance:** Uses a multi-threaded webcam stream for a smooth, high-FPS experience.
* **High Accuracy:**
    * Uses the powerful `yolov8s-hand.pt` model for robust hand detection.
    * Employs a custom-trained MobileNetV2 classifier with ~98% validation accuracy.
* **Stable Predictions:** Implements a two-layer "debounce" system to prevent flickering:
    1.  **Confidence Thresholding:** Ignores low-confidence guesses from the classifier.
    2.  **Majority Vote Buffer:** A prediction must be the "majority" in a buffer of 10 frames to be accepted.
* **Sentence Building:** Includes logic for adding letters, spaces (`space`), and deleting characters (`del`).

## 🛠️ Technology Stack

* **Python 3**
* **PyTorch:** For loading and running the classification model.
* **Ultralytics (YOLOv8):** For the hand detection model.
* **OpenCV:** For camera handling and image processing.
* **NumPy:** For numerical operations.
* **tqdm:** For the training script's progress bar.

## 🚀 Getting Started

Follow these instructions to get the project running on your local machine.

### 1. Prerequisites

* Python 3.8 or newer
* A webcam
* (Optional but Recommended) A CUDA-enabled GPU for faster performance.

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/md-hzs-22/ASL_to_Text
    cd YourRepoName
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip freeze > requirements.txt
    pip install -r requirements.txt
    ```

### 3. Models & Dataset

This repository includes the two pre-trained models required to run the application:
* `best_asl_model.pth`: The custom-trained ASL classifier.
* `yolov8s-hand.pt`: The YOLO hand detector.

To **train the model yourself**, you must download the [ASL Alphabet Dataset from Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and place the `asl_alphabet_train` folder in the root of this project.

## 🏃‍♂️ How to Run

There are two main scripts: one to run the app, and one to train a new model.

### 1. Run the Real-Time ASL Translator
To start the live translation, run the `realtime_asl.py` script:
*(**Note:** I've named the final script `realtime_asl.py`. Please rename your script to match.)*

```bash
python realtime_asl.py
```
press `q` to quit the application.

### 2. Train Your Own Model
To re-train the classifier, run the train_asl_model.py script: (Note: I've named the training script train_asl_model.py. Please rename your script to match.)

```bash
python train_asl_model.py
```
This will train a new model and save the best version as `best_asl_model.pth`.

