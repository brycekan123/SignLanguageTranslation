# ✋ Real-Time Sign Language Gloss Recognition

This project is an end-to-end prototype for **real-time American Sign Language (ASL) gloss prediction** using a webcam. It brings together **MediaPipe**, **LSTM-based deep learning**, and **OpenCV** to detect hand gestures and classify them into ASL glosses (individual words or signs).

---

## 🧠 Project Summary

- The main goal is to **translate ASL signs into English in real time** using only a webcam. While full sentence-level translation is an ambitious long-term goal, this project focuses on **word-level ("gloss") classification**, serving as a stepping stone toward more complex models.

### 🔍 Key Components

- **Dataset Source**: The model is trained on a subset of the [WLASL (Word-Level American Sign Language)](https://github.com/dxli94/WLASL) dataset — a large-scale collection of labeled ASL videos.
  
- **Data Preprocessing**:
  - From the WLASL JSON metadata (`WLASL_v0.3.json`), glosses (sign words) and associated video IDs were extracted.
  - Each video was preprocessed to extract hand landmarks (using **MediaPipe**) and converted into `.npy` files of shape `(90, 126)` — representing 90 frames of left/right hand 3D keypoints (21 landmarks × 3 coordinates × 2 hands).
  - These `.npy` arrays were mapped back to their gloss labels, and filtered to remove invalid or empty sequences.

- **Model Architecture**:
  - A deep **LSTM (Long Short-Term Memory)** model was built using Keras to capture temporal dependencies in hand movement.
  - The model takes in sequences of shape `(90, 126)` and outputs a softmax probability across the full vocabulary of glosses (around 80–100 words in this prototype).
  - Trained using `categorical_crossentropy` with accuracy monitored over 50–300 epochs.

- **Real-Time Inference**:
  - Live webcam feed is processed using **MediaPipe Holistic** to extract hand landmarks in real time.
  - A rolling window of 30 frames is used to feed the model, which then outputs the most likely gloss prediction.
  - Predictions above a confidence threshold are rendered on the video display using OpenCV.

---

## 🎯 Objective

The goal of this project is to create a working prototype that:
- Connects real-time webcam input to meaningful ASL recognition
- Tests the feasibility of gloss-level classification using hand landmark sequences
- Provides a foundation for scaling to sentence-level ASL recognition or larger vocabularies

This project is also a learning exercise in:
- Preprocessing real-world video data into ML-ready formats
- Building and tuning sequence models for temporal data
- Integrating computer vision with machine learning inference in real time

> ⚠️ **Note:** This prototype is still under development.  
> Inspired by Nicholas Renotte's [GitHub Project](https://github.com/nicknochnack/ActionDetectionforSignLanguage) and [Tutorial](https://www.youtube.com/watch?v=doDUihpj6ro&t=8363s) by Nick Nochnack.  
> Current limitations include small dataset size, signer variability, and a narrow vocabulary.

