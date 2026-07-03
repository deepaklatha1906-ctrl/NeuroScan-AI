# 🧠 Brain Tumor Detection AI with Grad-CAM Explainability

An explainable deep learning web application that classifies brain MRI scans into four categories: **Glioma**, **Meningioma**, **No Tumor**, and **Pituitary**. Built using MobileNetV2 and Streamlit, this tool not only predicts the tumor type but also visualizes *where* the AI is looking using Grad-CAM heatmaps.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- **Multi-Class Classification:** Accurately identifies Glioma, Meningioma, Pituitary tumors, and healthy scans.
- **Explainable AI (XAI):** Integrated **Grad-CAM** heatmaps to highlight the exact regions of the brain the model focuses on, building trust in medical AI.
- **Interactive Web UI:** A clean, user-friendly Streamlit interface for easy image uploads and instant analysis.
- **Command Line Support:** Standalone Python script for quick, batch, or automated predictions.
- **Transfer Learning:** Powered by MobileNetV2, optimized for high accuracy and efficient CPU/GPU inference.

---

## ️ Tech Stack

- **Language:** Python 3.10+
- **Deep Learning:** TensorFlow, Keras
- **Architecture:** MobileNetV2 (Transfer Learning)
- **Web Framework:** Streamlit
- **Data Processing:** NumPy, Pillow
- **Visualization:** Matplotlib, Seaborn

---

## 📂 Project Structure

```text
brain_tumor/
├── dataset/              # Training and Testing MRI images
│   ├── Training/
│   └── Testing/
├── train.py              # Script to train the MobileNetV2 model
├── predict.py            # Standalone command-line prediction script
├── app.py                # Streamlit web application (with Grad-CAM)
├── best_model.keras      # Trained model weights
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation