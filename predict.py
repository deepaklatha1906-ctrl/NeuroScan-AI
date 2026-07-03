"""
Brain Tumor Prediction Script (Fresh Start)
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import sys
import os

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = 'best_model.keras' # Use the modern format!
IMG_SIZE = (224, 224)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ==========================================
# PREDICT FUNCTION
# ==========================================
def predict_tumor(img_path):
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print(f"Processing image: {img_path}")
    # Load and resize
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    
    # CRITICAL FIX: Apply preprocessing manually here
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    # Results
    print("\n" + "="*50)
    print(f"🧠 RESULT: {img_path}")
    print("="*50)
    print(f"🔬 PREDICTION: {CLASS_NAMES[predicted_idx].upper()}")
    print(f"📊 CONFIDENCE: {confidence:.2f}%")
    print("="*50 + "\n")
    
    # Show Image
    plt.figure(figsize=(5, 5))
    plt.imshow(image.load_img(img_path, target_size=IMG_SIZE))
    plt.title(f"Prediction: {CLASS_NAMES[predicted_idx].upper()}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    plt.show()

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        print("Example: python predict.py dataset/Testing/glioma/Te-gl_10.jpg")
        sys.exit(1)
        
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
    else:
        predict_tumor(img_path)