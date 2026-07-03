"""
Brain Tumor Training Script (Fresh Start)
Saves model in modern .keras format to prevent loading errors.
"""
import tensorflow as tf
from tensorflow.keras import layers, applications
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# CONFIGURATION
# ==========================================
import os
# Automatically finds the 'dataset' folder inside the current script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 4
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ==========================================
# 1. LOAD DATA
# ==========================================
def load_data():
    train_dir = os.path.join(DATASET_DIR, 'Training')
    test_dir = os.path.join(DATASET_DIR, 'Testing')
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, 
        label_mode='categorical', shuffle=True, seed=42
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, 
        label_mode='categorical', shuffle=False
    )
    
    # CRITICAL FIX: Apply preprocessing OUTSIDE the model graph
    def preprocess(images, labels):
        images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
        return images, labels

    train_ds = train_ds.map(preprocess).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).cache().prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds

# ==========================================
# 2. BUILD PURE MODEL
# ==========================================
def build_model():
    # Base model
    base_model = applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, 
        weights='imagenet', pooling='avg'
    )
    base_model.trainable = False # Freeze base
    
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)

# ==========================================
# 3. TRAIN & SAVE
# ==========================================
def main():
    print("🧠 Starting Brain Tumor Training...\n")
    
    train_ds, test_ds = load_data()
    model = build_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save in modern .keras format (Fixes all loading bugs)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', 
                                 save_best_only=True, verbose=1)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        train_ds, validation_data=test_ds, epochs=EPOCHS,
        callbacks=[checkpoint, early_stop]
    )
    
    # Save final model
    model.save('final_model.keras')
    print("\n✅ Training Complete! Models saved as .keras files.")

if __name__ == "__main__":
    main()