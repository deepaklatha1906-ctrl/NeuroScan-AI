"""
Brain Tumor Detection AI with Grad-CAM Heatmap
"""
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.cm as cm

# ==========================================
# GRAD-CAM FUNCTIONS
# ==========================================
def get_last_conv_layer(model):
    """Dynamically find the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if hasattr(layer, 'output') and len(layer.output.shape) == 4: 
            return layer.name
    raise ValueError("Could not find a convolutional layer.")

def compute_gradcam(model, img_array, class_index):
    """Compute the Grad-CAM heatmap for a given image and class."""
    last_conv_layer_name = get_last_conv_layer(model)
    
    # 1. Create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Ensure input is a float32 tensor for GradientTape
    img_tensor = tf.cast(img_array, tf.float32)

    # 2. Get the gradient of the top predicted class with regard to the output feature map
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        
        # CRITICAL FIX: Explicitly tell the tape to watch the intermediate tensor
        tape.watch(conv_outputs)
        
        loss = predictions[:, class_index]

    # 3. Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    
    if grads is None:
        raise ValueError("Gradients are None. Cannot compute Grad-CAM.")

    # 4. Weight the feature maps by the gradients
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)[0]

    # 5. Process the CAM (ReLU and normalize)
    cam = tf.maximum(cam, 0)
    cam = cam / tf.reduce_max(cam)
    return cam.numpy()

def create_heatmap_overlay(original_img, heatmap, alpha=0.4):
    """Overlay the heatmap on the original image."""
    # Resize heatmap to match original image size
    heatmap_resized = np.array(Image.fromarray(heatmap).resize((original_img.width, original_img.height)))
    
    # Apply a colormap (JET) to make it colorful
    heatmap_colored = np.uint8(255 * cm.jet(heatmap_resized)[..., :3])
    
    # Convert original PIL image to numpy array
    img_array = np.array(original_img)
    
    # Superimpose the heatmap on the image
    superimposed_img = heatmap_colored * alpha + img_array
    return Image.fromarray(np.clip(superimposed_img, 0, 255).astype(np.uint8))

# ==========================================
# STREAMLIT APP
# ==========================================
st.set_page_config(page_title="Brain Tumor AI", page_icon="🧠", layout="wide")

@st.cache_resource
def load_model():
    model_path = 'best_model.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

st.title("🧠 Brain Tumor Detection AI")
st.markdown("---")

with st.sidebar:
    st.header("ℹ️ Information")
    st.write("""
    **Categories:**
    - Glioma
    - Meningioma
    - No Tumor
    - Pituitary
    
    **Model:** MobileNetV2  
    **Accuracy:** ~82%
    """)
    st.warning("⚠️ For educational purposes only. Consult a doctor for diagnosis.")

uploaded_file = st.file_uploader("Upload Brain MRI Scan", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Uploaded Image")
        original_image = Image.open(uploaded_file).convert('RGB')
        st.image(original_image, use_column_width=True)
    
    if st.button("🔬 Analyze Now", type="primary"):
        with st.spinner("Analyzing and generating attention map..."):
            model = load_model()
            
            if model is None:
                st.error("❌ Model not found!")
            else:
                # Preprocess for model
                img_resized = original_image.resize((224, 224))
                img_array = np.array(img_resized)
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                predictions = model.predict(img_array, verbose=0)
                pred_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                
                class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
                result = class_names[pred_class]
                
                # Generate Grad-CAM Heatmap
                try:
                    heatmap = compute_gradcam(model, img_array, pred_class)
                    overlay_img = create_heatmap_overlay(original_image, heatmap, alpha=0.5)
                    show_heatmap = True
                except Exception as e:
                    show_heatmap = False
                    st.error(f"Could not generate heatmap: {e}")
                
                with col2:
                    st.subheader("📊 Analysis Results")
                    
                    if result == "No Tumor":
                        st.success(f"### ✅ {result}")
                    else:
                        st.error(f"### ⚠️ {result} Detected")
                    
                    st.write(f"**Confidence:** {confidence:.2f}%")
                    
                    st.markdown("---")
                    st.write("**All Probabilities:**")
                    for cls, prob in zip(class_names, predictions[0]):
                        st.write(f"{cls}: {prob*100:.2f}%")
                    
                    if show_heatmap:
                        st.markdown("---")
                        st.subheader("🔥 AI Attention Map (Grad-CAM)")
                        st.write("The red/yellow areas show where the AI is focusing to make its decision.")
                        st.image(overlay_img, use_column_width=True)
                    
                    if result != "No Tumor":
                        st.warning("Please consult a neurologist immediately.")

else:
    st.info("👆 Please upload an MRI scan image to begin")