from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)


try:
 
    model_path = 'final_plant_disease_model.keras'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully.")
    else:
        print(f"⚠️ Warning: Model not found at {model_path}. Please train and save the model first.")
        model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy'
]

def preprocess_image(image_bytes):
    """Preprocesses the image to match the model's input requirements."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224)) # IMG_SIZE used during training
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    return img_array


@app.route('/', methods=['GET'])
def index():
    # Render the beautiful GSAP frontend
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'AI Model is not loaded. Please ensure final_plant_disease_model.keras exists.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    try:
  
        image_bytes = file.read()
        
       
        processed_image = preprocess_image(image_bytes)
        
        # Get AI Predictions
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        predicted_class_name = CLASS_NAMES[predicted_class_idx]
        
    
        human_readable_class = predicted_class_name.replace('___', ' - ').replace('_', ' ')
        
        return jsonify({
            'prediction': human_readable_class,
            'confidence': f"{confidence * 100:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)
