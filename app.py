from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained model
model = load_model('/Users/shaunak/Desktop/untitled folder/cnn_model.h5')

# Define the function to predict tumor type
def predict_tumor_type(img, model):
    img = img.resize((224, 224))  # Resize the image to match model input size
    img_array = np.array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input
    
    predictions = model.predict(img_array)  # Get model predictions
    labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    predicted_class = labels[np.argmax(predictions)]  # Get the predicted class
    confidence = predictions[0][np.argmax(predictions)] * 100  # Get the confidence score
    
    return predicted_class, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            img = Image.open(io.BytesIO(file.read()))  # Open the image
            predicted_class, confidence = predict_tumor_type(img, model)  # Predict tumor type
            return jsonify({
                'predicted_class': predicted_class,
                'confidence': confidence
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
