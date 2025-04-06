import os
import sys
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import logging
from werkzeug.utils import secure_filename
import tempfile
import traceback

# Suppress TensorFlow optimization messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# IMPORTANT: Configure logging to only use stdout (no file handlers)
# This is the critical fix for the Render deployment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # ONLY log to stdout
    ]
)
logger = logging.getLogger(__name__)
logger.info("Starting application with stdout logging")

# Configure TensorFlow to avoid memory issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Found {len(gpus)} GPU(s), configured for memory growth")
    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'development-key')
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logger.info(f"Created upload folder at {app.config['UPLOAD_FOLDER']}")

# Global variables
MODEL = None
CLASS_NAMES = ["Ant", "Bee", "Beetle", "Cockroach", "Fly", "Mosquito", "Spider"]

# Load model function
def load_model():
    global MODEL
    try:
        # Adjust path according to your model location
        model_path = os.path.join(os.getcwd(), 'models', 'pest_detection_model.h5')
        if os.path.exists(model_path):
            MODEL = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully!")
        else:
            logger.warning(f"Model file not found at {model_path}")
            # Create a placeholder model for testing if needed
            MODEL = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(128, 128, 1)),
                tf.keras.layers.Dense(7, activation='softmax')
            ])
            logger.info("Created placeholder model for testing")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False
    return True

# Audio preprocessing function
def preprocess_audio(audio_path):
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050)
        logger.info(f"Loaded audio file: {audio_path}, sample rate: {sr}")
        
        # Extract features - example using mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=128,
            fmax=8000
        )
        
        # Convert to decibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))
        
        # Resize if needed to match model input size
        if mel_spec_db.shape[1] > 128:
            mel_spec_db = mel_spec_db[:, :128]
        else:
            pad_width = 128 - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        
        logger.info(f"Preprocessed audio to shape: {mel_spec_db.shape}")
        
        # Reshape for model input - adjust dimensions based on your model
        features = np.expand_dims(mel_spec_db, axis=-1)
        
        return features, None
    
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        logger.error(traceback.format_exc())
        return None, f"Error processing audio: {str(e)}"

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    logger.info("Accessed index page")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    logger.info("Upload endpoint called")
    if 'file' not in request.files:
        logger.warning("No file part in request")
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        logger.warning("No selected file")
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logger.info(f"Saved file to {file_path}")
        
        # Process the audio file
        features, error = preprocess_audio(file_path)
        
        if error:
            logger.error(f"Error during preprocessing: {error}")
            return jsonify({'error': error})
        
        # Make prediction
        try:
            logger.info("Making prediction")
            predictions = MODEL.predict(np.expand_dims(features, axis=0))[0]
            
            # Get top 3 predictions
            top_indices = predictions.argsort()[-3:][::-1]
            top_pests = [CLASS_NAMES[i] for i in top_indices]
            top_scores = [float(predictions[i]) for i in top_indices]
            
            logger.info(f"Prediction result: {CLASS_NAMES[np.argmax(predictions)]}")
            
            result = {
                'prediction': {
                    'class': CLASS_NAMES[np.argmax(predictions)],
                    'confidence': float(np.max(predictions))
                },
                'top_3': [
                    {'class': pest, 'confidence': score} 
                    for pest, score in zip(top_pests, top_scores)
                ]
            }
            
            return render_template(
                'result.html',
                audio_file=url_for('static', filename=f'uploads/{filename}'),
                result=result
            )
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f"Prediction error: {str(e)}"})
    
    flash('Invalid file type. Please upload WAV, MP3, OGG, or FLAC files.')
    return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    logger.info("API predict endpoint called")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            file.save(tmp.name)
            file_path = tmp.name
        
        # Process the audio file
        features, error = preprocess_audio(file_path)
        
        # Clean up temp file
        try:
            os.unlink(file_path)
        except:
            pass
        
        if error:
            return jsonify({'error': error})
        
        # Make prediction
        try:
            predictions = MODEL.predict(np.expand_dims(features, axis=0))[0]
            
            # Create response
            result = {
                'prediction': {
                    'class': CLASS_NAMES[np.argmax(predictions)],
                    'confidence': float(np.max(predictions))
                },
                'all_classes': {
                    class_name: float(pred) 
                    for class_name, pred in zip(CLASS_NAMES, predictions)
                }
            }
            
            return jsonify(result)
        
        except Exception as e:
            logger.error(f"API prediction error: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f"Prediction error: {str(e)}"})
    
    return jsonify({'error': 'Invalid file type'})

# Health check endpoint for Render
@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

# Initial model loading
@app.before_first_request
def before_first_request():
    logger.info("Loading model before first request")
    load_model()

# For local development
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask application on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)