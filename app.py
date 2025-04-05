from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
import os
import numpy as np
import tensorflow as tf
import librosa
from scipy import signal
import logging
import uuid
from datetime import datetime
import io
import tempfile
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/pest_detection_web.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants - must match training parameters
SAMPLE_RATE = 16000
DURATION = 2.5
N_MELS = 64
HOP_LENGTH = 256
N_FFT = 1024
CONFIDENCE_THRESHOLD = 0.30

# Model paths - can be overridden by environment variables
MODEL_PATH = os.getenv('MODEL_PATH', 'models/pest_detection_model.h5')
CLASS_NAMES_PATH = os.getenv('CLASS_NAMES_PATH', 'models/class_names.npy')
MODEL_URL = os.getenv('MODEL_URL', '')
CLASS_NAMES_URL = os.getenv('CLASS_NAMES_URL', '')

# Pest information dictionary
PEST_INFO = {
    "Asian tiger mosquito": {
        "scientific_name": "Aedes albopictus",
        "description": "Invasive mosquito species with distinctive black and white stripes.",
        "impact": "Vector for diseases including dengue, chikungunya, and Zika viruses.",
        "frequency_range": "400-800 Hz",
        "sound_pattern": "High-pitched whining sound with frequency of 500-700 Hz"
    },
    "Caribbean fruit fly": {
        "scientific_name": "Anastrepha suspensa",
        "description": "Yellow-brown fruit fly with dark wing markings.",
        "impact": "Damages citrus and tropical fruits.",
        "frequency_range": "200-450 Hz",
        "sound_pattern": "Rapid wing-beat frequencies around 350 Hz"
    },
    "Fire ants": {
        "scientific_name": "Solenopsis invicta",
        "description": "Aggressive red ants that build large mound nests.",
        "impact": "Damage crops, electrical equipment, and deliver painful stings.",
        "frequency_range": "100-300 Hz",
        "sound_pattern": "Low frequency stridulation sounds"
    },
    "Termites": {
        "scientific_name": "Various species",
        "description": "Social insects that feed on wood and plant material.",
        "impact": "Cause structural damage to buildings and wooden structures.",
        "frequency_range": "50-250 Hz",
        "sound_pattern": "Head-banging sounds when alarmed, clicking and rustling"
    },
    "Asian longhorned beetle": {
        "scientific_name": "Anoplophora glabripennis",
        "description": "Large black beetle with white spots and long antennae.",
        "impact": "Invasive pest that kills hardwood trees by boring into the wood.",
        "frequency_range": "65-220 Hz",
        "sound_pattern": "Gnawing and chewing sounds when larvae feed on wood"
    },
    "Black vine weevil": {
        "scientific_name": "Otiorhynchus sulcatus",
        "description": "Black beetle with pear-shaped body and short snout.",
        "impact": "Damages ornamental plants and small fruits; larvae feed on roots.",
        "frequency_range": "80-250 Hz",
        "sound_pattern": "Quiet clicking and scraping sounds"
    },
    "Mediterranean fruit fly": {
        "scientific_name": "Ceratitis capitata",
        "description": "Fruit fly with yellowish body and dark markings on wings.",
        "impact": "Attacks over 250 types of fruits and vegetables.",
        "frequency_range": "200-600 Hz",
        "sound_pattern": "Wing-beat frequencies around 300-400 Hz"
    },
    "Butterfly": {
        "scientific_name": "Various species",
        "description": "Insects with large, often colorful wings and slender bodies.",
        "impact": "Mostly beneficial as pollinators, but some species can damage crops in larval stage.",
        "frequency_range": "30-100 Hz",
        "sound_pattern": "Mostly silent, occasional low frequency wing movements"
    },
    "June beetle": {
        "scientific_name": "Phyllophaga spp.",
        "description": "Medium to large beetles with reddish-brown coloration.",
        "impact": "Adults feed on leaves of trees and shrubs; larvae (white grubs) damage roots of grasses.",
        "frequency_range": "70-150 Hz",
        "sound_pattern": "Buzzing flight sound around 100 Hz"
    },
    "Queensland fruit fly": {
        "scientific_name": "Bactrocera tryoni",
        "description": "Reddish-brown fruit fly with yellow markings.",
        "impact": "Major pest of fruit crops in Australia and Pacific islands.",
        "frequency_range": "200-400 Hz",
        "sound_pattern": "Wing vibrations around 300 Hz"
    }
}

# Default information for pests not in the dictionary
DEFAULT_PEST_INFO = {
    "scientific_name": "Not available",
    "description": "A pest species detected by sound analysis.",
    "impact": "May cause damage to agricultural or structural systems.",
    "frequency_range": "Unknown",
    "sound_pattern": "Specific sound pattern not documented"
}

# Initialize model
model = None
class_names = None

def download_model_files():
    """Download model files if URLs are provided and files don't exist."""
    os.makedirs('models', exist_ok=True)
    
    # Check if model URL is provided
    if MODEL_URL and not os.path.exists(MODEL_PATH):
        try:
            logger.info(f"Downloading model from {MODEL_URL}")
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            logger.info(f"Model downloaded successfully to {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
    
    # Check if class names URL is provided
    if CLASS_NAMES_URL and not os.path.exists(CLASS_NAMES_PATH):
        try:
            logger.info(f"Downloading class names from {CLASS_NAMES_URL}")
            response = requests.get(CLASS_NAMES_URL)
            with open(CLASS_NAMES_PATH, 'wb') as f:
                f.write(response.content)
            logger.info(f"Class names downloaded successfully to {CLASS_NAMES_PATH}")
        except Exception as e:
            logger.error(f"Failed to download class names: {str(e)}")

def load_model():
    """Load the TensorFlow model and class names."""
    global model, class_names
    
    # First try to download model files if needed
    download_model_files()
    
    try:
        # Check if files exist
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model file not found at {MODEL_PATH}")
            return False
            
        if not os.path.exists(CLASS_NAMES_PATH):
            logger.warning(f"Class names file not found at {CLASS_NAMES_PATH}")
            return False
        
        # Load model and class names
        model = tf.keras.models.load_model(MODEL_PATH)
        class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)
        logger.info(f"Model loaded with {len(class_names)} classes")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def denoise_audio(audio):
    """Apply simple noise reduction technique."""
    try:
        # Calculate noise threshold (assuming first 0.1s is background noise)
        noise_sample = audio[:int(SAMPLE_RATE * 0.1)]
        if len(noise_sample) > 0:  
            noise_threshold = np.mean(np.abs(noise_sample)) * 2
            
            # Apply soft thresholding
            denoised = np.copy(audio)
            denoised[np.abs(denoised) < noise_threshold] *= 0.1
            return denoised
        return audio  # Return original if empty slice
    except:
        return audio  # Return original if any error occurs

def process_audio(audio_file):
    """Process audio file to mel spectrogram with enhanced preprocessing."""
    try:
        # Try to use pydub for more robust file handling
        import io
        from pydub import AudioSegment
        import tempfile
        
        # Save to temporary file if the input is BytesIO
        if isinstance(audio_file, io.BytesIO):
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"temp_audio_{uuid.uuid4()}.wav")
            with open(temp_path, 'wb') as f:
                f.write(audio_file.getvalue())
            file_path = temp_path
        else:
            file_path = audio_file
        
        try:
            # First try with librosa directly
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        except Exception as e:
            logger.warning(f"Librosa failed to load audio, trying with pydub: {str(e)}")
            # If librosa fails, try with pydub
            try:
                # Load with pydub (handles many formats)
                sound = AudioSegment.from_file(file_path)
                # Convert to mono and proper sample rate
                sound = sound.set_channels(1).set_frame_rate(SAMPLE_RATE)
                # Export to standard format
                temp_converted = os.path.join(tempfile.gettempdir(), f"converted_{uuid.uuid4()}.wav")
                sound.export(temp_converted, format="wav")
                # Now try loading with librosa again
                audio, sr = librosa.load(temp_converted, sr=SAMPLE_RATE, duration=DURATION)
                # Clean up temp file
                os.remove(temp_converted)
            except Exception as e2:
                logger.error(f"Both librosa and pydub failed: {str(e2)}")
                return None, None, None
            
        # Delete temporary file if created
        if isinstance(audio_file, io.BytesIO) and os.path.exists(temp_path):
            os.remove(temp_path)
            
        # Check for empty audio
        if audio is None or len(audio) == 0:
            logger.warning(f"Empty audio data")
            return None, None, None
        
        # Apply noise reduction
        audio = denoise_audio(audio)
        
        # Pad or truncate to fixed duration
        target_length = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Check for NaN or Inf values
        if np.isnan(audio).any() or np.isinf(audio).any():
            logger.warning(f"Audio contains NaN or Inf values")
            # Replace NaN/Inf with zeros
            audio = np.nan_to_num(audio)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
            fmin=50,
            fmax=8000
        )
        
        # Check for all-zero spectrogram
        if np.all(mel_spec == 0):
            logger.warning(f"All zero mel spectrogram")
            mel_spec_norm = np.random.uniform(0, 0.01, (N_MELS, mel_spec.shape[1]))
            return mel_spec_norm, audio, 0
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        if mel_spec_db.max() != mel_spec_db.min():
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        else:
            mel_spec_norm = np.zeros_like(mel_spec_db)
        
        # Compute dominant frequency
        dominant_freq = extract_dominant_frequency(audio, SAMPLE_RATE)
        
        return mel_spec_norm, audio, dominant_freq
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return None, None, None

def extract_dominant_frequency(audio, sr):
    """Extract the dominant frequency from audio signal."""
    try:
        # Use Welch's method to estimate power spectral density
        freqs, psd = signal.welch(audio, sr, nperseg=1024)
        
        # Check for valid PSD
        if len(psd) == 0 or np.all(psd == 0) or np.isnan(psd).any() or np.isinf(psd).any():
            return 0
            
        # Find the frequency with maximum energy
        dominant_freq_idx = np.argmax(psd)
        dominant_freq = freqs[dominant_freq_idx]
        
        return dominant_freq
    except:
        return 0  # Return 0 if extraction fails

def generate_feature_variations(mel_spec):
    """Generate slight variations of the input feature for ensemble prediction."""
    variations = [mel_spec]
    
    try:
        # Add slight noise
        noise_variation = mel_spec.copy()
        noise = np.random.normal(0, 0.01, mel_spec.shape)
        noise_variation = np.clip(noise_variation + noise, 0, 1)
        variations.append(noise_variation)
        
        # Small time shift
        shift_variation = mel_spec.copy()
        shift = 2
        shift_variation = np.roll(shift_variation, shift, axis=1)
        shift_variation[:, :shift] = 0
        variations.append(shift_variation)
    except:
        # If variations fail, just return the original
        return [mel_spec]
    
    return variations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/setup')
def setup():
    model_loaded = model is not None and class_names is not None
    return render_template('setup.html', model_loaded=model_loaded)

@app.route('/load-model', methods=['GET'])
def reload_model():
    success = load_model()
    return redirect(url_for('setup', success=success))

@app.route('/api/analyze', methods=['POST'])
def analyze_audio():
    try:
        # Check if model is loaded
        if model is None or class_names is None:
            if not load_model():
                return jsonify({"error": "Failed to load model"}), 500
        
        # Check if request contains file
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        # Save uploaded file for reference
        upload_dir = os.path.join(app.root_path, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        file_path = os.path.join(upload_dir, f"{file_id}.wav")
        audio_file.save(file_path)
        audio_file.seek(0)  # Reset file pointer
        
        # Process audio
        mel_spec, audio, dominant_freq = process_audio(audio_file)
        
        if mel_spec is None:
            return jsonify({"error": "Failed to process audio"}), 400
        
        # Generate variations for ensemble prediction
        variations = generate_feature_variations(mel_spec)
        
        # Prepare variations for prediction
        variation_inputs = [np.expand_dims(var, axis=[0, -1]) for var in variations]
        
        # Make ensemble predictions
        all_predictions = []
        for X in variation_inputs:
            try:
                # Handle NaN/Inf values
                if np.isnan(X).any() or np.isinf(X).any():
                    logger.warning("Input contains NaN or Inf values, replacing with zeros")
                    X = np.nan_to_num(X)
                
                pred = model.predict(X, verbose=0)
                all_predictions.append(pred[0])
            except Exception as e:
                logger.error(f"Prediction error with variation: {str(e)}")
                # Continue with other variations
                continue
        
        # If all predictions failed, return error
        if len(all_predictions) == 0:
            return jsonify({"error": "All prediction attempts failed"}), 500
        
        # Average predictions from all variations
        avg_predictions = np.mean(all_predictions, axis=0)
        
        # Get top predictions
        top_indices = np.argsort(avg_predictions)[-3:][::-1]
        
        # Format detection results
        detections = []
        for i, idx in enumerate(top_indices):
            pest_name = str(class_names[idx])
            confidence = float(avg_predictions[idx])
            
            # Get pest info
            pest_info = PEST_INFO.get(pest_name, DEFAULT_PEST_INFO)
            
            # Add to detections
            detections.append({
                "pest": pest_name,
                "confidence": confidence,
                "scientific_name": pest_info["scientific_name"],
                "description": pest_info["description"],
                "impact": pest_info["impact"],
                "frequency_range": pest_info["frequency_range"],
                "sound_pattern": pest_info["sound_pattern"],
                "dominant_freq": int(dominant_freq)
            })
        
        # Log the detection
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        primary_pest = detections[0]["pest"] if detections else "Unknown"
        primary_conf = detections[0]["confidence"] if detections else 0
        
        logger.info(f"{timestamp} - File: {file_id}.wav - Detected: {primary_pest} (Confidence: {primary_conf:.2f})")
        
        # Return JSON response with detection results
        return jsonify({
            "success": True,
            "timestamp": timestamp,
            "detections": detections,
            "dominant_frequency": int(dominant_freq),
            "file_id": file_id
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_audio: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    model_loaded = model is not None and class_names is not None
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Try to load model on startup
    load_model()
    # Run the Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
