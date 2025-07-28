# app.py - Updated Flask API for Hugging Face Model
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import pandas as pd
from datetime import datetime
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
tokenizer = None
device = None
model_loading_error = None

def load_model():
    """Load the trained model from Hugging Face Hub"""
    global model, tokenizer, device, model_loading_error
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Your Hugging Face model
        model_name = "abishektiwari/Nepali-Bert-offensiveDetection"
        
        logger.info(f"Loading model from Hugging Face: {model_name}")
        
        # Load tokenizer and model from HF Hub
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
        
        logger.info("✅ Model loaded successfully from Hugging Face!")
        return True
        
    except Exception as e:
        model_loading_error = str(e)
        logger.error(f"❌ Error loading model: {str(e)}")
        logger.error("Make sure to update model_name with your HF repo!")
        return False

def detect_offensive_text(text):
    """Detect if Nepali text is offensive"""
    # Check if model is loaded
    if model is None or tokenizer is None:
        return {
            'error': 'Model not loaded yet. Please wait or check /health endpoint.',
            'is_offensive': False,
            'confidence': 0.0,
            'severity': 'ERROR'
        }
    
    try:
        if not text or not text.strip():
            return {
                'error': 'Empty text provided',
                'is_offensive': False,
                'confidence': 0.0,
                'severity': 'NONE'
            }
        
        # Tokenize input
        encoding = tokenizer(
            text.strip(),
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Extract results
        confidence = float(predictions[0][1].item())  # Confidence for offensive class
        is_offensive = bool(predictions.argmax().item())
        
        # Determine severity level
        if confidence > 0.8:
            severity = "HIGH"
        elif confidence > 0.5:
            severity = "MEDIUM"
        elif confidence > 0.3:
            severity = "LOW"
        else:
            severity = "MINIMAL"
        
        return {
            'is_offensive': is_offensive,
            'confidence': round(confidence, 4),
            'severity': severity,
            'flag_emoji': '🚩' if is_offensive else '✅'
        }
        
    except Exception as e:
        logger.error(f"Error in detection: {str(e)}")
        return {
            'error': str(e),
            'is_offensive': False,
            'confidence': 0.0,
            'severity': 'ERROR'
        }

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'message': 'Nepali Offensive Language Detection API',
        'status': 'running',
        'model_status': 'loaded' if model is not None else 'loading' if model_loading_error is None else 'failed',
        'endpoints': {
            'health': 'GET /',
            'detect': 'POST /detect',
            'batch': 'POST /batch_detect'
        },
        'example': {
            'url': '/detect',
            'method': 'POST',
            'body': {'text': 'तपाईंको नेपाली पाठ यहाँ'}
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_loading_error': model_loading_error,
        'device': str(device) if device else 'unknown',
        'timestamp': datetime.now().isoformat(),
        'message': 'Nepali Offensive Detection API is running'
    })

@app.route('/detect', methods=['POST'])
def detect_offensive():
    """Main detection endpoint for Postman testing"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model is still loading. Please wait and try again.',
                'model_error': model_loading_error
            }), 503  # Service Unavailable
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON body provided',
                'example': {'text': 'तपाईंको नेपाली पाठ'}
            }), 400
        
        if 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "text" field in request body',
                'example': {'text': 'तपाईंको नेपाली पाठ'}
            }), 400
        
        text = data['text']
        
        if not text or not text.strip():
            return jsonify({
                'success': False,
                'error': 'Empty text provided'
            }), 400
        
        # Detect offensive content
        result = detect_offensive_text(text)
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
        
        # Return successful response
        response = {
            'success': True,
            'input_text': text,
            'result': {
                'is_offensive': result['is_offensive'],
                'confidence': result['confidence'],
                'severity': result['severity'],
                'flag': result['flag_emoji']
            },
            'message': 'Offensive content detected! 🚩' if result['is_offensive'] else 'Content appears clean ✅',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Detection completed - Offensive: {result['is_offensive']}, Confidence: {result['confidence']}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Error in detect endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/batch_detect', methods=['POST'])
def batch_detect():
    """Batch detection endpoint"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model is still loading. Please wait and try again.'
            }), 503
        
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "texts" field in request body',
                'example': {'texts': ['पहिलो पाठ', 'दोस्रो पाठ']}
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                'success': False,
                'error': '"texts" field must be an array'
            }), 400
        
        results = []
        for i, text in enumerate(texts):
            if text and text.strip():
                detection_result = detect_offensive_text(text.strip())
                results.append({
                    'index': i,
                    'text': text.strip(),
                    'is_offensive': detection_result.get('is_offensive', False),
                    'confidence': detection_result.get('confidence', 0.0),
                    'severity': detection_result.get('severity', 'UNKNOWN'),
                    'flag': detection_result.get('flag_emoji', '❓')
                })
        
        return jsonify({
            'success': True,
            'total_processed': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error in batch_detect endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

# Load model in background (non-blocking)
def init_model():
    """Initialize model in background"""
    logger.info("🚀 Starting model loading in background...")
    load_model()

if __name__ == '__main__':
    print("🚀 Starting Nepali Offensive Detection API...")
    print("📡 Model will load in background...")
    
    # Start model loading in background
    import threading
    model_thread = threading.Thread(target=init_model)
    model_thread.daemon = True
    model_thread.start()
    
    # Get port from environment (Render provides this)
    port = int(os.environ.get('PORT', 5000))
    
    # Start Flask server immediately (don't wait for model)
    print("🌐 Starting Flask server...")
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    # For production servers like gunicorn
    # Load model when module is imported
    init_model_thread = threading.Thread(target=init_model)
    init_model_thread.daemon = True
    init_model_thread.start()