from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

app = Flask(__name__)
CORS(app)

# Global variables
model = None
tokenizer = None
device = None

def load_model():
    global model, tokenizer, device
    
    device = torch.device('cpu')  # Railway free tier uses CPU
    model_name = "abishektiwari/Nepali-Bert-offensiveDetection"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return True

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400
    
    text = data['text']
    
    # Tokenize and predict
    encoding = tokenizer(text, truncation=True, padding='max_length', 
                        max_length=128, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    confidence = float(predictions[0][1].item())
    is_offensive = bool(predictions.argmax().item())
    
    return jsonify({
        'success': True,
        'input_text': text,
        'is_offensive': is_offensive,
        'confidence': round(confidence, 4),
        'flag': '🚩' if is_offensive else '✅'
    })

if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
