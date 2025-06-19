from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

app = Flask(__name__)

# Global model and tokenizer
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen1.5-1.8B-Chat")
    
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    print(f"Model loaded successfully")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/generate', methods=['POST'])
def generate():
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json()
    if not data or 'inputs' not in data:
        return jsonify({"error": "Missing 'inputs' field"}), 400
    
    inputs = data['inputs']
    max_new_tokens = data.get('parameters', {}).get('max_new_tokens', 100)
    
    # Tokenize and generate
    input_ids = tokenizer.encode(inputs, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = generated_text[len(inputs):].strip()
    
    return jsonify({"generated_text": response_text})

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=80) 