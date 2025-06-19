from flask import Flask, request, jsonify
import time

app = Flask(__name__)

# Mock model responses
responses = [
    "This is a chemical process simulation response.",
    "The optimal temperature for this reaction is 350°C.",
    "Consider adjusting the pressure to improve yield.",
    "The catalyst efficiency can be enhanced by temperature control.",
    "Process optimization suggests increasing flow rate by 15%."
]

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        if not data or 'inputs' not in data:
            return jsonify({"error": "Missing 'inputs' field"}), 400
        
        inputs = data['inputs']
        max_new_tokens = data.get('parameters', {}).get('max_new_tokens', 100)
        
        # Simulate processing time
        time.sleep(1)
        
        # Return mock response
        import random
        response = random.choice(responses)
        
        return jsonify({"generated_text": response})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting simple model server...")
    app.run(host='0.0.0.0', port=80, debug=False) 