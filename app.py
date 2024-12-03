import logging
import requests
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from unsloth import FastLanguageModel
# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

def initialize_model():
    """Initialize model and tokenizer with error handling"""
    global model, tokenizer
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "a-hamdi/NGILlama3-merged",
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )
        # model_name = "a-hamdi/NGILlama3-merged"
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info("Model a-hamdi/NGILlama3-merged loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def LLMrequest(resp):
    """Send response to external service with robust error handling"""
    url = "http://127.0.0.1:10000/response"
    
    try:
        response = requests.post(url, json={"response": resp}, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        logger.info('Response sent successfully')
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during request: {e}")
        return False

def Prompt(data):
    """Generate system prompt with medical news classification instructions"""
    return f'''
    <SYSTEM>
    You are an advanced medical professional capable of accurately classifying medical news articles. 
    Provide a comprehensive JSON analysis evaluating the credibility of the input.

    Examples are provided in the previous system description.

    Instructions:
    <instructions>
    A - Respond ONLY with the JSON output.
    B - JSON must contain:
        1. "medical": "True"/"False" (medical relevance)
        2. "news": "True"/"False" (news or information status)
        3. "label": "Fake"/"Doubtful"/"Trustworthy"
        4. "reasoning": Explanation for the label
        5. "sources": List of reference links
    C - No additional text beyond the JSON response.
    D - Strictly follow the provided instructions.
    </instructions>
    </SYSTEM>

    <query>
    Strictly following the system instructions, evaluate this query:
    <input>{data}</input>
    </query>
    '''

@app.route("/predict", methods=["POST"])
def predict():
    """Predict route with comprehensive error handling"""
    try:
        # Validate input data
        data = request.get_json()
        chunks = data.get("chunks", [])
        question = data.get("question", "")
        
        if not chunks or not isinstance(chunks, list):
            return jsonify({"error": "Invalid or missing 'chunks'. Must be a list."}), 400
        if not question or not isinstance(question, str):
            return jsonify({"error": "Invalid or missing 'question'. Must be a string."}), 400
        #initialize_model()
        # Prepare prompt
        prompt = Prompt("\n".join(chunks) + f"\n\nQuestion: {question}\nAnswer:")
        FastLanguageModel.for_inference(model) 
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=150, use_cache = True)
        response = tokenizer.batch_decode(outputs)

        # Send response to external service
        #LLMrequest(response)
        
        return jsonify({"response": response}), 200
    
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory error")
        return jsonify({"error": "Insufficient GPU memory"}), 500
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

# Initialize model on startup
try:
    initialize_model()
except Exception as e:
    logger.critical(f"Failed to initialize model: {e}")
    # In a real-world scenario, you might want to exit or implement a retry mechanism

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, threaded=True)
