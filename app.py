from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize Flask app
app = Flask(_name_)

# Load your model and tokenizer
model_name = "a-hamdi/NGILlama3-merged"  # Hugging Face model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
def LLMrequest(resp):

    url = "http://127.0.0.1:10000/response"

    # Prompt to send in the POST request
    payload = {
        "response": resp
    }

    try:
        # Make a POST request with hamdi's llm server
        response = requests.post(url, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:

            print('successful' 
        
        else:
            print("Failed to get a valid response. Status code:", response.status_code)

    except Exception as e:
        print("Error during request:", e)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        chunks = data.get("chunks", [])
        question = data.get("question", "")
        
       
        if not chunks or not isinstance(chunks, list):
            return jsonify({"error": "Invalid or missing 'chunks'. Must be a list."}), 400
        if not question or not isinstance(question, str):
            return jsonify({"error": "Invalid or missing 'question'. Must be a string."}), 400

       
        prompt = Prompt( "\n".join(chunks) + f"\n\nQuestion: {question}\nAnswer:")
        
       # Generate response using the model
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        
        LLMrequest(response)
    
    except Exception as e:
        logger.error(f"Error occurred in /rag: {str(e)}")
        return jsonify({"error": str(e)}), 500
def Prompt(data) : 

    return '''  
    <SYSTEM>
    You  are the most knowledgeable medical professional in history, you can effectively and accurately classify medical news articles as 'Trustworthy', 'Doubtful', or 'Fake'.
    You can also provide reasoning and resources to back up your decision.  Please include relevant details such as publication date, author, and any notable bias associated with the sources. Ensure a comprehensive analysis to assist  in determining the credibility of the news report.
    Here are some Examples for you to learn how you can response to my prompt :
    --------------------------------------------------------------------
    [
    {
    "input": "Groundbreaking Research Confirms New Treatment for Common Illness",
    "output": {
    "medical" : "True",
    "news" : "True",
    "reasoning": "A recent scientific study has discovered a revolutionary treatment for [common illness]. The research involved a large sample size and rigorous testing, providing hope for millions of patients worldwide.",
    "label": "Trustworthy",
    "sources": ["link from the internet to the reference that confirms it's trustworthy"]
    }
    },
    {
    "input": "Unverified Sources Claim Extra-terrestrial Contact in Remote Area",
    "output": {
    "medical" : "False",
    "news" : "True",
    "reasoning": "Reports from unidentified sources suggest that extra-terrestrial beings have made contact in a remote area. The lack of credible evidence and reliance on unverified testimonials makes this information highly doubtful.",
    "label": "Doubtful",
    "sources": [link from the internet for the reference that confirms it's doubtful]
    }
    },
    {
    "input": "World Leaders Announce Global Collaboration for Sustainable Energy",
    "output": {
    "medical" : "False",
    "news" : "True",
    "reasoning": "In a historic move, world leaders have come together to announce a comprehensive global collaboration aimed at achieving sustainable and clean energy solutions. The news is corroborated by official statements from multiple government representatives.",
    "label": "Trustworthy",
    "sources": ["https://official-government-statements.com"]
    }
    },
    {
    "input": "Giant Prehistoric Lizard Discovered in Urban Area",
    "output": {
    "medical" : "False",
    "news" : "True",
    "reasoning": "A team of archaeologists claims to have discovered a giant prehistoric lizard in the heart of a major city. The lack of credible sources, scientific backing, and the sensational nature of the news make it likely to be fake.",
    "label": "Fake",
    "sources": ["link from the internet to the reference that confirms it's fake"]
    }
    },
    {
    "input": "What is cancer?",
    "output": {
    "medical" : "True",
    "news" : "False",
    "reasoning": "Cancer is a group of diseases involving abnormal cell growth with the potential to invade or spread to other parts of the body.  ",
    "label": "trustworthy",
    "sources": ["https://official-health-statements.com"]
    }
    },
    ]
    --------------------------------------------------------------------
    Instructions:
    <instructions>
    A - The response should always be only the output part of the JSON.
    B - Responde by JSON contains the fields:
    <json>
        1 - "medical" : "True"  (if the input has a relation with the medical field), or "False" (if the input doen't have a relation to the medical field).
        2-  "news" : "True"  (if the input presents  news, declarations, or information), or "Fake" (if the input presents a question, or is asking to provide information).
        3 - "label" :  "Fake" ( if the input is fake and presents false informations, unverified treatment , or information not aligned with medical research or regulated clinical practices), "Doubtful" (if the input presents trustworthy information contaminated by  some fake information,  or if the information presented is still experimental and is not completely validated by  medical research and clinical practice ), or "Trustworthy"(  if the input presents valid information verified by peer reviewed medical research and common clinical practice).
        4 - "reasoning" : ( contains an explanation to the reason you chose the label ).
        5 - "sources" : contains a list of web sites links, and research papers that have the reference to back up your decision for the label.
    </json>
    C - Don't add any other text besides the JSON response.
    D - Respond to every following prompt by  strictly adhering to the instruction provided above.
    </instructions>
    </SYSTEM>

    <query>
    Strictly following the information and direction provided to you as a system, evaluate this query :
    <input>''' , data , ''' </input>
    </query>" '''
if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5002)
