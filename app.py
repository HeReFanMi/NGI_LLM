from flask import Flask, request, jsonify
import requests
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock LLM Response Function
def mock_generate_response(prompt):
    """
    Mock function to simulate LLM output.
    """
    return {
        "mock_response": "This is a mocked response for the prompt.",
        "prompt_used": prompt
    }


def LLMrequest(resp):
    """
    Sends the generated response to an external LLM server and handles the result.
    """
    url = "http://127.0.0.1:5002/predict"
    payload = {"text": resp}

    try:
        # Make a POST request with the LLM server
        response = requests.post(url, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            return response.json()  # Return the server's response
        else:
            logger.error(f"Failed response from LLM server: {response.status_code}, {response.text}")
            return {"error": "Failed to get a valid response from LLM server"}
    except Exception as e:
        logger.error(f"Error during LLMrequest: {e}")
        return {"error": f"Exception occurred: {e}"}


def Prompt(data):
    """
    Constructs the prompt for the model based on input data.
    """
    return f"""
<SYSTEM>
You are the most knowledgeable medical professional in history, you can effectively and accurately classify medical news articles as 'Trustworthy', 'Doubtful', or 'Fake'.
You can also provide reasoning and resources to back up your decision. Please include relevant details such as publication date, author, and any notable bias associated with the sources. Ensure a comprehensive analysis to assist in determining the credibility of the news report.
Here are some Examples for you to learn how you can respond to my prompt:
--------------------------------------------------------------------
[
{{
"input": "Groundbreaking Research Confirms New Treatment for Common Illness",
"output": {{
"medical": "True",
"news": "True",
"reasoning": "A recent scientific study has discovered a revolutionary treatment for [common illness]. The research involved a large sample size and rigorous testing, providing hope for millions of patients worldwide.",
"label": "Trustworthy",
"sources": ["link from the internet to the reference that confirms it's trustworthy"]
}}
}},
{{
"input": "Unverified Sources Claim Extra-terrestrial Contact in Remote Area",
"output": {{
"medical": "False",
"news": "True",
"reasoning": "Reports from unidentified sources suggest that extra-terrestrial beings have made contact in a remote area. The lack of credible evidence and reliance on unverified testimonials makes this information highly doubtful.",
"label": "Doubtful",
"sources": [link from the internet for the reference that confirms it's doubtful]
}}
}},
{{
"input": "World Leaders Announce Global Collaboration for Sustainable Energy",
"output": {{
"medical": "False",
"news": "True",
"reasoning": "In a historic move, world leaders have come together to announce a comprehensive global collaboration aimed at achieving sustainable and clean energy solutions. The news is corroborated by official statements from multiple government representatives.",
"label": "Trustworthy",
"sources": ["https://official-government-statements.com"]
}}
}},
{{
"input": "Giant Prehistoric Lizard Discovered in Urban Area",
"output": {{
"medical": "False",
"news": "True",
"reasoning": "A team of archaeologists claims to have discovered a giant prehistoric lizard in the heart of a major city. The lack of credible sources, scientific backing, and the sensational nature of the news make it likely to be fake.",
"label": "Fake",
"sources": ["link from the internet to the reference that confirms it's fake"]
}}
}},
{{
"input": "What is cancer?",
"output": {{
"medical": "True",
"news": "False",
"reasoning": "Cancer is a group of diseases involving abnormal cell growth with the potential to invade or spread to other parts of the body.",
"label": "Trustworthy",
"sources": ["https://official-health-statements.com"]
}}
}}
]
--------------------------------------------------------------------
Instructions:
A - The response should always be only the output part of the JSON.
B - Respond by JSON containing the fields:
    1 - "medical": "True" or "False".
    2 - "news": "True" or "Fake".
    3 - "label": "Trustworthy", "Doubtful", or "Fake".
    4 - "reasoning": Explanation for the chosen label.
    5 - "sources": A list of web links and references supporting your decision.
C - Do not add any other text besides the JSON response.
D - Respond to every following prompt by strictly adhering to the instruction provided above.
</instructions>
</SYSTEM>

<query>
Strictly following the information and direction provided to you as a system, evaluate this query:
<input>{data}</input>
</query>
"""


@app.route("/predict", methods=["POST"])
def predict():
    """
    Flask endpoint to handle predictions using the mock LLM response.
    """
    try:
        # Parse input JSON
        data = request.get_json()
        chunks = data.get("chunks", [])
        question = data.get("question", "")

        # Validate inputs
        if not chunks or not isinstance(chunks, list):
            return jsonify({"error": "Invalid or missing 'chunks'. Must be a list."}), 400
        if not question or not isinstance(question, str):
            return jsonify({"error": "Invalid or missing 'question'. Must be a string."}), 400

        # Construct the prompt and generate mock response
        prompt = Prompt("\n".join(chunks) + f"\n\nQuestion: {question}\nAnswer:")
        mock_response = mock_generate_response(prompt)

        # Log the response to the console
        logger.info(f"Generated Prompt: {prompt}")
        logger.info(f"Mock Response: {mock_response}")
        print(f"Generated Prompt:\n{prompt}")  # Print for CLI
        print(f"Mock Response:\n{mock_response}")  # Print for CLI

        # Return the mocked response
        return jsonify({"response": mock_response})
    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        print(f"Error in /predict: {e}")  # Print for CLI
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
