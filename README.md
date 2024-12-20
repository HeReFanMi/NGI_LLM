
# NGILlama3 API

This repository provides a RESTful API built using Flask for interacting with a custom language model, `NGILlama3`, based on the `Llama` architecture. The API allows for text generation using a pre-trained language model from Hugging Face, designed for a range of natural language processing tasks.

## Features

- **Text Generation**: Generate responses based on user input.
- **Custom Model**: The API uses the `NGILlama3` model, a fine-tuned version of Llama, for improved natural language understanding and generation.
- **Hugging Face Integration**: Utilizes the Hugging Face `transformers` library for easy access to pre-trained models and tokenizers.

## Getting Started

### Prerequisites

To use the application, ensure the following dependencies are installed:

- **Docker**: Required for running the application in a containerized environment.
- **Python 3.8+**: If running the application outside of Docker, you need Python and the associated libraries.

### Docker Setup

#### Build the Docker Image

1. Clone this repository:
   ```bash
   git clone https://github.com/HeReFanMi/NGI_LLM.git
   cd NGI_LLM
   ```

2. Build the Docker image:
   ```bash
   docker build -t ngillama3-flask-api .
   ```

#### Run the Docker Container

Once the image is built, you can run the container using:

```bash
docker run -d -p 5002:5002 ngillama3-flask-api
```

This will start the Flask application on port `5002` inside the container and expose it to your host machine.

### API Endpoints

The Flask API exposes a single endpoint:

- `POST /predict`: Takes a JSON payload with a text input and returns a generated response.

#### Request

Make a `POST` request to `/predict` with the following JSON payload:

```json
{
  "text": "Your input text here."
}
```

#### Response

The API will respond with a JSON object containing the generated text:

```json
{
  "response": "The model-generated text here."
}
```

### Example cURL Request

```bash
curl -X POST http://127.0.0.1:5002/predict -H "Content-Type: application/json" -d '{"chunks":["A new study has shown that regular exercise can help reduce the risk of chronic diseases such as diabetes and heart disease.","Research also indicates that physical activity improves mental health and overall quality of life."],"question":"What are the health benefits of regular exercise?"}'
'
```

### Model Information

- **Model Name**: `a-hamdi/NGILlama3-merged`
- **Architecture**: Fine-tuned `Llama` model.
- **Hugging Face Model**: [NGILlama3-merged on Hugging Face](https://huggingface.co/a-hamdi/NGILlama3-merged)

### Development Setup

To run the application locally without Docker, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/HeReFanMi/NGI_LLM.git
   cd NGI_LLM
   ```

2. **Set up the Conda environment:**
   ```bash
   conda create --name unsloth_env python=3.10 pytorch-cuda=11.8 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y
   conda activate unsloth_env
   ```

3. **Install the required Python dependencies:**
   ```bash
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes
   ```

4. Run the Flask app:
   ```bash
   flask run
   ```

This will start the application at `http://127.0.0.1:5002`.

### Requirements

The application relies on the following Python libraries:

- **transformers==4.33.2**: Hugging Face Transformers library for working with pre-trained models.
- **torch==2.0.1**: PyTorch for model inference.
- **flask==2.3.2**: Flask web framework for building the API.

### Troubleshooting

- **Model Loading Issues**: Ensure the model is available on Hugging Face and the internet connection is stable.
- **Out of Memory Errors**: If you are running the app locally and encounter memory issues, consider using a machine with a more powerful GPU or reduce the model size.

