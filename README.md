# Ollama Movies Recommendation Chatbot

"Ollama Movies Recommendation Chatbot" is a chatbot that uses Ollama, RAG, and Qdrant to provide personalized movie recommendations through a Gradio interface.

## Installation

Make sure you has [ollama](https://ollama.com/) installed on your computer, we will be using [qwen:7b](https://ollama.com/library/qwen:7b) as our llm.

1. Clone the repository:

```sh
git clone https://github.com/Santhote/Ollama-Movies-Recommendation-Chatbot
```

2. Navigate to the project directory:

```sh
cd Ollama-Movies-Recommendation-Chatbot
```

3. Install the required Python packages:

```sh
pip install -r requirements.txt
```

4. You can see the [data_processing](https://github.com/Santhote/Ollama-Movies-Recommendation-Chatbot/tree/main/data_processing) folder for the steps to prepare the qdrant "movies" collection.

For more information, refer to the [data_processing] folder.

```sh
docker-compose up -d
```

## Running the Application

To run the application, use the following command:

```sh
python main.py
```

You can access the gradio chatbot at http://localhost:7860/

## Demo

![alt text](image/image4.png)
