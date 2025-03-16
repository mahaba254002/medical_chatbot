import os
import warnings
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from langchain_community.llms import Ollama
import re

# Suppress all warnings
warnings.filterwarnings("ignore")

# Disable TensorFlow oneDNN logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

load_dotenv()

# Load Pinecone API key
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Download Hugging Face embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone index name
index_name = "medicalbot"

# Initialize Pinecone vector store
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Initialize the Ollama model
MODEL = "deepseek-r1:1.5b"
llm = Ollama(model=MODEL)

# Create the document chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create the retrieval chain
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Function to validate context and invoke the chain
def ask_question(question):
    # Retrieve context
    retrieved_context = retriever.invoke(question)
    
    # Check if the retrieved context is empty or irrelevant
    if not retrieved_context or all(doc.page_content.strip() == "" for doc in retrieved_context):
        return "I don't know."
    
    # Invoke the RAG chain with the question
    response = rag_chain.invoke({"input": question})
    return response["answer"]

# Function to remove <think> tags
def remove_think_tags(text):
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        print("User Input:", msg)
        
        # Get the response from the RAG chain
        response = ask_question(msg)
        
        # Clean the response by removing <think> tags
        cleaned_response = remove_think_tags(response)
        
        print("Response:", cleaned_response)
        return cleaned_response
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)