# End-to-end-Medical-Chatbot-Generative-AI


# How to run?
### STEPS:

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medical python=3.12.8 -y
```

```bash
conda activate medical
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone  as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- Deepseek
- Pinecone