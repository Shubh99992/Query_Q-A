import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModel, pipeline
import faiss
import numpy as np
import torch

# Load the Hugging Face API key from environment variable or set it directly
hf_api_key = "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"

# Function to create embeddings
def get_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Function to perform similarity search
def similarity_search(query, index, chunks, tokenizer, model):
    query_embedding = get_embeddings([query], tokenizer, model)
    _, indices = index.search(query_embedding, k=5)
    return [chunks[i] for i in indices[0]]

def main():
    load_dotenv()
    st.set_page_config(page_title='Ask your PDF')
    st.title('💬 LLM Chat App')
    pdf = st.file_uploader('Upload your PDF', type='pdf')

    if pdf is not None:  # Extract text from pdf
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(  # split into small chunks
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.write(chunks)

        # Load the tokenizer and model from Hugging Face using the API key
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', use_auth_token=hf_api_key)
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', use_auth_token=hf_api_key)

        # Create embeddings for the chunks
        embeddings = get_embeddings(chunks, tokenizer, model)

        # Create FAISS index and add embeddings
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        user_question = st.text_input('Ask me anything')  # show input

        if user_question:
            docs = similarity_search(user_question, index, chunks, tokenizer, model)
            qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", use_auth_token=hf_api_key)

            # Prepare the context and the question for the QA model
            context = " ".join(docs)
            qa_input = {
                'question': user_question,
                'context': context
            }
            response = qa_pipeline(qa_input)

            st.write(response['answer'])

if __name__ == '__main__':
    main()
