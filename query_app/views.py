from django.shortcuts import render
from .forms import PDFUploadForm
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer, AutoModel, pipeline
import faiss
import torch
import numpy as np

# Load the Hugging Face API key from environment variable or set it directly
hf_api_key = "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"

def get_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

def similarity_search(query, index, chunks, tokenizer, model):
    query_embedding = get_embeddings([query], tokenizer, model)
    _, indices = index.search(query_embedding, k=5)
    return [chunks[i] for i in indices[0]]

def home(request):
    context = {}
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            pdf = request.FILES['pdf']
            question = form.cleaned_data['question']

            # Extract text from pdf
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                separator='\n',
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # Load the tokenizer and model from Hugging Face
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', use_auth_token=hf_api_key)
            model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', use_auth_token=hf_api_key)

            # Create embeddings for the chunks
            embeddings = get_embeddings(chunks, tokenizer, model)

            # Create FAISS index and add embeddings
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)

            if question:
                docs = similarity_search(question, index, chunks, tokenizer, model)
                qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", use_auth_token=hf_api_key)

                # Prepare the context and the question for the QA model
                context_text = " ".join(docs)
                qa_input = {
                    'question': question,
                    'context': context_text
                }
                response = qa_pipeline(qa_input)
                answer = response['answer']
                context['answer'] = answer

    else:
        form = PDFUploadForm()
    
    context['form'] = form
    return render(request, 'home.html', context)
