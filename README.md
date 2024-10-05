# Talk to PDFs

## Description

"Talk to PDFs" is a web application that allows users to interact with multiple PDF documents through a conversational interface. Utilizing language models, this application extracts text from uploaded PDFs, processes it, and enables users to ask questions related to the content of the documents.

## Features

- Upload multiple PDF documents
- Extract and chunk text from PDFs
- Use conversational AI to answer questions based on the PDF content
- Continuous conversation history for a better user experience

## Technologies Used

- Python
- Streamlit
- LangChain
- OpenAI / Hugging Face Transformers (for embeddings and LLM)
- PyPDF2 (for PDF text extraction)
- FAISS (for vector storage)

## Prerequisites

- Python 3.7 or higher
- An OpenAI or Hugging Face API key (if using their models)
- Required libraries ( do the following -> pip install -r requirements.txt)
