import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        #Looping through pages to add the text 
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(pdf_text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size = 1000,
        chunk_overlap=200,
        length_function = len

    )
    chunks = text_splitter.split_text(pdf_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():

    load_dotenv() #Create variables of your OpenAI and HuggingFace API Keys in .env file and load it fro using it via langchain


    #GUI
    st.set_page_config(page_title='Talk To PDFs', page_icon=':books:')

    st.header("Talk to Mutliple PDFs! :books:")
    st.text_input("Ask me anything from your documents: ")


    with st.sidebar:
        st.subheader("Documents:")
        pdf_documents = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing"):
                #Getting the texxt from the PDF
                pdf_text = get_pdf_text(pdf_documents)


                #Getting the text after chunking 
                text_chunks = get_text_chunks(pdf_text)

                #Creating a VectorDB
                vector_store = get_vectorstore(text_chunks)

                #Creating continuous conversation
                st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()
    