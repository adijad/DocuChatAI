import json
import os
import sys
import boto3
import streamlit as st
import numpy as np
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv


## Load environment variables from .env file
load_dotenv()
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")


## Bedrock Clients

def create_bedrock_client():
    try:
        bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=f"https://bedrock-runtime.{aws_region}.amazonaws.com"
        )
        return bedrock
    except Exception as e:
        st.error(f"❌ Error creating Bedrock client: {str(e)}")
        return None


bedrock = create_bedrock_client()
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


def list_available_models():
    try:
        response = bedrock.list_models()
        model_ids = [model['modelId'] for model in response['models']]
        return model_ids
    except Exception as e:
        st.error(f"❌ Error fetching available models: {str(e)}")
        return []


available_models = list_available_models() if bedrock else []

## Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("input_data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store

def get_vector_store(docs):
    try:
        vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        vectorstore_faiss.save_local("faiss_index")
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")


def get_llm(model_id):
    try:
        llm = Bedrock(model_id=model_id, client=bedrock, model_kwargs={'max_gen_len': 512})
        return llm
    except Exception as e:
        st.error(f"❌ Error creating LLM for model {model_id}: {str(e)}")
        return None


prompt_template = '''
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:'''

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']


def save_uploaded_file(uploaded_file):
    os.makedirs("input_data", exist_ok=True)
    file_path = os.path.join("input_data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def main():
    st.set_page_config(page_title="Chat with PDF using AWS Bedrock 💁", layout="wide")

    st.markdown("# 📚 Chat with Your PDFs using AWS Bedrock")
    st.markdown("---")

    if available_models:
        selected_model = st.sidebar.selectbox("Select a Model to Use:", available_models)
    else:
        st.sidebar.warning("No available models found. Check your AWS credentials or region.")
        return

    user_question = st.text_input("💬 Ask a question about the PDF files", "", placeholder="Type your question here...")

    st.sidebar.title("🗃️ Upload and Update Vector Store")
    st.sidebar.write("Upload your PDF files only. Other file types are not allowed.")

    uploaded_file = st.sidebar.file_uploader("📂 Upload PDF File", type=["pdf"], accept_multiple_files=True)

    if uploaded_file:
        with st.spinner("Saving uploaded files..."):
            for file in uploaded_file:
                save_uploaded_file(file)
        st.sidebar.success("✅ Files saved successfully!")

    if st.sidebar.button("🔄 Update Vectors"):
        with st.spinner("Processing documents and updating vector store..."):
            docs = data_ingestion()
            get_vector_store(docs)
            st.sidebar.success("✅ Vector store updated successfully!")

    if st.button("🤖 Get Response"):
        if user_question.strip():
            with st.spinner("Generating response..."):
                try:
                    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                    llm = get_llm(selected_model)
                    if llm:
                        response = get_response_llm(llm, faiss_index, user_question)
                        st.markdown("### 📝 Response")
                        st.write(response)
                        st.success("✅ Response generated successfully!")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question before proceeding.")


if __name__ == "__main__":
    main()
