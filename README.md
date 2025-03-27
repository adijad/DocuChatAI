# **DocuChatAI - AI-Powered Document Question-Answering System**

**DocuChatAI** is an AI-powered document question-answering system that leverages **AWS Bedrock and LangChain** to interact with PDF files, extract knowledge, and respond to user queries intelligently. With seamless PDF ingestion, advanced vector embeddings using **Amazon Titan**, and dynamic model selection, DocuChatAI empowers users to extract insights from documents with precision and speed.

---

## ** Features**
- ** PDF Upload and Ingestion:** Upload multiple PDF files directly through the user interface.  
- ** Intelligent Embeddings:** Uses **Amazon Titan via AWS Bedrock** to generate vector embeddings.  
- ** Dynamic Model Selection:** Choose from available Bedrock models to generate the most accurate responses.  
- ** Smart Query Answering:** Provides concise and relevant answers to user questions by leveraging vector stores and embeddings.  
- ** Vector Store Management:** Efficiently updates vectors with new data to enhance response accuracy.  

---

## ** Tech Stack**
- **Frontend:** Streamlit for a user-friendly web interface.  
- **Backend:** Python with Boto3 and AWS Bedrock.  
- **Vector Store:** FAISS for fast and efficient vector similarity search.  
- **Embedding Models:** Amazon Titan via AWS Bedrock.  
- **Language Models:** Various Bedrock models like Llama2 and others.  
- **Document Parsing:** LangChain for text splitting and processing.  

---

## ** Example Use Cases**
- **Research Assistance:** Quickly find answers from research papers.  
- **Customer Support Documentation:** Get precise answers from support manuals.  
- **Legal Document Analysis:** Extract relevant information from contracts and case files.  

---

## ** Future Improvements**
- **Multi-Document Merging:** Combine insights from multiple documents.  
- **Answer Summarization:** Summarize long responses for quicker insights.  
- **Fine-Tuned Models:** Use specialized models for specific domains like medical or legal texts.  
