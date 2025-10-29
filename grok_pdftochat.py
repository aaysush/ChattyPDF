import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os
import requests

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("🦜🔗 Chat with your PDF")

# Sidebar for API key
st.sidebar.markdown("""
### Setup
Get a free API key from:
- [Groq](https://console.groq.com) (Fast & Free)
- Or [OpenAI](https://platform.openai.com)

### How it works
1. Enter API key
2. Upload PDF
3. Ask questions
""")

api_choice = st.sidebar.radio("Choose API:", ["Groq (Free & Fast)", "OpenAI"])

if api_choice == "Groq (Free & Fast)":
    api_key = st.sidebar.text_input("Groq API Key:", type="password")
    api_endpoint = "https://api.groq.com/openai/v1/chat/completions"
    model_name = "llama-3.1-8b-instant"
else:
    api_key = st.sidebar.text_input("OpenAI API Key:", type="password")
    api_endpoint = "https://api.openai.com/v1/chat/completions"
    model_name = "gpt-3.5-turbo"


if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def call_llm(prompt, api_key, endpoint, model):
    """Call LLM API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and api_key:
    # Process PDF
    if st.session_state.vectorstore is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        with st.spinner("📄 Processing PDF..."):
            progress = st.progress(0)
            
            try:
                # Load PDF
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                progress.progress(33)
                st.info(f"✅ Loaded {len(docs)} pages")

                # Split
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(docs)
                progress.progress(66)
                st.info(f"✅ Created {len(chunks)} chunks")

                
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                
                
                st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
                progress.progress(100)
                st.success("🎉 PDF ready!")
                
                os.unlink(file_path)

            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Chat interface
    if st.session_state.vectorstore:
        st.divider()
        
        # Display history
        for q, a in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)

        
        query = st.chat_input("Ask about your PDF...")

        if query:
            with st.chat_message("user"):
                st.write(query)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                       
                        retriever = st.session_state.vectorstore.as_retriever(
                            search_kwargs={"k": 3}
                        )
                        docs = retriever.get_relevant_documents(query)
                        context = "\n\n".join([doc.page_content for doc in docs])
                        
                        
                        prompt = f"""Based on the following context from a PDF, answer the question. If the answer is not in the context, say "I cannot find this information in the document."

Context:
{context}

Question: {query}

Answer:"""
                        
                        
                        answer = call_llm(prompt, api_key, api_endpoint, model_name)
                        
                        st.write(answer)
                        
                     
                        st.session_state.chat_history.append((query, answer))
                        
                        
                        with st.expander("📚 View Sources"):
                            for i, doc in enumerate(docs):
                                st.markdown(f"**Source {i+1}:**")
                                st.text(doc.page_content[:300] + "...")
                                st.divider()
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

elif not api_key:
    st.warning("⬅️ Please enter your API key in the sidebar")
else:
    st.warning("⬆️ Please upload a PDF to start")


if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()