import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import tempfile
import torch
import os

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("🦜🔗 Chat with your PDF (Free & Local)")

# --- Sidebar info ---
st.sidebar.markdown("""
### How it works
1. Upload a PDF  
2. Ask questions about it  
3. Everything runs locally — no API key needed

### Tech Stack
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **LLM**: TinyLlama-1.1B (runs on CPU)
- **Vector Store**: FAISS
""")

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# upload pdf
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Only process if new file uploaded
    if st.session_state.qa_chain is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        with st.spinner("📄 Processing PDF... this might take a minute"):
            try:
                # 1. Load and split PDF
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                st.info(f"✅ Loaded {len(docs)} pages from PDF")

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
                chunks = splitter.split_documents(docs)
                st.info(f"✅ Split into {len(chunks)} chunks")

                # 2. Create embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                
                # 3. Create vector store
                vectorstore = FAISS.from_documents(chunks, embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                st.info("✅ Vector store created")

                # 4. Load LLM
                model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=300,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.95,
                    repetition_penalty=1.15
                )
                llm = HuggingFacePipeline(pipeline=pipe)
                st.info("✅ LLM loaded")

                # 5. Create custom QA function
                def answer_question(query):
                    # Get relevant documents
                    relevant_docs = retriever.get_relevant_documents(query)
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    
                    # Create prompt
                    prompt = f"""Based on the following context, answer the question. If you cannot find the answer in the context, say "I cannot find this information in the document."

Context:
{context}

Question: {query}

Answer:"""
                    
                    # Generate answer
                    response = llm(prompt)
                    return response, relevant_docs

                st.session_state.qa_chain = answer_question
                st.success("🎉 PDF ready! Ask your questions below:")
                
                # Clean up temp file
                os.unlink(file_path)

            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.session_state.qa_chain = None

    # Chat interface
    if st.session_state.qa_chain:
        st.divider()
        
        # Display chat history
        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**🙋 You:** {q}")
                st.markdown(f"**🤖 Bot:** {a}")
                st.divider()

        # Input for new question
        query = st.text_input("Ask something about your PDF:", key="query_input")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("Ask", type="primary")
        with col2:
            clear_button = st.button("Clear History")

        if clear_button:
            st.session_state.chat_history = []
            st.rerun()

        if ask_button and query:
            with st.spinner("🤔 Thinking..."):
                try:
                    answer, sources = st.session_state.qa_chain(query)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((query, answer))
                    
                    # Display answer
                    st.markdown("### Answer:")
                    st.write(answer)
                    
                    # Display sources
                    with st.expander("📚 View Source Documents"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Source {i+1}:**")
                            st.text(doc.page_content[:500] + "...")
                            st.markdown("---")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

else:
    st.warning("⬆️ Please upload a PDF to start chatting.")
    st.session_state.qa_chain = None
    st.session_state.chat_history = []