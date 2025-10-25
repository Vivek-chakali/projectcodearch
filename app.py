import streamlit as st
import os
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import GitLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Helper Function for Ingestion & Processing ---

@st.cache_resource(show_spinner="Analyzing repository... This may take a few minutes.")
def process_repository(repo_url, openai_api_key):
    """
    Clones a GitHub repo, splits the code, creates embeddings, and stores in ChromaDB.
    """
    try:
        # Use a temporary directory to clone the repo
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # 1. Clone the repository
            st.write(f"Cloning {repo_url}...")
            loader = GitLoader(
                clone_url=repo_url,
                repo_path=temp_dir,
                branch="main",  # You can make this configurable
            )
            documents = loader.load()

            if not documents:
                st.error("Could not load repository. Check the URL and ensure it's public.")
                return None

            st.write(f"Loaded {len(documents)} documents from the repository.")

            # 2. Split the documents into chunks
            # This is a generic text splitter, good for various code files
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200
            )
            docs = text_splitter.split_documents(documents)
            
            if not docs:
                st.error("Failed to split documents.")
                return None

            st.write(f"Split into {len(docs)} text chunks.")

            # 3. Create Embeddings
            # This is where we use the ML model to turn text into vectors
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

            # 4. Store in ChromaDB (our vector store)
            # This creates an in-memory vector database
            st.write("Creating vector store (ChromaDB)...")
            db = Chroma.from_documents(docs, embeddings)
            
            st.write("Repository analysis complete!")
            
            # Return the retriever interface
            return db.as_retriever(
                search_type="mmr",  # Use Maximal Marginal Relevance for diverse results
                search_kwargs={"k": 8} # Return top 8 results
            )
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# --- Prompt Engineering ---
# This is our custom prompt to guide the RAG agent.
QA_PROMPT_TEMPLATE = """
You are "The Legacy Code Archaeologist" 🏛️. 
You are an expert AI assistant that helps developers understand complex, legacy codebases.
Your goal is to provide clear, concise, and accurate answers based *only* on the code snippets provided as context.

Use the following pieces of context (code snippets) to answer the user's question.

Context:
{context}

Question:
{question}

Helpful Answer:
"""

# --- Main Streamlit App UI ---

st.set_page_config(page_title="The Legacy Code Archaeologist", layout="wide")
st.title("🏛️ The Legacy Code Archaeologist")
st.write("Your AI agent for understanding any codebase.")

# --- 1. Sidebar for Inputs ---
with st.sidebar:
    st.header("1. Analyze Repository")
    repo_url = st.text_input("GitHub URL", placeholder="https.github.com/user/repo")
    openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    if st.button("Analyze"):
        if not repo_url:
            st.warning("Please enter a GitHub URL.")
        elif not openai_api_key:
            st.warning("Please enter your OpenAI API Key.")
        else:
            # This runs the ingestion and caching
            retriever = process_repository(repo_url, openai_api_key)
            st.session_state.retriever = retriever  # Store retriever in session state
            st.session_state.openai_api_key = openai_api_key # Store key
            st.success("Repository analyzed! You can now ask questions.")

    st.markdown("---")
    st.header("About")
    st.info("This is an MVP built for a hackathon, demonstrating a RAG pipeline (LangChain, ChromaDB, OpenAI) for code understanding.")


# --- 2. Chat Interface ---
st.header("2. Ask Questions About the Code")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("e.g., 'What does the main function in setup.py do?'"):
    
    # Check if repo is analyzed
    if "retriever" not in st.session_state or st.session_state.retriever is None:
        st.error("Please analyze a repository from the sidebar first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- This is the RAG Agentic Call ---
        with st.chat_message("assistant"):
            with st.spinner("Digging through the code..."):
                try:
                    # 1. Setup the LLM (with parameters)
                    llm = ChatOpenAI(
                        model="gpt-3.5-turbo", # Fast and cheap for a hackathon
                        openai_api_key=st.session_state.openai_api_key,
                        temperature=0.1 # Low temperature for factual code answers
                    )
                    
                    # 2. Create the Prompt Template
                    QA_PROMPT = PromptTemplate(
                        template=QA_PROMPT_TEMPLATE,
                        input_variables=["context", "question"]
                    )

                    # 3. Create the RAG Chain (This is our "Agent")
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff", # "stuff" means we 'stuff' all context into one prompt
                        retriever=st.session_state.retriever,
                        chain_type_kwargs={"prompt": QA_PROMPT},
                        return_source_documents=True # Request source docs
                    )
                    
                    # 4. Run the chain (Token Generation happens here)
                    result = qa_chain({"query": prompt})
                    response = result["result"]
                    
                    # Add source documents to the response for citations
                    sources = [doc.metadata.get('source', 'Unknown') for doc in result['source_documents']]
                    unique_sources = sorted(list(set(sources)))
                    response_with_sources = f"{response}\n\n---\n*Sources:*\n" + "\n".join(f"- `{s}`" for s in unique_sources)

                    st.markdown(response_with_sources)
                    st.session_state.messages.append({"role": "assistant", "content": response_with_sources})

                except Exception as e:
                    st.error(f"An error occurred while generating the response: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I ran into an error: {e}"})
