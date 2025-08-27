import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from newspaper import Article
import re
import tempfile

st.set_page_config(page_title="EquityLens", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def init_resources():
    # Use Streamlit secrets for production, fallback to env vars for local dev
    model_name = st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature = float(st.secrets.get("OPENAI_TEMPERATURE", os.getenv("OPENAI_TEMPERATURE", "0.3")))
    max_tokens = int(st.secrets.get("OPENAI_MAX_TOKENS", os.getenv("OPENAI_MAX_TOKENS", "1500")))
    embedding_model = st.secrets.get("OPENAI_EMBEDDING_MODEL", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    
    # API key - prioritize Streamlit secrets
    openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set it in Streamlit secrets or environment variables.")
        st.stop()
    
    # Set key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # LLM
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Embeddings
    embeddings = OpenAIEmbeddings(model=embedding_model)

    # Use temporary directory for vector storage in cloud environment
    vector_path = os.path.join(tempfile.gettempdir(), "faiss_index")
    
    vectorstore = None
    index_faiss = os.path.join(vector_path, "index.faiss")
    index_pkl = os.path.join(vector_path, "index.pkl")
    
    if os.path.exists(index_faiss) and os.path.exists(index_pkl):
        vectorstore = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)

    # QA chain
    qa_chain = None
    if vectorstore:
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="map_reduce"
        )

    return llm, embeddings, vectorstore, qa_chain

def clean_parsed_text(text):
    # Add spaces before numbers followed by letters
    text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)
    
    # Add spaces around B,M,compared to
    text = re.sub(r'billion', ' billion ', text)
    text = re.sub(r'million', ' million ', text)
    text = re.sub(r'comparedto', ' compared to ', text)
    
    # Remove asterix and cleanup
    text = text.replace('*', '')
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_source_quotes(llm, sources, processed_docs, query, answer):
    source_quotes = {}
    
    try:
        source_to_content = {}
        for doc in processed_docs:
            source_url = doc.metadata.get("source", "")
            if source_url:
                source_to_content[source_url] = doc.page_content
        
        for source in sources:
            if source in source_to_content:
                content = source_to_content[source]
                
                # Check if uncertain
                uncertainty_indicators = ["i don't know", "i'm not sure", "unclear", "cannot determine", "no information", "not mentioned"]
                answer_lower = answer.lower()
                is_uncertain = any(indicator in answer_lower for indicator in uncertainty_indicators)
                
                if is_uncertain:
                    source_quotes[source] = "N/A"
                else:
                    quote_prompt = f"""
                    CRITICAL INSTRUCTIONS:
                    - You must extract EXACTLY ONE quote from the article below
                    - The quote must be DIRECTLY related to this specific answer: "{answer[:400]}"
                    - The quote must help answer this question: "{query}"
                    - Extract the quote word-for-word from the article text
                    - Maximum 300 characters
                    - Return ONLY the quote text with NO quotation marks, NO explanations, NO additional text
                    - If no quote directly relates to the answer, return exactly: "No relevant quote"
                    
                    Article content:
                    {content[:2000]}
                    """
                    
                    response = llm.invoke(quote_prompt)
                    quote = response.content.strip().replace('"', '').replace("'", "").replace("Quote:", "").strip()
                    
                    # Edge cases
                    if not quote or quote.lower() in ["no relevant quote", "no quote found", "none"]:
                        quote = "N/A"
                    elif len(quote) > 300:
                        quote = quote[:297] + "..."
                    
                    source_quotes[source] = quote
            else:
                source_quotes[source] = "N/A"
    
    except Exception as e:
        # Quote is N/A if error
        for source in sources:
            source_quotes[source] = "N/A"
    
    return source_quotes

# Initialize resources
llm, embeddings, vectorstore, qa_chain = init_resources()

# UI
st.markdown("## 🔍 EquityLens : *AI-powered news analysis for smarter equity research.*")
# url = "https://ghj95.github.io/portfolio//"
# st.markdown(
#     f"<a href='{url}' target='_blank' style='text-decoration: none; color: inherit;'>`By : Gabriel Hardy-Joseph`</a>",
#     unsafe_allow_html=True,
# )

# aws deploy info
col1, col2, col3 = st.columns([0.45, 0.1, 0.1], gap="small")
with col1:
   url = "https://ghj95.github.io/portfolio//"
   st.markdown(
       f"<a href='{url}' target='_blank' style='text-decoration: none; color: inherit;'>`By : Gabriel Hardy-Joseph`</a>",
       unsafe_allow_html=True,
   )
with col2:
   st.markdown(
       "![AWS](https://img.shields.io/badge/AWS-ECS%20%7C%20Fargate-orange?style=flat&logo=amazon-aws&logoColor=white)",
       unsafe_allow_html=True
   )
with col3:
   st.markdown(
       "![Infrastructure](https://img.shields.io/badge/IaC-Terraform-purple?style=flat&logo=terraform&logoColor=white)",
       unsafe_allow_html=True
   )

# App description
def appinfo():
    with st.container(border=True):
        st.write(
            "**EquityLens** is an end-to-end LLM-powered news research tool built with *LangChain* and *OpenAI API*. "
            "It helps equity research analysts efficiently analyze financial news by retrieving, summarizing, "
            "and highlighting key market insights. Designed as a real-world NLP project, it demonstrates how large "
            "language models can streamline research workflows and transform unstructured news into actionable intelligence."
        )
    with st.container(border=True):
        st.write(
            "Try it now by adding up to three URLs and clicking on **Process**. You'll then be able to ask questions about these specific articles below."
        )
    with st.expander("**Technical Architecture & Deployment**"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Infrastructure:**
            - **AWS ECS Fargate**
            - **Application Load Balancer (ALB)**
            - **CloudFront CDN**
            - **Amazon S3** (Assets & Artifacts, versioned)
            - **Auto-scaling**
            """)
            
        with col2:
            st.markdown("""
            **DevOps Pipeline:**
            - **Terraform** (.zip packaging & S3 upload)
            - **CodePipeline** (triggered by new S3 version)
            - **CodeBuild** (Docker image build & push to ECR)
            - **Amazon ECR** (latest + versioned tags)
            - **ECS Deployment** (auto-updated task definition)
            """)


appinfo()
st.markdown("---")

# Sidebar
st.sidebar.header("News Articles \n (paste URLs below)")
url1 = st.sidebar.text_input("Source 1")
url2 = st.sidebar.text_input("Source 2")
url3 = st.sidebar.text_input("Source 3")

# Fetch article + source
def fetch_article_document(url: str) -> Document:
    try:
        article = Article(url)
        article.download()
        article.parse()
        content = clean_parsed_text(article.text) 
        if not content.strip():
            return None
        return Document(page_content=content, metadata={"source": url})
    except Exception as e:
        st.warning(f"Error fetching article from {url}: {str(e)}")
        return None

# Store docs in session state
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = []

# Store URL mapping
if 'url_mapping' not in st.session_state:
    st.session_state.url_mapping = {}

# Store processing status
if 'articles_processed' not in st.session_state:
    st.session_state.articles_processed = False

# Process URLs
if st.sidebar.button("Process"):
    # Use temporary directory for cloud environment
    path = os.path.join(tempfile.gettempdir(), "faiss_index")
    urls = [u for u in [url1, url2, url3] if u]

    if not urls:
        st.error("Please enter at least one article URL.")
    else:
        try:
            with st.spinner("Fetching article texts..."):
                docs = []
                url_mapping = {}
                # Map URLs to position
                input_urls = [url1, url2, url3]
                for i, u in enumerate(input_urls):
                    if u:  
                        doc = fetch_article_document(u)
                        if doc:
                            docs.append(doc)
                            url_mapping[u] = f"Source {i + 1}"
                        else:
                            st.warning(f"No content found for URL: {u}")

            if not docs:
                st.error("No valid article content found. Please check your URLs.")
            else:
                # Store docs for quote extraction
                st.session_state.processed_docs = docs
                st.session_state.url_mapping = url_mapping
                
                with st.spinner("Splitting documents into chunks..."):
                    text_splitter = RecursiveCharacterTextSplitter(
                        separators=["\n\n", "\n", ".", ","],
                        chunk_size=500,
                        chunk_overlap=50
                    )
                    split_docs = text_splitter.split_documents(docs)

                with st.spinner("Embedding into vector database..."):
                    # Ensure directory exists
                    os.makedirs(path, exist_ok=True)
                    vector = FAISS.from_documents(split_docs, embeddings)
                    vector.save_local(path)

                st.session_state.articles_processed = True
                st.rerun()

        except Exception as e:
            st.error("Error processing URLs: The content may be blocked by a paywall. Please try using freely accessible sites instead.")

# Show status
if st.session_state.articles_processed:
    st.success("Articles processed and indexed! Ready for questions.")

# Ask questions
vector_path = os.path.join(tempfile.gettempdir(), "faiss_index")
index_faiss = os.path.join(vector_path, "index.faiss")
index_pkl = os.path.join(vector_path, "index.pkl")

if st.session_state.articles_processed and os.path.exists(index_faiss) and os.path.exists(index_pkl):
    st.markdown("### 💬 Ask Questions About Your Articles")
    
    query = st.text_input("Enter your question about the processed articles:")

    if query:
        try:
            with st.spinner("Retrieving relevant chunks..."):
                vs = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
                retriever = vs.as_retriever()

            with st.spinner("Analyzing your question..."):
                chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=llm, 
                    retriever=retriever
                )
                result = chain({"question": query}, return_only_outputs=True)
            
            if "sources" in result and result["sources"]:
                sources = result["sources"]
                
                if isinstance(sources, str):
                    sources = [s.strip() for s in sources.replace("\n", ",").split(",") if s.strip()]

                # Extract key quotes
                if st.session_state.processed_docs:
                    with st.spinner("Extracting source quotes..."):
                        source_quotes = extract_source_quotes(llm, sources, st.session_state.processed_docs, query, result["answer"])
                        
                        # Display main answer
                        st.markdown("### 🎯 Answer")
                        st.write(result['answer'])
                        
                        # Display sources
                        for src in sources:
                            quote = source_quotes.get(src, "N/A")
                            if quote != "N/A":
                                # Get correct source number
                                source_label = st.session_state.url_mapping.get(src, "Unknown Source")
                                with st.container(border=True):
                                    st.markdown(f"**Quote:** \"{quote}\"")
                                st.markdown(
                                    f'<a href="{src}" target="_blank" style="display:inline-block; background-color:#F0F2F6; color:black; padding:8px 12px; margin-top:8px; border-radius:5px; text-decoration:none;">{source_label}</a>',
                                    unsafe_allow_html=True
                                )

        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")