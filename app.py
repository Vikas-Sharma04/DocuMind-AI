import html
import logging
import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from ingestion import load_pdf, load_web_page, validate_pdf
from chunking import split_docs
from embedding import get_embeddings
from vectorstore import create_vectorstore
from retriever import get_retriever
from qa_chain import get_llm, create_rag_chain, create_chat_chain


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  PAGE CONFIG & STYLING
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

DOCUMIND_AI_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;700&display=swap');

:root {
    --bg-main: #0a0a0a;
    --sidebar-bg: #0d0d0e;
    --card-bg: #161618;
    /* Gradient Color Palette */
    --accent-orange: #ff6b2b;
    --accent-light: #ff9e64;
    --accent-gradient: linear-gradient(135deg, #ff6b2b 0%, #ff9e64 50%, #ffffff 100%);
    --button-gradient: linear-gradient(90deg, #ff6b2b 0%, #ff8c52 100%);
    --text-primary: #ffffff;
    --text-secondary: #a1a1aa;
    --border-color: #27272a;
    --input-bg: #111111;
    --user-bubble: #1e1e20;
}

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] { background-color: var(--bg-main) !important; }
[data-testid="stMain"] { background: transparent; }

#MainMenu, footer { display: none !important; }
header { background: transparent !important; }

/* GRADIENT BRANDING */
.gradient-brand {
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
}

button[kind="header"] { color: var(--text-primary) !important; }

[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border-color);
}

[data-testid="stMainBlockContainer"] {
    max-width: 920px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}

[data-testid="stChatMessage"] {
    max-width: 820px;
    margin: 0 auto !important;
    padding: 1.1rem 0 !important;
    background-color: transparent !important;
    animation: slideUp 0.3s ease-out;
    gap: 1rem !important;
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

[data-testid="stChatMessageAvatarUser"] {
    order: 2;
    margin: 0 0 0 12px !important;
    border: 1px solid var(--border-color);
}

[data-testid="stChatMessageAvatarAssistant"] {
    border: 1px solid rgba(255, 107, 43, 0.35);
    box-shadow: 0 0 14px rgba(255, 107, 43, 0.12);
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row !important;
    justify-content: flex-end;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown {
    max-width: min(78%, 680px);
    background: linear-gradient(145deg, #1b1b1d, #151516);
    padding: 0.8rem 1.1rem;
    border-radius: 20px 20px 5px 20px;
    border: 1px solid #303033;
    order: 1;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.12);
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown p {
    margin: 0;
    line-height: 1.65;
}

[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) .stMarkdown {
    line-height: 1.75;
    color: #e4e4e7;
}

[data-testid="stChatMessage"] .stMarkdown pre {
    border: 1px solid var(--border-color);
    border-radius: 12px;
}

[data-testid="stChatMessage"] .stMarkdown :not(pre) > code {
    color: var(--accent-light);
    background: #1b1715;
    border: 1px solid rgba(255, 107, 43, 0.18);
    border-radius: 5px;
    padding: 0.12rem 0.35rem;
}

[data-testid="stFileUploader"] {
    background-color: var(--input-bg) !important;
    border: 1px dashed var(--border-color) !important;
    border-radius: 8px !important;
    padding: 10px !important;
}

[data-testid="stFileUploaderDropzone"] { background-color: transparent !important; border: none !important; }
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
    background: linear-gradient(180deg, transparent 0%, var(--bg-main) 22%) !important;
}

[data-testid="stChatInput"] {
    max-width: 820px;
    margin: 0 auto;
    border: 1px solid #333336 !important;
    border-radius: 18px !important;
    background: #141415 !important;
    box-shadow: 0 10px 36px rgba(0, 0, 0, 0.34);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stChatInput"]:focus-within {
    border-color: rgba(255, 107, 43, 0.62) !important;
    box-shadow: 0 0 0 3px rgba(255, 107, 43, 0.1), 0 10px 36px rgba(0, 0, 0, 0.34);
}

[data-testid="stChatInput"] textarea {
    line-height: 1.55;
    padding-top: 0.9rem;
    padding-bottom: 0.9rem;
}

[data-testid="stChatInput"] button {
    color: var(--accent-light) !important;
    border-radius: 10px !important;
    transition: background-color 0.2s ease, transform 0.2s ease;
}

[data-testid="stChatInput"] button:hover {
    background: rgba(255, 107, 43, 0.14) !important;
    transform: translateY(-1px);
}

.stButton > button {
    border: none;
    background: var(--button-gradient) !important;
    color: white !important;
    font-weight: 600;
    border-radius: 8px;
    transition: all 0.2s ease;
}

/* FIX: BLOCK ALL CLICKS WHEN DISABLED */
.stButton > button:disabled {
    pointer-events: none !important;
    cursor: not-allowed !important;
    opacity: 0.65 !important;
}

.stButton > button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255, 107, 43, 0.3);
}

div[data-testid="stRadio"] > label {
    color: white !important;
    font-weight: 600 !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] {
    background-color: var(--input-bg);
    padding: 10px;
    border-radius: 10px;
    border: 1px solid var(--border-color);
}

</style>
"""

st.markdown(DOCUMIND_AI_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "source_loaded": False,
        "source_name": None,
        "source_type": None,
        "vectorstore": None,
        "llm": None,
        "embeddings": None,
        "mode": "Chat",
        "processing": False,
        "process_error": None 
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────
#  DOCUMENT PROCESSING
# ─────────────────────────────────────────────
def process_documents(docs, source_name, source_type, update_progress):
    if not docs:
        raise ValueError("The source contains no readable text.")

    update_progress(0.5, "Splitting source into chunks...")
    chunks = split_docs(docs)

    update_progress(0.7, "Generating embeddings...")
    embeddings = st.session_state.embeddings or get_embeddings()

    update_progress(0.9, "Building search index...")
    vectorstore = create_vectorstore(chunks, embeddings)
    llm = st.session_state.llm or get_llm()

    st.session_state.embeddings = embeddings
    st.session_state.vectorstore = vectorstore
    st.session_state.llm = llm
    st.session_state.source_loaded = True
    st.session_state.source_name = source_name
    st.session_state.source_type = source_type
    update_progress(1.0, f"{'PDF' if source_type == 'pdf' else 'Webpage'} ready.")


def process_pdf(uploaded_file):
    st.session_state.process_error = None
    progress_bar = st.sidebar.progress(0, text="Preparing PDF...")

    def update_progress(progress, message):
        progress_bar.progress(progress, text=message)

    file_path = None

    try:
        update_progress(0.1, "Saving PDF...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            file_path = tmp.name

        update_progress(0.25, "Validating PDF...")
        if not validate_pdf(file_path):
            raise ValueError("PDF validation failed: File may be corrupt or encrypted.")

        update_progress(0.4, "Extracting PDF text...")
        docs = load_pdf(file_path)

        page_count = len(docs)
        if page_count > 100:
            raise ValueError(f"Document exceeds RAG limit ({page_count}/100 pages).")

        process_documents(docs, uploaded_file.name, "pdf", update_progress)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Processing Error: {error_msg}")
        st.session_state.process_error = error_msg
    finally:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
        st.session_state.processing = False


def process_web_url(url):
    st.session_state.process_error = None
    progress_bar = st.sidebar.progress(0, text="Preparing webpage...")

    def update_progress(progress, message):
        progress_bar.progress(progress, text=message)

    try:
        update_progress(0.2, "Fetching webpage...")
        docs = load_web_page(url)
        title = docs[0].metadata.get("title") or docs[0].metadata.get("source") or url
        update_progress(0.4, "Webpage loaded.")
        process_documents(docs, title, "webpage", update_progress)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Webpage processing error: {error_msg}")
        st.session_state.process_error = error_msg
    finally:
        st.session_state.processing = False


def get_response(query: str):
    llm = st.session_state.llm or get_llm()

    if st.session_state.mode in {"RAG", "Web Search"}:
        if not st.session_state.source_loaded or st.session_state.vectorstore is None:
            raise ValueError("Load a PDF or webpage before asking questions in this mode.")
        if st.session_state.mode == "RAG" and st.session_state.source_type != "pdf":
            raise ValueError("Load a PDF to use RAG mode.")
        if st.session_state.mode == "Web Search" and st.session_state.source_type != "webpage":
            raise ValueError("Load a webpage to use Web Search mode.")

        retriever = get_retriever(st.session_state.vectorstore)
        rag_chain = create_rag_chain(llm, retriever)
        response = rag_chain.invoke(query)
        content = response.content if hasattr(response, 'content') else str(response)
        label = "Deep Search" if st.session_state.mode == "RAG" else "Web Page Search"
        icon = "🔍" if st.session_state.mode == "RAG" else "🌐"
        return f"{icon} **[{label}]**\n\n{content}"

    else:
        chat_chain = create_chat_chain(llm)
        response = chat_chain.invoke({"question": query})
        return response.content if hasattr(response, 'content') else str(response)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    logo_col, text_col = st.columns([1, 4])
    
    with logo_col:
        st.image("logo.png") 
    
    with text_col:
        st.markdown("""
            <h2 class="gradient-brand" style="margin-top: -10px; font-size: 24px; white-space: nowrap;">
                DocuMind AI
            </h2>
        """, unsafe_allow_html=True)
    
    st.divider()

    st.markdown("### ⚙️ Select Mode")
    st.session_state.mode = st.radio(
        "Choose how to chat:",
        ["Chat", "RAG", "Web Search"],
        index=["Chat", "RAG", "Web Search"].index(st.session_state.mode)
        if st.session_state.mode in ["Chat", "RAG", "Web Search"] else 0,
        horizontal=True,
        key="mode_selector",
    )

    st.divider()

    if st.session_state.mode == "RAG":
        st.markdown("### 📄 Add a PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF to ask questions about",
            type="pdf",
            key="pdf_uploader",
            disabled=st.session_state.processing,
        )
        if uploaded_file:
            if uploaded_file.name != st.session_state.source_name or st.session_state.source_type != "pdf":
                if st.button(
                    "PROCESS PDF",
                    use_container_width=True,
                    disabled=st.session_state.processing,
                ):
                    st.session_state.processing = True
                    st.rerun()
            else:
                st.button("PDF READY", use_container_width=True, disabled=True)

        if st.session_state.processing and uploaded_file and not st.session_state.process_error:
            process_pdf(uploaded_file)
            st.rerun()

    elif st.session_state.mode == "Web Search":
        st.markdown("### 🌐 Add a webpage")
        with st.form("web_page_form"):
            web_url = st.text_input(
                "Paste a public webpage URL",
                placeholder="https://example.com/article",
            )
            load_webpage = st.form_submit_button(
                "LOAD WEBPAGE",
                use_container_width=True,
                disabled=st.session_state.processing,
            )
        if load_webpage:
            st.session_state.processing = True
            process_web_url(web_url)
            st.rerun()

    if st.session_state.source_loaded:
        source_type_label = "PDF" if st.session_state.source_type == "pdf" else "Webpage"
        st.caption(
            f"Loaded {source_type_label}: "
            f"{html.escape(str(st.session_state.source_name))}"
        )

    if st.session_state.process_error:
        st.error(f"Ingestion Error: {st.session_state.process_error}")
        if st.button("CLEAR ERROR", use_container_width=True):
            st.session_state.process_error = None
            st.session_state.processing = False
            st.rerun()

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("Reset All", use_container_width=True):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

# ─────────────────────────────────────────────
#  MAIN INTERFACE
# ─────────────────────────────────────────────
st.markdown(f"""
<div style="padding-bottom:20px; border-bottom:1px solid var(--border-color); display:flex; justify-content:space-between; align-items:center;">
    <div>
        <h1 class="gradient-brand" style="font-size:28px; margin:0;">Workspace</h1>
        <p style="color:var(--text-secondary); font-size:12px; margin:0;">
            {st.session_state.mode} Mode
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

chat_placeholder = st.container()

with chat_placeholder:
    if not st.session_state.messages:
        _, col_img, _ = st.columns([3, 1, 3])
        with col_img:
            st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
            st.image("logo.png", width=120)

        st.markdown("""
            <div style="text-align: center; margin-top: 20px;">
                <h1 class="gradient-brand" style='font-size: 42px;'>Welcome to DocuMind AI</h1>
                <p style='color: #a1a1aa; font-size: 16px; max-width: 600px; margin: auto;'>
                    Your intelligent companion for asking questions about PDFs, web pages, or anything else.
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.write("---")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 🤖 Chat Mode")
            st.markdown("""
            **No PDF Required.** Uses global AI knowledge to answer general questions, brainstorm ideas, or help with coding tasks. This is your standard assistant brain.
            """)

        with c2:
            st.markdown("### 🔍 RAG Search")
            st.markdown("""
            **PDF Required.** *Deep Retrieval Mode (Max 100 pages).* Load a PDF when you want answers based on your document.
            """)

        with c3:
            st.markdown("### 🌐 Web Search")
            st.markdown("""
            **Webpage URL Required.** Load a public webpage to ask questions about its content.
            """)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "system":
                st.caption(msg["content"])
            else:
                avatar = "logo.png" if msg["role"] == "assistant" else "🧑"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

# ─────────────────────────────────────────────
#  CHAT INPUT
# ─────────────────────────────────────────────
user_input = st.chat_input(f"Message in {st.session_state.mode} mode...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with chat_placeholder:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

    try:
        with chat_placeholder:
            with st.chat_message("assistant", avatar="logo.png"):
                spinner_msg = "Thinking..."
                if st.session_state.mode == "RAG":
                    spinner_msg = "Searching document..."
                elif st.session_state.mode == "Web Search":
                    spinner_msg = "Searching webpage..."
                elif st.session_state.mode == "Chat":
                    spinner_msg = "Generating response..."
                
                with st.spinner(spinner_msg):
                    response = get_response(user_input)
                    st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    except Exception as e:
        st.error(f"Chat Error: {str(e)}")