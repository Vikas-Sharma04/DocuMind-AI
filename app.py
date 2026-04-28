import streamlit as st
import tempfile
import logging
import os

from dotenv import load_dotenv

from ingestion import load_pdf, validate_pdf
from chunking import split_docs
from embedding import get_embeddings
from vectorstore import create_vectorstore
from retriever import get_retriever
from qa_chain import get_llm, create_rag_chain, create_chat_chain

load_dotenv()

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

[data-testid="stChatMessage"] {
    padding: 1.5rem 0 !important;
    background-color: transparent !important;
    animation: slideUp 0.4s ease-out;
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

[data-testid="stChatMessageAvatarUser"] { order: 2; margin-left: 15px; margin-right: 0px !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row !important;
    justify-content: flex-end;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown {
    background-color: var(--user-bubble);
    padding: 12px 20px;
    border-radius: 20px 20px 4px 20px;
    border: 1px solid var(--border-color);
    order: 1;
}

[data-testid="stFileUploader"] {
    background-color: var(--input-bg) !important;
    border: 1px dashed var(--border-color) !important;
    border-radius: 8px !important;
    padding: 10px !important;
}

[data-testid="stFileUploaderDropzone"] { background-color: transparent !important; border: none !important; }
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] { background-color: var(--bg-main) !important; }

[data-testid="stChatInput"] {
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    background-color: var(--input-bg) !important;
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

.status-badge {
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    background: var(--button-gradient);
    color: white;
    box-shadow: 0 0 15px rgba(255, 107, 43, 0.2);
}

.doc-card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 10px;
}

.process-step {
    font-size: 13px;
    color: var(--text-secondary);
    margin: 4px 0;
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
        "pdf_loaded": False,
        "pdf_name": None,
        "vectorstore": None,
        "full_text": None,
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
#  PDF PROCESSING
# ─────────────────────────────────────────────
def process_pdf(uploaded_file):
    # Clear previous error state
    st.session_state.process_error = None
    st.session_state.pdf_loaded = False
    
    step_container = st.sidebar.container()
    
    def update_steps(current_step, completed_steps):
        step_container.empty()
        with step_container:
            for step in completed_steps:
                st.markdown(f"✅ <span class='process-step'>{step}</span>", unsafe_allow_html=True)
            if current_step:
                st.markdown(f"⏳ <span class='process-step'>{current_step}...</span>", unsafe_allow_html=True)

    completed = []
    file_path = None
    
    try:
        update_steps("Saving temporary file", completed)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            file_path = tmp.name
        completed.append("File saved")

        update_steps("Validating PDF structure", completed)
        if not validate_pdf(file_path):
            raise ValueError("PDF validation failed: File may be corrupt or encrypted.")
        completed.append("PDF validated")

        update_steps("Parsing document text", completed)
        docs = load_pdf(file_path)
        if not docs:
            raise ValueError("PDF loading error: PDF contains no readable text.")
        
        page_count = len(docs)
        
        if page_count > 100:
             raise ValueError(f"Document exceeds RAG limit ({page_count}/100 pages).")
        
        st.session_state.full_text = "\n\n".join([doc.page_content for doc in docs[:15]])
        
        if page_count > 30:
             st.session_state.consultant_allowed = False
        else:
             st.session_state.consultant_allowed = True

        completed.append("Text parsed")

        update_steps("Splitting text into chunks", completed)
        chunks = split_docs(docs)
        completed.append("Chunks created")

        update_steps("Generating vector embeddings", completed)
        if not st.session_state.embeddings:
            st.session_state.embeddings = get_embeddings()
        completed.append("Embeddings generated")

        update_steps("Creating vector database", completed)
        st.session_state.vectorstore = create_vectorstore(chunks, st.session_state.embeddings)
        st.session_state.llm = st.session_state.llm or get_llm()
        completed.append("Database ready")

        update_steps(None, completed)
        st.session_state.pdf_loaded = True
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.mode = "RAG"
        
        st.session_state.messages.append({
            "role": "system",
            "content": f"✅ '{uploaded_file.name}' is ready. Switched to RAG Mode.",
        })

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Processing Error: {error_msg}")
        st.session_state.process_error = error_msg 
    finally:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
        st.session_state.processing = False

def get_response(query: str):
    llm = st.session_state.llm or get_llm()
    
    if st.session_state.pdf_loaded and st.session_state.mode == "Consultant":
        prompt = f"Consultant AI: Analyze this text: {st.session_state.full_text}\nUser: {query}"
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        return f"👔 **[Consultant Analysis]**\n\n{content}"

    elif st.session_state.pdf_loaded and st.session_state.mode == "RAG":
        retriever = get_retriever(st.session_state.vectorstore)
        rag_chain = create_rag_chain(llm, retriever)
        response = rag_chain.invoke(query)
        content = response.content if hasattr(response, 'content') else str(response)
        return f"🔍 **[Deep Search]**\n\n{content}"
    
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

    st.markdown("### ⚙️ Response Mode")
    if st.session_state.pdf_loaded:
        modes = ["Chat", "RAG", "Consultant"]
        if not st.session_state.get('consultant_allowed', True):
            st.warning("Consultant Mode disabled (>30 pages)")
            modes = ["Chat", "RAG"]
            if st.session_state.mode == "Consultant":
                st.session_state.mode = "RAG"

        st.session_state.mode = st.radio(
            "Select Response Mode:",
            modes,
            index=modes.index(st.session_state.mode) if st.session_state.mode in modes else 0,
            horizontal=True
        )
    else:
        st.info("Upload a PDF to unlock RAG/Consultant modes.")
        st.session_state.mode = "Chat"

    st.divider()

    if st.session_state.pdf_loaded:
        st.markdown(f"""
        <div class="doc-card">
            <div style="color:var(--accent-orange); font-size:10px; font-weight:bold;">LOADED</div>
            <div style="color:white; font-size:13px; margin-top:4px;">{st.session_state.pdf_name}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### ⬆️ Ingest")
    uploaded_file = st.file_uploader("Drop PDF", type="pdf", label_visibility="collapsed", key="pdf_uploader", disabled=st.session_state.processing)
    
    if uploaded_file:
        if uploaded_file.name != st.session_state.pdf_name:
            # FIX: Button is now strictly controlled by the processing state
            if st.button("PROCESS DOCUMENT", use_container_width=True, disabled=st.session_state.processing):
                st.session_state.processing = True
                st.rerun() # Forces the UI to refresh and disable the button immediately
        else:
            st.button("DOCUMENT ACTIVE", use_container_width=True, disabled=True)

    # Secondary check to trigger logic after rerun
    if st.session_state.processing and not st.session_state.pdf_loaded and not st.session_state.process_error:
        process_pdf(uploaded_file)
        st.rerun()

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
            {st.session_state.mode} Mode {("- " + st.session_state.pdf_name) if st.session_state.pdf_loaded else ""}
        </p>
    </div>
    <span class="status-badge">
        Active: {st.session_state.mode}
    </span>
</div>
""", unsafe_allow_html=True)

if st.session_state.process_error:
    st.warning(f"⚠️ **Processing Failed:** {st.session_state.process_error}")

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
                    Your intelligent document companion. Switch modes to search specific facts, 
                    get professional consulting, or just chat.
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
            **PDF Required.** *Deep Retrieval Mode (Max 100 Pgs).* The AI scans your document to find specific facts or data points. Best for finding "needles in a haystack".
            """)

        with c3:
            st.markdown("### 👔 Consultant")
            st.markdown("""
            **PDF Required.** *Contextual Analysis (Max 30 Pgs).* Instead of searching for facts, the AI evaluates structure, logic, and thematic flow.
            """)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "system":
                st.caption(msg["content"])
            else:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

# ─────────────────────────────────────────────
#  CHAT INPUT
# ─────────────────────────────────────────────
user_input = st.chat_input(f"Message in {st.session_state.mode} mode...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with chat_placeholder:
        with st.chat_message("user"):
            st.markdown(user_input)

    try:
        with chat_placeholder:
            with st.chat_message("assistant"):
                spinner_msg = "Thinking..."
                if st.session_state.mode == "RAG":
                    spinner_msg = "Searching document..."
                elif st.session_state.mode == "Consultant":
                    spinner_msg = "Analysing context..."
                elif st.session_state.mode == "Chat":
                    spinner_msg = "Generating response..."
                
                with st.spinner(spinner_msg):
                    response = get_response(user_input)
                    st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    except Exception as e:
        st.error(f"Chat Error: {str(e)}")