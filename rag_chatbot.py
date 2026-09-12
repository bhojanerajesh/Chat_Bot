import os
import re
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

NAME = "Rajesh Bhojane"
ABOUT_PATH = Path(__file__).resolve().parent / "about_me.txt"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"
CHUNK_SENTENCES = 4
CHUNK_OVERLAP = 1
OPENAI_EMBEDDINGS_OK = True

EXAMPLE_QUESTIONS = [
    "What is your background?",
    "What are your key skills?",
    "What kind of work do you do?",
    "What are your hobbies?",
]

st.set_page_config(
    page_title=f"Ask Me Anything About {NAME}",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
    #MainMenu, header, footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"],
    [data-testid="stHeader"], #stDecoration {
        visibility: hidden;
        height: 0;
        display: none;
    }

    .stApp {
        background:
            radial-gradient(900px 420px at 10% -10%, #dbeafe 0%, transparent 55%),
            radial-gradient(800px 380px at 100% 0%, #e0e7ff 0%, transparent 50%),
            #f4f7fb;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 10rem;
        max-width: 820px;
    }

    .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #38bdf8 100%);
        border-radius: 22px;
        padding: 1.7rem 1.8rem 1.5rem;
        color: #fff;
        margin-bottom: 1rem;
        box-shadow: 0 16px 36px rgba(37, 99, 235, 0.22);
    }

    .hero h1 {
        margin: 0 0 0.4rem 0;
        font-size: 1.85rem;
        letter-spacing: -0.03em;
        font-weight: 750;
    }

    .hero p {
        margin: 0;
        opacity: 0.92;
        font-size: 1.02rem;
    }

    .hero-badge {
        display: inline-block;
        margin-top: 0.85rem;
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.32);
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 0.15rem 0.2rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    }

    .stButton > button {
        border-radius: 999px;
        border: 1px solid #bfdbfe;
        background: #eff6ff;
        color: #1e3a8a;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #2563eb;
        background: #dbeafe;
        color: #1e3a8a;
    }

    .source-chunk {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.75rem 0.85rem;
        margin-bottom: 0.55rem;
        color: #334155;
        font-size: 0.92rem;
    }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def groq_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()


def is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "insufficient_quota",
            "credit_balance_exhausted",
            "exceeded your current quota",
            "you have no credits",
            "billing",
        )
    )


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def load_knowledge_text(uploaded_pdf) -> str:
    if uploaded_pdf is not None:
        text = extract_pdf_text(uploaded_pdf.getvalue())
        if text:
            return text
        raise RuntimeError("Could not read text from that PDF. Try another file.")
    if not ABOUT_PATH.exists():
        raise RuntimeError(
            f"about_me.txt was not found at {ABOUT_PATH}. Add that file and refresh."
        )
    text = ABOUT_PATH.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError("about_me.txt is empty. Add a short personal introduction.")
    return text


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def make_chunks(text: str) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks = []
    step = max(1, CHUNK_SENTENCES - CHUNK_OVERLAP)
    for start in range(0, len(sentences), step):
        piece = sentences[start : start + CHUNK_SENTENCES]
        if piece:
            chunks.append(" ".join(piece))
        if start + CHUNK_SENTENCES >= len(sentences):
            break
    return chunks or [text.strip()]


STOPWORDS = {
    "a", "an", "and", "are", "about", "for", "from", "his", "him", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "rajesh", "bhojane",
    "sudam", "the", "to", "what", "when", "where", "which", "who", "your",
    "you", "do", "does", "did", "with", "that", "this", "have", "has",
}

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+#]+", text.lower())


def content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS and len(token) > 2}


def local_vectorizer(texts: list[str]):
    docs = [tokenize(t) for t in texts]
    df = {}
    for doc in docs:
        for word in set(doc):
            df[word] = df.get(word, 0) + 1
    vocab = list(df)
    n = max(len(docs), 1)
    idf = np.array([np.log((n + 1) / (df[word] + 1)) + 1.0 for word in vocab], dtype=np.float32)
    return vocab, idf


def local_embed(texts: list[str], vocab: list[str], idf: np.ndarray) -> np.ndarray:
    index = {word: i for i, word in enumerate(vocab)}
    matrix = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = tokenize(text)
        if not tokens:
            continue
        counts = {}
        for word in tokens:
            counts[word] = counts.get(word, 0) + 1
        for word, count in counts.items():
            if word in index:
                matrix[row, index[word]] = (count / len(tokens)) * idf[index[word]]
    return matrix


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    docs = matrix / norms
    return docs @ query


def openai_embed(texts: list[str]) -> np.ndarray:
    client = OpenAI(api_key=openai_key())
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def ensure_knowledge_base(uploaded_pdf) -> None:
    source_name = uploaded_pdf.name if uploaded_pdf is not None else "about_me.txt"
    cache_key = (
        source_name,
        getattr(uploaded_pdf, "size", None),
        ABOUT_PATH.stat().st_mtime if ABOUT_PATH.exists() else 0,
    )
    if st.session_state.get("kb_cache_key") == cache_key and st.session_state.get("chunks"):
        return

    text = load_knowledge_text(uploaded_pdf)
    chunks = make_chunks(text)
    vocab, idf = local_vectorizer(chunks + [text])
    local_matrix = local_embed(chunks, vocab, idf)

    global OPENAI_EMBEDDINGS_OK
    mode = "local"
    embeddings = local_matrix
    if openai_key() and OPENAI_EMBEDDINGS_OK:
        try:
            embeddings = openai_embed(chunks)
            mode = "openai"
        except Exception as exc:
            OPENAI_EMBEDDINGS_OK = False
            if is_quota_error(exc):
                st.session_state.openai_unavailable = True
            else:
                st.session_state.kb_warning = str(exc)
            mode = "local"
            embeddings = local_matrix

    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings
    st.session_state.local_vocab = vocab
    st.session_state.local_idf = idf
    st.session_state.embed_mode = mode
    st.session_state.kb_cache_key = cache_key
    st.session_state.kb_source = source_name


def retrieve_chunks(question: str, top_k: int = 3) -> list[str]:
    if st.session_state.get("embed_mode") == "openai":
        try:
            query_vec = openai_embed([question])[0]
        except Exception:
            query_vec = local_embed(
                [question],
                st.session_state.local_vocab,
                st.session_state.local_idf,
            )[0]
    else:
        query_vec = local_embed(
            [question],
            st.session_state.local_vocab,
            st.session_state.local_idf,
        )[0]

    scores = cosine_similarity(query_vec, st.session_state.embeddings)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [st.session_state.chunks[int(i)] for i in top_idx]


def llm_answer(question: str, context_chunks: list[str]) -> str | None:
    context = "\n\n".join(f"Chunk {i + 1}: {chunk}" for i, chunk in enumerate(context_chunks))
    messages = [
        {
            "role": "system",
            "content": (
                f"Answer ONLY using this context about {NAME}. "
                "If the answer is not in the context, say you don’t have that information. "
                "Be friendly, clear, and concise."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]

    if groq_key():
        client = OpenAI(api_key=groq_key(), base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.2,
            messages=messages,
        )
        return (response.choices[0].message.content or "").strip()

    if openai_key() and not st.session_state.get("openai_unavailable"):
        client = OpenAI(api_key=openai_key())
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.2,
            messages=messages,
        )
        return (response.choices[0].message.content or "").strip()

    return None


def local_answer(question: str, context_chunks: list[str]) -> str:
    q_tokens = content_tokens(question)
    if not q_tokens:
        return f"I don’t have that information in {NAME}’s profile."

    ranked = []
    for chunk in context_chunks:
        for sentence in split_sentences(chunk):
            overlap = len(q_tokens & content_tokens(sentence))
            ranked.append((overlap, sentence))
    ranked.sort(key=lambda item: item[0], reverse=True)
    picked = []
    for score, sentence in ranked:
        if score <= 0:
            continue
        if sentence not in picked:
            picked.append(sentence)
        if len(picked) == 3:
            break
    if not picked:
        return f"I don’t have that information in {NAME}’s profile."
    return " ".join(picked)


def answer_question(question: str, context_chunks: list[str]) -> str:
    try:
        reply = llm_answer(question, context_chunks)
        if reply:
            return reply
    except Exception as exc:
        if not is_quota_error(exc):
            st.session_state.last_llm_error = str(exc)
    return local_answer(question, context_chunks)


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    f"Hi! I’m a resume chatbot for {NAME}. Ask me about his background, "
                    "skills, work, projects, or hobbies. I’ll answer only from his profile."
                ),
                "sources": [],
            }
        ]
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "kb_warning" not in st.session_state:
        st.session_state.kb_warning = None
    if "openai_unavailable" not in st.session_state:
        st.session_state.openai_unavailable = False


init_state()

st.markdown(
    f"""
    <div class="hero">
        <h1>🤖 Ask Me Anything About {NAME}</h1>
        <p>A resume answerer that replies only from Rajesh’s profile document.</p>
        <span class="hero-badge">RAG chatbot</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Knowledge source")
    uploaded_pdf = st.file_uploader(
        "Optional: upload a resume PDF",
        type=["pdf"],
        help="Leave empty to use about_me.txt",
    )
    st.caption("Default source: `about_me.txt`. Upload a PDF to answer from a resume instead.")
    if st.button("Clear chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    f"Chat cleared. Ask me anything about {NAME} — I’ll stick to the profile."
                ),
                "sources": [],
            }
        ]
        st.session_state.pending_question = None
        st.rerun()

startup_error = None
try:
    ensure_knowledge_base(uploaded_pdf)
except Exception as exc:
    startup_error = str(exc)

if startup_error:
    st.error(startup_error)
else:
    source = st.session_state.get("kb_source", "about_me.txt")
    mode = st.session_state.get("embed_mode", "local")
    chunk_count = len(st.session_state.get("chunks", []))
    st.caption(f"Answering from **{source}** · {chunk_count} chunks indexed")
    if mode != "openai" and not groq_key():
        st.info(
            "OpenAI embeddings are unavailable (no credits). "
            "The app is using local document search so you can still ask questions. "
            "Add credits to OpenAI, or add GROQ_API_KEY to `.env` for fuller AI answers."
        )

st.markdown("**Try a question**")
cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, question in zip(cols, EXAMPLE_QUESTIONS):
    if col.button(question, use_container_width=True, disabled=bool(startup_error)):
        st.session_state.pending_question = question

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources") or []
        if sources:
            with st.expander("Sources"):
                for i, chunk in enumerate(sources, start=1):
                    st.markdown(
                        f'<div class="source-chunk"><strong>Chunk {i}</strong><br>{chunk}</div>',
                        unsafe_allow_html=True,
                    )

prompt = st.chat_input(
    f"Ask me anything about {NAME}…",
    disabled=bool(startup_error),
)
if prompt:
    st.session_state.pending_question = prompt

question = st.session_state.pending_question
if question and not startup_error:
    st.session_state.pending_question = None
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                ensure_knowledge_base(uploaded_pdf)
                sources = retrieve_chunks(question)
                reply = answer_question(question, sources)
            except Exception as exc:
                reply = f"I ran into a problem: {exc}"
                sources = []
        st.markdown(reply)
        if sources:
            with st.expander("Sources"):
                for i, chunk in enumerate(sources, start=1):
                    st.markdown(
                        f'<div class="source-chunk"><strong>Chunk {i}</strong><br>{chunk}</div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )
