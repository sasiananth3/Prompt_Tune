"""
Prompt_Tune: AI Prompt Optimization Pipeline
Starting with simple text input and demonstrating the Interception Architecture.
"""
import os
import time
import streamlit as st
from dotenv import load_dotenv

from context_engine import ContextEngine, DEFAULT_KNOWLEDGE_BASE
from prompt_compiler import MasterPromptCompiler
from llm_client import OpenRouterClient, fetch_live_free_models, DEFAULT_FALLBACK_FREE_MODELS

# Load local environment variables
load_dotenv()

@st.cache_data(ttl=600)
def get_cached_free_models():
    return fetch_live_free_models()

st.set_page_config(
    page_title="Prompt_Tune | Prompt Optimization Pipeline",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Sleek Dark Glassmorphism UI
st.markdown("""
<style>
    /* Metric Cards and Headers */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .pipeline-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 0.5rem;
    }
    .badge-raw {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-opt {
        background-color: rgba(14, 165, 233, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(14, 165, 233, 0.3);
    }
    .step-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "context_engine" not in st.session_state:
    st.session_state.context_engine = ContextEngine()

if "compiler" not in st.session_state:
    st.session_state.compiler = MasterPromptCompiler()

# ----------------- SIDEBAR CONFIGURATION -----------------
with st.sidebar:
    st.image("https://api.iconify.design/heroicons:cpu-chip-20-solid.svg?color=%2338bdf8", width=36)
    st.title("Engine Config")
    st.caption("Powered by OpenRouter Free Tier")

    # API Key Handling
    env_api_key = os.getenv("OPENROUTER_API_KEY", "")
    api_key_input = st.text_input(
        "OpenRouter API Key",
        value=env_api_key,
        type="password",
        help="Get a free key at https://openrouter.ai/keys"
    )

    live_free_models = get_cached_free_models()
    default_env_model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
    default_idx = live_free_models.index(default_env_model) if default_env_model in live_free_models else 0

    selected_model = st.selectbox(
        "Model (Free Tier)",
        options=live_free_models,
        index=default_idx,
        help="Live free models fetched directly from OpenRouter API"
    )

    with st.expander("⚙️ Generation Parameters"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
        max_tokens = st.slider("Max Tokens", 256, 4096, 2500, 128, help="Higher token limit prevents code from getting truncated prematurely.")

    st.divider()

    # RAG Knowledge Base Inspector
    with st.expander("📚 RAG Knowledge Base"):
        st.write(f"**Loaded Documents:** {len(st.session_state.context_engine.knowledge_base)}")
        for doc in st.session_state.context_engine.knowledge_base:
            st.markdown(f"**• {doc['title']}** (`{doc['category']}`)")

        st.caption("New knowledge documents can be injected dynamically by the context engine.")

# Instantiate LLM Client
llm_client = OpenRouterClient(api_key=api_key_input)

# ----------------- MAIN UI -----------------
st.markdown('<div class="main-header">Prompt_Tune Interception Pipeline</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Transform vague raw text into high-precision AI execution through automated Intent Parsing, RAG Context Injection, and Master Prompt Compilation.</div>',
    unsafe_allow_html=True
)

if not llm_client.is_configured():
    st.warning("⚠️ **OpenRouter API Key missing:** Please enter your OpenRouter API key in the left sidebar or in `.env` to execute live models.")

# Phase 1: Simple Text Input
st.subheader("1. Enter Your Idea or Request")

# Example quick-fill buttons
col_ex1, col_ex2, col_ex3 = st.columns(3)
with col_ex1:
    if st.button("📦 Delivery Complaint", use_container_width=True):
        st.session_state["user_input_val"] = "A customer is complaining their laptop order #1024 is late. Write a response."
with col_ex2:
    if st.button("🐍 Python CSV Script", use_container_width=True):
        st.session_state["user_input_val"] = "Write a python function to read a messy CSV file and filter negative numbers."
with col_ex3:
    if st.button("📈 Growth Summary", use_container_width=True):
        st.session_state["user_input_val"] = "Summarize our quarterly cloud migration results for the leadership team."

# Simple text area for raw prompt
raw_user_input = st.text_area(
    label="Raw User Thought / Unstructured Prompt",
    value=st.session_state.get("user_input_val", ""),
    placeholder="Type any raw, unstructured request here (e.g., 'Draft a marketing email for my mobile app')...",
    height=110
)

col_run1, col_run2 = st.columns([1, 1])
with col_run1:
    run_comparison = st.button("⚡ Run Side-by-Side Comparison (Direct vs. Pipeline)", type="primary", use_container_width=True)
with col_run2:
    run_pipeline_only = st.button("🚀 Run Optimized Pipeline Only", use_container_width=True)

# ----------------- PIPELINE EXECUTION -----------------
if (run_comparison or run_pipeline_only) and raw_user_input.strip():
    if not llm_client.is_configured():
        st.error("Please provide an OpenRouter API key in the sidebar before running.")
    else:
        # Step 1: Context Engine Fetch (RAG)
        with st.spinner("Step 1/3: Retrieving domain context from RAG engine..."):
            retrieved_docs = st.session_state.context_engine.retrieve_context(raw_user_input)

        # Step 2: Master Prompt Compilation
        with st.spinner("Step 2/3: Compiling Master Prompt (injecting persona & constraints)..."):
            compilation_result = st.session_state.compiler.compile(
                raw_user_input=raw_user_input,
                retrieved_contexts=retrieved_docs
            )
            master_prompt = compilation_result["master_prompt"]

        # Step 3: LLM Execution
        col_left, col_right = st.columns(2)

        # Direct LLM Call (if Comparison requested)
        raw_output_text = ""
        raw_duration = 0.0
        if run_comparison:
            with col_left:
                st.markdown('<span class="pipeline-badge badge-raw">Standard Direct Call</span> **(Zero-Shot)**', unsafe_allow_html=True)
                with st.spinner(f"Calling {selected_model} with raw prompt..."):
                    start_t = time.time()
                    raw_res = llm_client.call_model(
                        prompt=raw_user_input,
                        model=selected_model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    raw_duration = round(time.time() - start_t, 2)
                    if raw_res["success"]:
                        st.markdown(raw_res["output"])
                        st.caption(f"⏱️ {raw_duration}s | Model: {selected_model}")
                    else:
                        st.error(raw_res["error"])

        # Optimized Pipeline Execution
        with col_right if run_comparison else st.container():
            st.markdown('<span class="pipeline-badge badge-opt">Optimized Pipeline</span> **(Interception Model)**', unsafe_allow_html=True)
            with st.spinner(f"Calling {selected_model} with compiled Master Prompt..."):
                start_t = time.time()
                opt_res = llm_client.call_model(
                    prompt=master_prompt,
                    model=selected_model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                opt_duration = round(time.time() - start_t, 2)
                if opt_res["success"]:
                    st.markdown(opt_res["output"])
                    st.caption(f"⏱️ {opt_duration}s | Injected Docs: {len(retrieved_docs)} | Model: {selected_model}")
                else:
                    st.error(opt_res["error"])

        # ----------------- DETAILED PIPELINE INSPECTION -----------------
        st.divider()
        st.subheader("🔍 Behind the Scenes: Interception Pipeline Breakdown")

        with st.expander("View Pipeline Stages (Intent Parsing, RAG Context, & Compiled Master Prompt)", expanded=True):
            tab1, tab2, tab3 = st.tabs(["1. Inferred Intent & Persona", "2. Injected RAG Context", "3. Compiled Master Prompt Payload"])

            with tab1:
                st.markdown(f"**Inferred Task:** `{compilation_result['inferred_task']}`")
                st.markdown(f"**Assigned Persona:** `{compilation_result['inferred_persona']}`")
                st.markdown(f"**Enforced Tone:** `{compilation_result['inferred_tone']}`")
                st.markdown(f"**Structural Format Rule:** {compilation_result['inferred_format']}")

            with tab2:
                st.write(f"The Context Engine matched **{len(retrieved_docs)}** relevant document(s) for this request:")
                for i, doc in enumerate(retrieved_docs, 1):
                    st.info(f"**Document {i}: {doc['title']}** (`{doc['category']}`)\n\n{doc['content']}")

            with tab3:
                st.write("This is the exact full prompt compiled by the software and delivered to the LLM:")
                st.code(master_prompt, language="markdown")
