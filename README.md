# Prompt_Tune: Automated Prompt Optimization Pipeline

An implementation of the **Interception Model** architecture for AI Prompt Tuning. Rather than forcing users to master complex prompt engineering techniques, the software intercepts raw thoughts, retrieves relevant domain context (RAG), compiles a structured Master Prompt, and executes models with high precision.

---

## 🎯 Architecture Roadmap

### Phase 1: Simple Text Input & Interception Demonstration (Current)
- Simple, unconstrained text area for user input.
- **Context Engine (RAG)**: Retrieves domain-specific reference materials.
- **Master Prompt Compiler**: Dynamically injects persona, task objectives, constraints, and Chain-of-Thought directives into a structured meta-prompt.
- **OpenRouter Free Tier Integration**: Connects to free LLMs (`meta-llama/llama-3.3-70b-instruct:free`, `google/gemini-2.0-flash-exp:free`, `deepseek/deepseek-r1:free`, `qwen/qwen-2.5-coder-32b-instruct:free`).
- **Side-by-Side Comparison**: Direct Raw Output vs. Pipeline Optimized Output.

### Phase 2: Intent Capture Layer (Next)
- Guided UI elements: Objective selector cards, persona dropdowns, tone pills, and format chips.
- Custom knowledge document uploads directly from the UI.
- Evaluation metrics and accuracy benchmarking against zero-shot baselines.

---

## 🚀 Getting Started

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
Create or edit `.env`:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
```
*(You can also enter your OpenRouter API key directly in the application sidebar).*

### 3. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
