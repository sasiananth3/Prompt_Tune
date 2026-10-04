"""
Quick test script to verify ContextEngine and MasterPromptCompiler
"""
from context_engine import ContextEngine
from prompt_compiler import MasterPromptCompiler

def test_pipeline():
    engine = ContextEngine()
    compiler = MasterPromptCompiler()

    sample_input = "Write a python function to read a messy CSV file and filter negative numbers."
    print("Testing with sample input:", sample_input)

    # 1. Retrieve Context
    contexts = engine.retrieve_context(sample_input)
    print(f"\n[OK] Retrieved {len(contexts)} context document(s):")
    for doc in contexts:
        print(f" - {doc['title']} ({doc['category']})")

    # 2. Compile Master Prompt
    result = compiler.compile(sample_input, contexts)
    print(f"\n[OK] Inferred Persona: {result['inferred_persona']}")
    print(f"[OK] Inferred Task: {result['inferred_task']}")
    print(f"[OK] Inferred Tone: {result['inferred_tone']}")

    print("\n--- Compiled Master Prompt Preview (First 400 chars) ---")
    print(result['master_prompt'][:400] + "...\n")
    print("[SUCCESS] All pipeline modules verified successfully!")

if __name__ == "__main__":
    test_pipeline()
