"""
Prompt Compiler (Master Prompt Generator)
Synthesizes user intent, retrieved domain context, and structural guardrails
into an optimized instruction payload for the target LLM.
"""
from typing import List, Dict, Optional


def infer_task_and_persona(user_input: str) -> Dict[str, str]:
    """
    Lightweight heuristic intent classifier to infer role and goal from simple raw text.
    In Phase 2, this can be directly controlled via the Intent Capture UI.
    """
    text = user_input.lower()

    code_keywords = [
        "code", "script", "python", "java", "javascript", "typescript", "c++", "cpp", "c#",
        "golang", "rust", "html", "css", "react", "sql", "php", "ruby", "swift", "kotlin",
        "function", "class", "method", "bug", "api", "csv", "regex", "algorithm",
        "palindrome", "plaindrome", "leetcode", "data structure", "array", "binary", "sorting"
    ]
    if any(k in text for k in code_keywords):
        return {
            "task": "Technical Implementation",
            "persona": "Principal Software Architect",
            "format": "Clean, complete, fully working code block with inline comments, followed by brief complexity analysis and test cases.",
            "tone": "Technical, precise, and practical"
        }
    elif any(k in text for k in ["email", "marketing", "pitch", "copy", "newsletter", "blog"]):
        return {
            "task": "Strategic Communication & Copywriting",
            "persona": "Senior Copywriter & Conversion Strategist",
            "format": "Compelling subject line suggestions, followed by structured email body with clear CTA.",
            "tone": "Engaging, professional, and value-driven"
        }
    elif any(k in text for k in ["customer", "delay", "shipping", "refund", "support", "complaint"]):
        return {
            "task": "Customer Relations & Support",
            "persona": "Lead Customer Experience Specialist",
            "format": "Empathetic acknowledgment, clear explanation of status/policy, and concrete resolution steps.",
            "tone": "Empathetic, accountable, and reassuring"
        }
    elif any(k in text for k in ["summarize", "analyze", "report", "review", "evaluate"]):
        return {
            "task": "Analytical Evaluation",
            "persona": "Senior Strategic Intelligence Analyst",
            "format": "Executive summary, key insights in bullet points, and actionable next steps.",
            "tone": "Objective, analytical, and concise"
        }
    else:
        return {
            "task": "General Problem Solving",
            "persona": "Elite Domain Expert & Strategic Advisor",
            "format": "Structured response with step-by-step breakdown and clear key takeaways.",
            "tone": "Authoritative, clear, and structured"
        }


class MasterPromptCompiler:
    def __init__(self):
        pass

    def compile(
        self,
        raw_user_input: str,
        retrieved_contexts: List[Dict],
        custom_persona: Optional[str] = None,
        custom_tone: Optional[str] = None,
        custom_format: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Compiles the raw input into a full Master Prompt.
        """
        inferred = infer_task_and_persona(raw_user_input)

        persona = custom_persona or inferred["persona"]
        tone = custom_tone or inferred["tone"]
        output_format = custom_format or inferred["format"]
        task_category = inferred["task"]

        # Format context documents
        context_blocks = []
        for i, doc in enumerate(retrieved_contexts, 1):
            context_blocks.append(
                f"[REFERENCE CONTEXT {i} - {doc.get('title', 'Document')}]:\n{doc.get('content', '')}"
            )
        formatted_context = "\n\n".join(context_blocks) if context_blocks else "No external documents retrieved."

        # Assemble the Master Prompt
        master_prompt = f"""You are an AI acting as a {persona}.

OBJECTIVE:
Fulfill the following user request with extreme precision, adherence to constraints, and professional depth.

PRIMARY USER REQUEST:
\"\"\"{raw_user_input.strip()}\"\"\"

TASK CATEGORY:
{task_category}

VERIFIED BACKGROUND CONTEXT (RAG KNOWLEDGE):
The following verified reference material must strictly guide your response:
{formatted_context}

EXECUTION CONSTRAINTS & GUARDRAILS:
1. Tone & Style: Maintain a {tone} tone throughout.
2. Grounding: Rely strictly on the verified background context above where applicable. Do not invent contradictory policies, specifications, or false details.
3. Structured Reasoning: Think step-by-step before producing the final response to ensure all nuances of the request are addressed.
4. Output Clarity: {output_format}
5. Avoid generic filler, repetitive apologies, or robotic clichés.

Deliver the final optimized response directly below:
"""
        return {
            "master_prompt": master_prompt,
            "inferred_persona": persona,
            "inferred_task": task_category,
            "inferred_tone": tone,
            "inferred_format": output_format,
            "context_count": len(retrieved_contexts)
        }
