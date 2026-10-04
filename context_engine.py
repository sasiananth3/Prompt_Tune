"""
Context Engine (RAG Module)
Handles retrieval of background context, guidelines, and reference documents
to ground the LLM and eliminate hallucinations.
"""
from typing import List, Dict
import re

DEFAULT_KNOWLEDGE_BASE = [
    {
        "id": "code_standards",
        "category": "Software Engineering",
        "title": "Software Engineering & Clean Code Standards",
        "content": (
            "1. Produce complete, executable code without leaving placeholders or unfinished logic.\n"
            "2. Always include meaningful comments and edge-case handling (e.g. null/empty checks, boundary values).\n"
            "3. State Time and Space complexity (Big-O notation) clearly.\n"
            "4. Follow idiomatic standards and naming conventions for the requested programming language."
        ),
        "keywords": ["python", "java", "javascript", "code", "script", "function", "bug", "program", "csv", "api", "database", "algorithm", "palindrome", "plaindrome"]
    },
    {
        "id": "support_policy",
        "category": "Customer Support",
        "title": "E-Commerce Return & Shipping Policy",
        "content": (
            "1. Free returns within 30 days of purchase.\n"
            "2. Standard shipping takes 3-5 business days; expedited takes 1-2 business days.\n"
            "3. If delivery is delayed by courier network issues, issue an immediate tracking update and offer a $10 courtesy credit.\n"
            "4. Tone must always be empathetic, accountable, and solution-focused."
        ),
        "keywords": ["order", "delay", "shipping", "refund", "return", "support", "customer", "complaint", "package"]
    },
    {
        "id": "marketing_guidelines",
        "category": "Marketing & Copywriting",
        "title": "Brand Voice & Email Guidelines",
        "content": (
            "1. Keep subject lines under 50 characters with high curiosity and clarity.\n"
            "2. Opening hook must state the core customer value within the first 2 sentences.\n"
            "3. Use active voice and avoid hyperbolic buzzwords ('revolutionary', 'groundbreaking').\n"
            "4. Include a clear, single Call-To-Action (CTA)."
        ),
        "keywords": ["marketing", "email", "campaign", "copy", "blog", "newsletter", "cta", "launch", "product"]
    },
    {
        "id": "academic_research",
        "category": "Research & Analysis",
        "title": "Analytical & Research Formatting",
        "content": (
            "1. Ground all claims in verifiable facts and cite logical assumptions.\n"
            "2. Provide an Executive Summary before diving into methodology.\n"
            "3. Use structured bullet points and comparative tables for quantitative metrics.\n"
            "4. Highlight trade-offs, limitations, and future outlook."
        ),
        "keywords": ["analysis", "report", "research", "summarize", "executive", "findings", "evaluate", "compare"]
    }
]


class ContextEngine:
    def __init__(self, knowledge_base: List[Dict] = None):
        self.knowledge_base = knowledge_base or DEFAULT_KNOWLEDGE_BASE

    def retrieve_context(self, user_intent: str, top_k: int = 2) -> List[Dict]:
        """
        Retrieves the most relevant knowledge chunks based on keyword similarity.
        (Can be upgraded to dense vector embeddings with ChromaDB/FAISS).
        """
        words = set(re.findall(r"\w+", user_intent.lower()))
        scored_docs = []

        for doc in self.knowledge_base:
            keywords = set(doc.get("keywords", []))
            # Calculate overlap with keywords + content tokens
            doc_content_words = set(re.findall(r"\w+", doc["content"].lower()))
            score = len(words.intersection(keywords)) * 3 + len(words.intersection(doc_content_words))
            if score > 0:
                scored_docs.append((score, doc))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        if not scored_docs:
            # Fallback to the first general doc or return empty
            return [self.knowledge_base[0]]

        return [doc for _, doc in scored_docs[:top_k]]

    def add_custom_document(self, title: str, category: str, content: str, keywords: List[str] = None):
        """Allows users to inject their own context documents at runtime."""
        if keywords is None:
            keywords = list(set(re.findall(r"\w+", (title + " " + content).lower())))[:10]
        
        doc = {
            "id": f"custom_{len(self.knowledge_base) + 1}",
            "category": category,
            "title": title,
            "content": content,
            "keywords": keywords
        }
        self.knowledge_base.append(doc)
        return doc
