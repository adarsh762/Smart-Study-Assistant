"""
LLM module: Groq API wrapper for chat completions.
"""

from typing import List

from groq import Groq

from config import GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE

# ── Groq client (initialized once) ──
_client = None


def get_client() -> Groq:
    """Get or initialize the Groq client."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


SYSTEM_PROMPT = (
    "You are EduBot, a concise and accurate educational assistant. "
    "Use the provided context from course notes to answer the user's question. "
    "If the answer is not contained in the context, say so clearly. "
    "Format your responses with markdown when helpful (bold, lists, code blocks). "
    "Always cite which source document you drew information from."
)


def ask(
    question: str,
    context_chunks: List[dict],
    model: str = LLM_MODEL,
    max_tokens: int = LLM_MAX_TOKENS,
    temperature: float = LLM_TEMPERATURE,
) -> str:
    """
    Send a RAG query to Groq and return the response text.

    Args:
        question: User's question
        context_chunks: Retrieved document chunks with text, source, score
        model: Groq model identifier
        max_tokens: Max response tokens
        temperature: Sampling temperature

    Returns:
        LLM response text
    """
    client = get_client()

    # Build context block
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']} | "
            f"chunk {chunk['chunk_id']} | "
            f"relevance {chunk['score']:.3f}]\n"
            f"{chunk['text']}"
        )
    context_text = "\n\n---\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": f"Context documents (use only these to answer):\n\n{context_text}",
        },
        {"role": "user", "content": question},
    ]

    try:
        completion = client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"**Error from Groq API:** {e}"
