"""
07_prompting.py
Generate answers using OpenRouter.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Load API Key
# -----------------------------

load_dotenv()

MODEL_NAME = "inclusionai/ling-3.0-flash:free"

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


# -----------------------------
# Build Prompt
# -----------------------------

def build_prompt(question, context):

    return f"""
You are a grounded Retrieval-Augmented Generation (RAG) assistant.

Your task is to answer the user's question using ONLY the provided context.

Rules:

1. Use ONLY the information contained in the provided context.

2. Never use outside knowledge, assumptions, or background information.

3. You may summarize, paraphrase, and combine information from multiple retrieved passages into one coherent answer.

4. Do NOT copy long passages from the context verbatim.

5. Every statement in your answer must be supported by the provided context.

6. If the answer cannot be found in the provided context, reply EXACTLY:
"The provided sources do not contain enough information to answer this question."

7. If the context contains conflicting information, explain the conflict instead of making assumptions.

8. Do not invent facts, dates, numbers, definitions, or examples that are not supported by the provided context.

9. Keep the answer concise and directly answer the user's question.
10. Prefer concise answers. Avoid unnecessary details unless requested by the user.
Context:
{context}

Question:
{question}

Answer:
"""


# -----------------------------
# Generate Answer
# -----------------------------

def generate_answer(question, context, sources):

    prompt = build_prompt(question, context)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content.strip()

    answer += "\n\nSources:\n"

    for source in sources:
        answer += f"- {source}\n"

    return answer


# -----------------------------
# Test
# -----------------------------

if __name__ == "__main__":

    question = input("Enter your question: ")

    sample_context = """
Artificial Intelligence (AI) is one of the fastest-growing fields in computer science.

Machine Learning is a branch of Artificial Intelligence that enables computers to learn from data.
"""

    sample_sources = [
        "chunk_0",
        "chunk_4"
    ]

    answer = generate_answer(
        question,
        sample_context,
        sample_sources
    )

    print("\n" + answer)