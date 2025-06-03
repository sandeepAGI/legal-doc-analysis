QNA_PROMPT_TEMPLATE = (
    "Answer the question using only the context below. Be concise and quote directly when possible.\n\n"
    "CONTEXT:\n{context}\n\nQUESTION:\n{question}"
)

DOUBLE_CHECK_PROMPT = """
You are a legal analyst reviewing excerpts from one or more legal documents, such as motions, opinions, filings, or expert reports. Each excerpt is tagged with its location (e.g., page number and section title).

Your task is to:
1. Read the excerpts carefully.
2. Identify key claims, facts, or interpretations relevant to the question.
3. Determine if there are conflicting, complementary, or reinforcing points.
4. Choose the answer best supported by the excerpts.
5. Justify your conclusion clearly using the most relevant portions.

---
Excerpts:
{context}

---
Question:
{question}

Answer:
"""