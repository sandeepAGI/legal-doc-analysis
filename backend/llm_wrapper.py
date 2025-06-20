from langchain_community.llms import Ollama
from .prompts import QNA_PROMPT_TEMPLATE

def get_llm():
    return Ollama(model="llama3", temperature=0.0)

def synthesize_answer(query, retrieved_chunks, llm=None):
    if not llm:
        llm = get_llm()
    context = "\n\n".join(doc.page_content for doc, _ in retrieved_chunks)
    prompt = QNA_PROMPT_TEMPLATE.format(context=context, question=query)
    return llm.invoke(prompt)