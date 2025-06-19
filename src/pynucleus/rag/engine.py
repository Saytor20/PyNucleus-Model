import chromadb
from ..settings import settings
from ..llm.model_loader import generate
from ..llm.prompting import build_prompt
from ..utils.logger import logger

client = chromadb.PersistentClient(settings.CHROMA_PATH)
collection = client.get_or_create_collection("pynucleus_documents")

def retrieve(query: str, top_k=None):
    top_k = top_k or settings.RETRIEVE_TOP_K
    res = collection.query(query_texts=[query], n_results=top_k)
    docs = res["documents"][0]
    if not docs:
        logger.warning(f"No chunks retrieved for '{query}'")
    return docs

def ask(question: str):
    ctx_chunks = retrieve(question)
    prompt = build_prompt("\n\n".join(ctx_chunks), question)
    answer = generate(prompt)
    return {"answer": answer.strip(), "sources": ctx_chunks} 