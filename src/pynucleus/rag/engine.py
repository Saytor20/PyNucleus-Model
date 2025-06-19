import chromadb, textwrap
from ..settings import settings
from ..llm.model_loader import generate
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
    ctx = "\n\n".join(ctx_chunks)
    prompt = textwrap.dedent(f"""\
        You are a concise chemical-engineering assistant.
        Answer using the context below.

        Context:
        {ctx}

        Question:
        {question}

        Answer:
    """)
    answer = generate(prompt)
    return {"answer": answer.strip(), "sources": ctx_chunks} 