"""
Retrieval and generation.

The actual RAG loop: retrieve relevant chunks, construct context, generate
a grounded answer.
"""
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from index import load_index
from langchain_cohere import CohereRerank
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

_store_cache:dict[str, object] = {}


SYSTEM_PROMPT = """
You are a careful medical information assistant.
Answer the user's question USING ONLY the provided context.
If the context does not contain the answer, say so explicitly rather than
guessing. Do not use outside knowledge. Be concise and factual.
"""


def get_vectorstore(index_path) -> object:
    """Load FAISS once per index path; reuse on later calls."""
    path = Path(index_path).resolve()
    key = str(path)
    if key not in _store_cache:
        _store_cache[key] = load_index(path)
    return _store_cache[key]


def retrieve(query: str, index_path, k: int = 4):
    """
    Return the top-k most relevant chunks for a query.
    """
    vectorstore = get_vectorstore(index_path)
    return vectorstore.similarity_search_with_score(query, k=k)

def retrieve_with_rerank(query: str, index_path, k: int = 4, fetch_k: int = 15):
    """
    Return the top-k most relevant chunks for a query, using Cohere reranking.
    """
    vectorstore = get_vectorstore(index_path)
    candidates = vectorstore.similarity_search(query, k=fetch_k)

    reranker = CohereRerank(model="rerank-english-v3.0", top_n=k)
    reranked_results = reranker.compress_documents(candidates, query)

    return [
        (doc, doc.metadata.get("relevance_score", 0.0))
        for doc in reranked_results
    ]

def retrieve_mmr(
    query: str,
    index_path,
    k: int = 4,
    fetch_k: int = 20,
    diversity: float = 0.8,
):
    """
    Return the top-k most relevant chunks for a query, using MMR.
    """
    vectorstore = get_vectorstore(index_path)
    docs = vectorstore.max_marginal_relevance_search(
        query, 
        k=k, 
        fetch_k=fetch_k,
        lambda_mult=diversity,
    )
    return [(doc, 0.0) for doc in docs]


def answer(query: str, index_path, k: int = 2, use_rerank: bool = False, use_mmr: bool = False) -> dict:
    """
    Full RAG: retrieve, build context, generate a grounded answer.
    """
    if use_rerank:
        results = retrieve_with_rerank(query, index_path, k=k)
    elif use_mmr:
        results = retrieve_mmr(query, index_path, k=k)
    else:
        results = retrieve(query, index_path, k=k)   
    context_parts = []
    sources = []
    for doc, score in results:
        source_id = doc.metadata.get("source_id", "unknown")
        sources.append(source_id)
        context_parts.append(
            f"Source: {source_id}\nScore: {score:.2f}\n{doc.page_content}"
        )

    context_used = "\n\n".join(context_parts)

    user_message = f"""Context:
    {context_used}
    
    Question: {query}
    """
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    answer_text = response.content

    return {
        "answer": answer_text,
        "sources": list(dict.fromkeys(sources)),
        "context_used": context_used,
    }


if __name__ == "__main__":
    
    q = sys.argv[1] if len(sys.argv) > 1 else "What is type 2 diabetes?"
    index_path = Path(__file__).resolve().parent / "index" / "faiss_qa"

    print(f"Query: {q}\n")

    result = answer(q, index_path)
    print(f"Answer: {result['answer']}\n")


