"""
Retrieval and generation.

The actual RAG loop: retrieve relevant chunks, construct context, generate
a grounded answer.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv
from index import load_index
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = """\
You are a careful medical information assistant.
Answer the user's question USING ONLY the provided context.
If the context does not contain the answer, say so explicitly rather than
guessing. Do not use outside knowledge. Be concise and factual.
"""


def retrieve(query: str, index_path, k: int = 4):
    """
    Return the top-k most relevant chunks for a query.
    """
    vectorstore = load_index(Path(index_path))
    return vectorstore.similarity_search_with_score(query, k=k)



def answer(query: str, index_path, k: int = 4) -> dict:
    """
    Full RAG: retrieve, build context, generate a grounded answer.
    """
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
    
    load_dotenv()
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


