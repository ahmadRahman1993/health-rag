from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from ingest import Chunk

def build_index(chunks, index_path: Path):
    """
    Embed chunks and persist a FAISS vector store.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    documents  = []
    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk.text,
                metadata={
                    **chunk.metadata,
                    "source_id": chunk.source_id,
                },
            )
        )
    vectorstore = FAISS.from_documents(documents, embeddings)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    print(f"Indexed {len(documents)} vectors to {index_path}")
    return vectorstore



def load_index(index_path: Path):
    """
    Load a previously persisted FAISS store.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"Loaded {vectorstore.index.ntotal} vectors from {index_path}")
    return vectorstore




if __name__ == "__main__":
    from ingest import load_corpus, qa_pair_chunking
    
    data_dir = Path(__file__).resolve().parent / "data"
    index_path = Path(__file__).resolve().parent / "index" / "faiss_qa"

    records = load_corpus(data_dir)
    qa_chunks = qa_pair_chunking(records)
    
    build_index(qa_chunks, index_path)

    store = load_index(index_path)
    results = store.similarity_search("What is the symptopms of leukemia?")
    for doc in results:
        print("-", doc.metadata.get("source_id"), doc.page_content[:100], "...")






