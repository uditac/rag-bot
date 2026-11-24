"""A small RAG pipeline: chunk -> embed -> retrieve -> answer with citations."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

from chunking import chunk_text, load_documents

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("RAG_MODEL", "claude-sonnet-5-5")
LOG_PATH = Path(os.getenv("RAG_LOG", "logs/queries.jsonl"))
REFUSAL = "I don't know based on the documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the context provided.
- Each context passage starts with its source in square brackets.
- Cite the sources you used, like [faq.txt].
- If the context does not contain the answer, reply exactly: {REFUSAL}
- Never guess or use outside knowledge."""


class RagBot:
    def __init__(self, db_path="db", chunk_size=800, overlap=100, k=4):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.k = k
        self.db_path = db_path
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.chroma = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma.get_or_create_collection(
            "docs", metadata={"hnsw:space": "cosine"}
        )
        self._llm = None

    # ---- indexing -------------------------------------------------------
    def index(self, data_dir="data", reset=True) -> int:
        if reset:
            self.chroma.delete_collection("docs")
            self.collection = self.chroma.get_or_create_collection(
                "docs", metadata={"hnsw:space": "cosine"}
            )
        ids, texts, metas = [], [], []
        for source, text in load_documents(data_dir):
            for i, piece in enumerate(chunk_text(text, self.chunk_size, self.overlap)):
                ids.append(f"{source}-{i}")
                texts.append(piece)
                metas.append({"source": source, "chunk": i})
        embeddings = self.embedder.encode(texts, normalize_embeddings=True).tolist()
        self.collection.add(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)
        return len(texts)

    # ---- retrieval ------------------------------------------------------
    def retrieve(self, question: str) -> list[dict]:
        q = self.embedder.encode([question], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=q, n_results=self.k)
        return [
            {"text": doc, "source": meta["source"], "distance": dist}
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    # ---- generation -----------------------------------------------------
    def _client(self):
        if self._llm is None:
            self._llm = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._llm

    def answer(self, question: str) -> dict:
        start = time.time()
        passages = self.retrieve(question)
        context = "\n\n".join(f"[{p['source']}]\n{p['text']}" for p in passages)
        result = {
            "question": question,
            "answer": None,
            "sources": sorted({p["source"] for p in passages}),
            "passages": passages,
            "input_tokens": None,
            "output_tokens": None,
            "error": None,
        }
        try:
            msg = self._client().messages.create(
                model=LLM_MODEL,
                max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                }],
            )
            result["answer"] = "".join(b.text for b in msg.content if b.type == "text")
            result["input_tokens"] = msg.usage.input_tokens
            result["output_tokens"] = msg.usage.output_tokens
        except Exception as exc:  # log failures instead of crashing the bot
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["answer"] = "Sorry, something went wrong while answering."
        result["latency_s"] = round(time.time() - start, 3)
        self._log(result)
        return result

    # ---- monitoring -----------------------------------------------------
    def _log(self, result: dict) -> None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "question": result["question"],
            "sources": result["sources"],
            "top_distance": result["passages"][0]["distance"] if result["passages"] else None,
            "latency_s": result["latency_s"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "error": result["error"],
        }
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
