import os, glob, json, hashlib, re
import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text:latest"
KEEP_ALIVE = "30m"
CHUNK_WORDS = 100      # ~100 words per chunk
OVERLAP_WORDS = 20


def _chunk(text):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    return [" ".join(words[i:i + CHUNK_WORDS])
            for i in range(0, max(len(words), 1), step) if words[i:i + CHUNK_WORDS]]


def _embed(texts, prefix):
    # nomic-embed-text NEEDS these prefixes: "search_document: " / "search_query: "
    out = []
    for i in range(0, len(texts), 32):
        batch = [prefix + t for t in texts[i:i + 32]]
        resp = ollama.embed(model=EMBED_MODEL, input=batch, keep_alive=KEEP_ALIVE)
        out.extend(resp["embeddings"])
    m = np.asarray(out, dtype=np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9   # normalize -> dot product = cosine
    return m


class RAG:
    def __init__(self, docs_dir="knowledge", cache_file="rag_cache.npz"):
        self.chunks, self.matrix = [], None
        self._load(docs_dir, cache_file)

    def _load(self, docs_dir, cache_file):
        files = sorted(glob.glob(os.path.join(docs_dir, "**", "*.*"), recursive=True))
        files = [f for f in files if f.endswith((".txt", ".md"))]
        texts = {f: open(f, encoding="utf-8").read() for f in files}
        sig = hashlib.md5(json.dumps(texts, sort_keys=True).encode()).hexdigest()

        # reuse the saved index if documents haven't changed
        if os.path.exists(cache_file):
            d = np.load(cache_file, allow_pickle=True)
            if str(d["sig"]) == sig:
                self.chunks, self.matrix = list(d["chunks"]), d["matrix"]
                print(f"RAG: loaded {len(self.chunks)} chunks from cache")
                return

        for t in texts.values():
            self.chunks += _chunk(re.sub(r"\s+", " ", t))
        if not self.chunks:
            print("RAG: no documents found")
            return
        print(f"RAG: embedding {len(self.chunks)} chunks (one time)...")
        self.matrix = _embed(self.chunks, "search_document: ")
        np.savez(cache_file, sig=sig, chunks=np.array(self.chunks, dtype=object),
                 matrix=self.matrix)

    def retrieve(self, query, k=3, min_score=0.5):
        if self.matrix is None:
            return []
        q = _embed([query], "search_query: ")[0]
        scores = self.matrix @ q
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] >= min_score]