"""
Alice - car dealership voice agent (low latency) with RAG:
  faster-whisper (ear) -> nomic-embed-text retrieval -> Ollama (brain) -> kokoro-onnx (mouth)

Install:
    python -m pip install kokoro-onnx faster-whisper ollama sounddevice numpy webrtcvad-wheels
    ollama pull nomic-embed-text

Run:
    python voiceagent.py

Put kokoro-v1.0.int8.onnx and voices-v1.0.bin next to this script, or let
them download automatically on first run.

Put your knowledge files (.txt / .md) in the  knowledge/  folder next to this script.
The index is built once and cached in rag_cache.npz (rebuilt automatically when files change).
"""

import os
import re
import io
import csv
import json
import glob
import time
import queue
import hashlib
import threading
import collections
import urllib.request

import numpy as np
import sounddevice as sd
import webrtcvad
import ollama
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

# ---------------- CONFIG ----------------
OLLAMA_MODEL = "alibayram/smollm3:latest"  # from `ollama list`
WHISPER_MODEL = "base.en"     # tiny.en = fastest, base.en = balanced, small.en = most accurate
VOICE = "af_heart"            # af_heart, af_bella, af_sarah, am_adam, bf_emma
SPEED = 1.1                   # slightly faster speech also feels snappier
MAX_RECORD_SECONDS = 20
MIC_RATE = 16000
TTS_RATE = 24000              # Kokoro always outputs 24 kHz

# --- LLM speed ---
MAX_REPLY_TOKENS = 120        # hard cap on reply length (shorter = faster)
KEEP_ALIVE = "30m"            # keep the models loaded in memory
FIRST_CHUNK_WORDS = 4         # start speaking after this many words (or at first comma/period)
CHUNK_WORDS = 12              # later chunks: speak after this many words (or at punctuation)
OLLAMA_OPTIONS = {"num_predict": MAX_REPLY_TOKENS, "num_ctx": 2048, "temperature": 0.7}

# --- RAG ---
EMBED_MODEL = "nomic-embed-text:latest"
KNOWLEDGE_DIR = "knowledge"   # folder with your .txt / .md files (relative to this script)
RAG_CACHE_FILE = "rag_cache.npz"
RAG_VERSION = "v2"            # bump this to force a rebuild when chunking logic changes
RAG_CHUNK_WORDS = 90          # long paragraphs are split into pieces of this many words
RAG_OVERLAP_WORDS = 20        # overlap between chunks
RAG_TOP_K = 3                 # how many chunks to give the LLM (fewer = faster)
RAG_MIN_SCORE = 0.55          # below this, the chunk is ignored (watch the printed scores and tune)

# --- VAD tuning ---
VAD_MODE = 2            # 0-3. Higher = stricter (ignores more noise, may clip quiet speech)
FRAME_MS = 30           # webrtcvad accepts 10, 20 or 30 ms
START_WINDOW = 8        # look at the last 8 frames (240 ms)...
START_NEEDED = 4        # ...and start only if 4 of them are speech
END_SILENCE_MS = 700    # pause that ends your turn (lower = faster, higher = fewer cut-offs)
PRE_ROLL_MS = 300       # keep audio from just before speech was detected
MIN_SPEECH_MS = 400     # ignore anything shorter than this
# ----------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
FILES = ["kokoro-v1.0.int8.onnx", "voices-v1.0.bin"]

SYSTEM_PROMPT = (
    "You are Alice, a friendly voice assistant for a multi-brand car dealership. "
    "You answer customer enquiries about cars, prices, stock and discounts. "
    "Replies are spoken aloud, so keep them to 1-2 short sentences, with no markdown, "
    "lists, emojis or symbols. Only state prices, stock counts and discounts that appear "
    "in the facts you are given; never guess them. If you do not know, say so and offer "
    "to connect the customer with a sales executive. /no_think"
)


# =====================================================================
# RAG
# =====================================================================
def _split_long(text):
    """Keep short chunks whole; split long ones into overlapping word windows."""
    words = text.split()
    if len(words) <= int(RAG_CHUNK_WORDS * 1.4):
        return [text]
    step = max(1, RAG_CHUNK_WORDS - RAG_OVERLAP_WORDS)
    out = []
    for i in range(0, len(words), step):
        piece = words[i:i + RAG_CHUNK_WORDS]
        if piece:
            out.append(" ".join(piece))
        if i + RAG_CHUNK_WORDS >= len(words):
            break
    return out


def _chunks_from_text(text):
    """.md / .txt -> one chunk per paragraph (paragraphs are separated by a blank line)."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        lines = [l for l in para.splitlines() if not l.lstrip().startswith("#")]  # drop headings
        para = re.sub(r"\s+", " ", " ".join(lines)).strip()
        if para:
            out += _split_long(para)
    return out


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _fmt(x):
    return f"{x:g}"


def _chunks_from_inventory(text):
    """inventory.csv -> one spoken-style sentence per car, plus a summary per company
    and an overall total (so 'how many Toyotas do you have?' can be answered)."""
    rows = [{(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
            for r in csv.DictReader(io.StringIO(text))]
    chunks, by_company = [], collections.OrderedDict()

    for r in rows:
        company, model, variant = r.get("company", ""), r.get("model", ""), r.get("variant", "")
        if not company or not model:
            continue
        price, disc = _num(r.get("price_lakh")), _num(r.get("discount_lakh"))
        stock = int(_num(r.get("in_stock")))
        final = max(price - disc, 0)
        name = " ".join(x for x in (company, model, variant) if x)
        fuel = r.get("fuel", "")

        s = f"{name}" + (f" ({fuel})" if fuel else "")
        s += f": ex-showroom price {_fmt(price)} lakh rupees. "
        s += (f"{stock} unit{'s' if stock != 1 else ''} left in stock. "
              if stock > 0 else "Currently out of stock. ")
        s += (f"Discount of {_fmt(disc)} lakh rupees, so the final price is {_fmt(final)} lakh rupees. "
              if disc > 0 else "No discount at the moment. ")
        if r.get("offer"):
            s += f"Offer: {r['offer']}."
        chunks.append(s.strip())
        by_company.setdefault(company, []).append((f"{model} {variant}".strip(), stock, disc))

    total_all = 0
    for company, items in by_company.items():
        total = sum(st for _, st, _ in items)
        total_all += total
        listing = "; ".join(
            f"{m} {st} left" + (f", {_fmt(d)} lakh discount" if d > 0 else "") for m, st, d in items)
        chunks.append(f"{company} summary: {len(items)} models listed and {total} cars in stock in total. "
                      f"{company} stock by model: {listing}.")
    if by_company:
        chunks.append(f"Dealership total: {total_all} cars in stock across {len(by_company)} brands "
                      f"({', '.join(by_company)}).")
    return [c for ch in chunks for c in _split_long(ch)]


def embed(texts, prefix):
    """Embed texts with nomic-embed-text. It needs the prefixes
    'search_document: ' (for stored chunks) and 'search_query: ' (for questions)."""
    out = []
    for i in range(0, len(texts), 32):
        batch = [prefix + t for t in texts[i:i + 32]]
        resp = ollama.embed(model=EMBED_MODEL, input=batch, keep_alive=KEEP_ALIVE)
        out.extend(resp["embeddings"])
    m = np.asarray(out, dtype=np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9   # normalized -> dot product = cosine
    return m


class RAG:
    def __init__(self, docs_dir, cache_file):
        self.chunks, self.matrix = [], None
        self._load(docs_dir, cache_file)

    def _load(self, docs_dir, cache_file):
        os.makedirs(docs_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(docs_dir, "**", "*.*"), recursive=True))
        files = [f for f in files if f.lower().endswith((".txt", ".md", ".csv"))]
        texts = {}
        for f in files:
            with open(f, encoding="utf-8-sig") as fh:
                texts[f] = fh.read()
        sig = hashlib.md5(json.dumps({"v": RAG_VERSION, "files": texts},
                                     sort_keys=True).encode()).hexdigest()

        # reuse the saved index if documents haven't changed
        if os.path.exists(cache_file):
            try:
                d = np.load(cache_file, allow_pickle=True)
                if str(d["sig"]) == sig:
                    self.chunks, self.matrix = list(d["chunks"]), d["matrix"]
                    print(f"RAG: loaded {len(self.chunks)} chunks from cache")
                    return
            except Exception as e:
                print(f"RAG: cache unreadable, rebuilding ({e})")

        for f, t in texts.items():
            if f.lower().endswith(".csv"):
                self.chunks += _chunks_from_inventory(t)
            else:
                self.chunks += _chunks_from_text(t)
        if not self.chunks:
            print(f"RAG: no .txt/.md/.csv files found in '{docs_dir}' - running without RAG")
            return

        print(f"RAG: embedding {len(self.chunks)} chunks (one time)...")
        t0 = time.time()
        self.matrix = embed(self.chunks, "search_document: ")
        np.savez(cache_file, sig=sig, chunks=np.array(self.chunks, dtype=object),
                 matrix=self.matrix)
        print(f"RAG: indexed in {time.time() - t0:.1f}s")

    def retrieve(self, query, k=RAG_TOP_K, min_score=RAG_MIN_SCORE):
        if self.matrix is None:
            return []
        q = embed([query], "search_query: ")[0]
        scores = self.matrix @ q
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] >= min_score]


# =====================================================================
# Models
# =====================================================================
def ensure_models():
    for name in FILES:
        path = os.path.join(BASE, name)
        if not os.path.exists(path):
            print(f"Downloading {name} (first run only)...")
            urllib.request.urlretrieve(URL + name, path)
    return [os.path.join(BASE, n) for n in FILES]


print("Loading models...")
model_path, voices_path = ensure_models()
tts = Kokoro(model_path, voices_path)
stt = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8",
                   cpu_threads=min(8, os.cpu_count() or 4))


def warm_up():
    """Run each model once so the first real turn isn't slow."""
    print("Warming up...")
    tts.create("Hello.", voice=VOICE, speed=SPEED, lang="en-us")
    list(stt.transcribe(np.zeros(MIC_RATE, dtype=np.float32), language="en", beam_size=1)[0])
    try:
        ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": "hi"}],
                    keep_alive=KEEP_ALIVE, options={**OLLAMA_OPTIONS, "num_predict": 1})
    except Exception as e:
        print(f"⚠️  Ollama warm-up failed: {e}")
    try:
        embed(["hello"], "search_query: ")
    except Exception as e:
        print(f"⚠️  Embedding warm-up failed (is '{EMBED_MODEL}' pulled?): {e}")


class Speaker:
    """Text -> (TTS thread) -> audio -> (playback thread), so everything overlaps."""

    def __init__(self):
        self.text_q = queue.Queue()
        self.audio_q = queue.Queue()
        self.out = sd.OutputStream(samplerate=TTS_RATE, channels=1, dtype="float32",
                                   latency="low")
        self.out.start()
        self.t_turn = None
        threading.Thread(target=self._synth, daemon=True).start()
        threading.Thread(target=self._play, daemon=True).start()

    def start_turn(self):
        self.t_turn = time.time()

    def say(self, text):
        text = re.sub(r"[*_#`>~]", "", text).strip()
        if text:
            self.text_q.put(text)

    def wait(self):
        """Block until everything queued has been synthesized AND played."""
        self.text_q.join()
        self.audio_q.join()
        time.sleep(self.out.latency + 0.1)  # let the tail finish so we don't hear ourselves

    def _synth(self):
        while True:
            text = self.text_q.get()
            try:
                samples, _ = tts.create(text, voice=VOICE, speed=SPEED, lang="en-us")
                self.audio_q.put(np.asarray(samples, dtype=np.float32))
            except Exception as e:
                print(f"[TTS error] {e}")
            finally:
                self.text_q.task_done()

    def _play(self):
        while True:
            samples = self.audio_q.get()
            try:
                if self.t_turn is not None:
                    print(f"\n   [first audio after {time.time() - self.t_turn:.2f}s]")
                    self.t_turn = None
                self.out.write(samples.reshape(-1, 1))
            finally:
                self.audio_q.task_done()


def listen():
    """Record one utterance using WebRTC VAD. Returns float32 audio at 16 kHz or None."""
    vad = webrtcvad.Vad(VAD_MODE)
    frame_len = int(MIC_RATE * FRAME_MS / 1000)
    q = queue.Queue()

    window = collections.deque(maxlen=START_WINDOW)
    pre_roll = collections.deque(maxlen=PRE_ROLL_MS // FRAME_MS)
    end_limit = END_SILENCE_MS // FRAME_MS
    frames, started, silence, speech_frames, t_start = [], False, 0, 0, 0.0

    with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="int16",
                        blocksize=frame_len,
                        callback=lambda d, f, t, s: q.put(d.copy().flatten())):
        print("🎤 Listening...")
        while True:
            data = q.get()
            is_speech = vad.is_speech(data.tobytes(), MIC_RATE)

            if not started:
                pre_roll.append(data)
                window.append(is_speech)
                if sum(window) >= START_NEEDED:
                    started = True
                    frames = list(pre_roll)
                    speech_frames = sum(window)
                    t_start = time.time()
            else:
                frames.append(data)
                if is_speech:
                    silence = 0
                    speech_frames += 1
                else:
                    silence += 1
                if silence >= end_limit or time.time() - t_start > MAX_RECORD_SECONDS:
                    break

    if speech_frames * FRAME_MS < MIN_SPEECH_MS:
        return None
    return np.concatenate(frames).astype(np.float32) / 32768.0


def transcribe(audio):
    t0 = time.time()
    segs, _ = stt.transcribe(
        audio, language="en", beam_size=1, temperature=0.0,   # greedy, no slow retries
        without_timestamps=True, vad_filter=False,            # we already did VAD
        condition_on_previous_text=False,
    )
    # keep only segments Whisper is confident are real speech
    text = " ".join(s.text.strip() for s in segs
                    if s.no_speech_prob < 0.6 and s.avg_logprob > -1.2).strip()
    print(f"   [STT {time.time() - t0:.2f}s]")
    junk = {"you", "thank you", "thanks for watching", "thank you for watching"}
    if len(audio) < 2 * MIC_RATE and text.lower().strip(" .!?") in junk:
        return ""
    return text


def next_chunk(buf, first):
    """Return (text_to_speak, remaining) as soon as a few complete words exist."""
    min_w, max_w = (2, FIRST_CHUNK_WORDS) if first else (4, CHUNK_WORDS)
    count = 0
    for m in re.finditer(r"\S+\s+", buf):      # only words already followed by a space
        count += 1
        end_punct = m.group().strip()[-1] in ".!?,;:"
        if (count >= min_w and end_punct) or count >= max_w:
            return buf[:m.end()].strip(), buf[m.end():]
    return None, buf


def reply(history, speaker, context=""):
    """Stream the LLM and hand tiny chunks to the speaker the moment they exist.
    `context` (retrieved facts) is injected for this turn only, not stored in history."""
    t0 = time.time()
    full, buf, first_token, first_chunk, last = "", "", True, True, None
    print("🤖 ", end="", flush=True)

    messages = [dict(m) for m in history]
    if context:
        messages[-1]["content"] = (
            f"Dealership facts:\n{context}\n\n"
            f"Customer question: {messages[-1]['content']}\n"
            "Answer in 1-2 short spoken sentences. Use only these facts for prices, stock and "
            "discounts. If the facts do not cover the question, say you are not sure and offer "
            "a sales executive."
        )

    stream = ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=True,
                         keep_alive=KEEP_ALIVE, options=OLLAMA_OPTIONS)
    for part in stream:
        last = part
        piece = part["message"]["content"]
        if not full.strip():
            piece = piece.lstrip()          # models often start with a blank line
        if not piece:
            continue
        if first_token:
            print(f"[LLM first token {time.time() - t0:.2f}s] ", end="", flush=True)
            first_token = False
        print(piece, end="", flush=True)
        full += piece
        buf += piece

        while True:
            ready, rest = next_chunk(buf, first_chunk)
            if ready is None:
                break
            speaker.say(ready)
            first_chunk, buf = False, rest

    if buf.strip():
        speaker.say(buf)
    print()

    n, d = getattr(last, "eval_count", None), getattr(last, "eval_duration", None)
    if n and d:
        print(f"   [LLM speed {n / (d / 1e9):.1f} tokens/s]")
    speaker.wait()
    return full.strip()


def main():
    warm_up()
    rag = RAG(os.path.join(BASE, KNOWLEDGE_DIR), os.path.join(BASE, RAG_CACHE_FILE))
    speaker = Speaker()
    print("Ready.\n")
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    speaker.say("Hi! I'm Alice. How can I help you today?")
    speaker.wait()

    while True:
        try:
            audio = listen()
            if audio is None:
                continue
            speaker.start_turn()
            text = transcribe(audio)
            if not text:
                speaker.t_turn = None
                continue
            print(f"🧑 You: {text}")
            if re.search(r"\b(goodbye|bye bye|exit|quit)\b", text.lower()):
                speaker.say("Goodbye!")
                speaker.wait()
                break

            # ---- RAG: retrieve relevant facts for this question ----
            context = ""
            try:
                t_rag = time.time()
                hits = rag.retrieve(text)
                print(f"   [RAG {time.time() - t_rag:.3f}s, scores={[round(s, 2) for _, s in hits]}]")
                context = "\n".join(c for c, _ in hits)
            except Exception as e:
                print(f"   [RAG error] {e}")

            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": reply(history, speaker, context)})
            if len(history) > 9:
                history = [history[0]] + history[-8:]
        except KeyboardInterrupt:
            print("\nBye!")
            break


if __name__ == "__main__":
    main()
