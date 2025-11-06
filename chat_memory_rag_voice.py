# chat_memory_rag_voice.py

import sys
import time
import struct
import sqlite3
import re
import logging
from typing import List, Tuple, Dict

import requests
import speech_recognition as sr
import pyttsx3
import sqlite_vec
from sentence_transformers import SentenceTransformer

# ========= Logging Setup =========
LOG_FILE = "chat_memory.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()]
)

# ========= RAG / Memory Config =========
DB_FILE = "chat_memory.db"
VEC_TABLE = "messages_vec"
META_TABLE = "messages_meta"
FACTS_TABLE = "facts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384
TOP_K = 5
MAX_SNIPPET_CHARS = 400

# ========= Ollama / TTS Config =========
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"
OLLAMA_TEMPERATURE = 0.0
OLLAMA_MAX_TOKENS = 20
OLLAMA_STREAM = False
HTTP_TIMEOUT = (5, 60)

# ========= Mic tuning =========
PHRASE_TIME_LIMIT = 7
AMBIENT_NOISE_DURATION = 0.4

# ========= Embedding Helpers =========
_model = None
def get_model():
    global _model
    if _model is None:
        logging.info("Loading embedding model (preload)...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logging.info("Embedding model loaded and ready.")
    return _model

def embed(text: str) -> List[float]:
    return get_model().encode(text, normalize_embeddings=True).tolist()

def serialize_f32(vec: List[float]) -> bytes:
    return struct.pack("%sf" % len(vec), *vec)

# ========= SQLite Setup =========
def ensure_db():
    conn = sqlite3.connect(DB_FILE)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        f"""CREATE VIRTUAL TABLE IF NOT EXISTS {VEC_TABLE}
            USING vec0(embedding float[{EMBEDDING_DIMS}])"""
    )
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {META_TABLE} (
                rowid INTEGER PRIMARY KEY,
                ts    REAL NOT NULL,
                role  TEXT NOT NULL,
                text  TEXT NOT NULL
            )"""
    )
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {FACTS_TABLE} (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )"""
    )
    conn.commit()
    conn.close()
    logging.info("Database initialized and verified.")

# ========= Warm-Up Function =========
def warmup_environment():
    """
    Preloads heavy resources before microphone loop starts:
      - Embedding model
      - SQLite + vec0 extension
      - Basic DB connection check
      - Tiny dummy embedding to prime PyTorch
    """
    start = time.time()
    logging.info("Performing system warm-up...")

    ensure_db()
    model = get_model()
    _ = model.encode("warmup test", normalize_embeddings=True)
    logging.info("Model embedding warm-up complete.")

    conn = sqlite3.connect(DB_FILE)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(f"SELECT count(*) FROM {META_TABLE}")
    conn.close()
    logging.info("SQLite vec0 warm-up complete.")

    logging.info(f"Warm-up finished in {time.time() - start:.2f} seconds.")

# ========= Memory & RAG Functions =========
def remember(conn: sqlite3.Connection, role: str, text: str) -> None:
    vec = embed(text)
    cur = conn.execute(
        f"INSERT INTO {VEC_TABLE}(embedding) VALUES (?)",
        (serialize_f32(vec),)
    )
    rid = cur.lastrowid
    conn.execute(
        f"INSERT INTO {META_TABLE}(rowid, ts, role, text) VALUES (?, ?, ?, ?)",
        (rid, time.time(), role, text)
    )
    conn.commit()
    logging.info(f"Remembered message ({role}): {text[:80]}...")

def recall(conn: sqlite3.Connection, query: str, k: int = TOP_K) -> list[Tuple[float, str, str]]:
    qv = embed(query)
    neighbors = conn.execute(
        f"""SELECT rowid, distance
            FROM {VEC_TABLE}
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?""",
        (serialize_f32(qv), k),
    ).fetchall()
    if not neighbors:
        logging.info("No similar memories recalled.")
        return []
    ids = ",".join(str(rid) for rid, _ in neighbors)
    meta = conn.execute(
        f"""SELECT rowid, role, text FROM {META_TABLE} WHERE rowid IN ({ids})"""
    ).fetchall()
    meta_by_id = {rowid: (role, text) for rowid, role, text in meta}
    logging.info(f"Recalled {len(neighbors)} memory snippets.")
    return [(dist, *meta_by_id.get(rid, ("unknown", ""))) for rid, dist in neighbors]

def format_memory_snippets(hits: list[Tuple[float, str, str]]) -> str:
    if not hits:
        return "None."
    lines = []
    for dist, role, text in hits:
        snippet = (text[:MAX_SNIPPET_CHARS] + "…") if len(text) > MAX_SNIPPET_CHARS else text
        lines.append(f"- ({role}, d={dist:.3f}) {snippet}")
    return "\n".join(lines)

# ========= Facts =========
def upsert_fact(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        f"""INSERT INTO {FACTS_TABLE}(key, value, updated_at)
             VALUES(?, ?, ?)
             ON CONFLICT(key) DO UPDATE SET
               value=excluded.value,
               updated_at=excluded.updated_at
        """,
        (key, value, time.time()),
    )
    conn.commit()
    logging.info(f"Fact updated: {key} = {value}")

def get_all_facts(conn: sqlite3.Connection) -> Dict[str, str]:
    rows = conn.execute(f"SELECT key, value FROM {FACTS_TABLE}").fetchall()
    return {k: v for k, v in rows}

def extract_and_store_facts(conn: sqlite3.Connection, user_text: str) -> None:
    text = user_text.strip()
    low = text.lower()
    m = re.search(r"\bmy name is\s+([A-Z][a-zA-Z\-']+)", text)
    if m:
        upsert_fact(conn, "name", m.group(1))
    else:
        m2 = re.search(r"\bcall me\s+([A-Z][a-zA-Z\-']+)", text, flags=re.IGNORECASE)
        if m2:
            upsert_fact(conn, "name", m2.group(1))
    m3 = re.search(r"\bmy age is\s+(\d{1,3})\b", low)
    if m3:
        upsert_fact(conn, "age", m3.group(1))
    else:
        m4 = re.search(r"\b(i'?m|i am)\s+(\d{1,3})\b", low)
        if m4:
            upsert_fact(conn, "age", m4.group(2))
        else:
            m5 = re.search(r"\bno[, ]+\s*it'?s\s+(\d{1,3})\b", low)
            if m5:
                upsert_fact(conn, "age", m5.group(1))

# ========= Prompt builder =========
def build_rag_prompt(memory_block: str, user_text: str, facts: dict) -> str:
    facts_block = "None."
    if facts:
        facts_block = "\n".join([f"- {k}: {v}" for k, v in facts.items()])
    return (
        "System role definition:\n"
        "You are an AI assistant named AVA, speaking to the USER (the human).\n"
        "You are distinct from the USER and never share the USER’s attributes.\n"
        "Facts listed below describe the USER only.\n"
        "Maintain correct conversational perspective at all times:\n"
        " - When the USER says 'my', interpret that as referring to the USER.\n"
        " - When the USER says 'your', interpret that as referring to AVA.\n"
        "Do not copy or claim any USER fact as your own identity.\n\n"
        f"USER facts:\n{facts_block}\n\n"
        f"Recalled conversation snippets:\n{memory_block}\n\n"
        f"USER says: {user_text}\n"
        "Respond in one short, natural sentence addressed to the USER."
    )

# ========= Main Loop =========
def listen_and_recognize():
    logging.info(f"Starting AVA Voice Assistant on Python {sys.version}")
    ensure_db()

    recognizer = sr.Recognizer()
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('volume', 1.0)

    with sr.Microphone() as source:
        logging.info("Calibrating ambient noise level...")
        recognizer.adjust_for_ambient_noise(source, duration=AMBIENT_NOISE_DURATION)
        logging.info(f"Energy threshold set to: {recognizer.energy_threshold:.2f}")
        print("Get Ready to Say something when I say I am Listening...")

        while True:
            try:
                print("Listening...")
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=PHRASE_TIME_LIMIT)

                if len(audio.frame_data) == 0:
                    logging.warning("Captured empty audio; skipping.")
                    continue

                text = recognizer.recognize_google(audio)
                logging.info(f"User said: {text}")

                conn = sqlite3.connect(DB_FILE)
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)

                extract_and_store_facts(conn, text)
                hits = recall(conn, text, TOP_K)
                facts = get_all_facts(conn)
                memory_block = format_memory_snippets(hits)

                constrained_prompt = build_rag_prompt(memory_block, text, facts)
                payload = {
                    "prompt": constrained_prompt,
                    "model": OLLAMA_MODEL,
                    "stream": OLLAMA_STREAM,
                    "temperature": OLLAMA_TEMPERATURE,
                    "max_tokens": OLLAMA_MAX_TOKENS
                }

                response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=HTTP_TIMEOUT)
                text_response = "Sorry, there was an error from the model."

                if response.status_code == 200:
                    result = response.json()
                    text_response = result.get("response", "").strip()
                    logging.info(f"Ollama response: {text_response}")
                else:
                    logging.error(f"Ollama error {response.status_code}: {response.text}")
                    text_response = f"Error {response.status_code}"

                remember(conn, "user", text)
                remember(conn, "assistant", text_response)
                conn.close()

                # Voice output
                tts_engine.say(text_response)
                tts_engine.runAndWait()
                tts_engine.stop()

            except sr.UnknownValueError:
                logging.warning("Speech not recognized (low confidence).")
                continue
            except sr.RequestError as e:
                logging.error(f"Speech Recognition API error: {e}")
                continue
            except KeyboardInterrupt:
                logging.info("User interrupted. Exiting.")
                break
            except Exception as e:
                logging.exception(f"Unexpected error: {e}")
                continue

    try:
        tts_engine.stop()
    except Exception:
        pass

# ========= Entry Point =========
if __name__ == "__main__":
    warmup_environment()
    listen_and_recognize()
