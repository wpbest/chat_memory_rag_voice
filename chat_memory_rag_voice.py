# chat_memory_rag_voice.py

import sys
import time
import struct
import sqlite3
import re
from typing import List, Tuple, Dict

import requests
import speech_recognition as sr
import pyttsx3
import sqlite_vec
from sentence_transformers import SentenceTransformer

# ========= RAG / Memory Config =========
DB_FILE = "chat_memory.db"
VEC_TABLE = "messages_vec"    # vec0 table: embedding only
META_TABLE = "messages_meta"  # regular table: ts, role, text
FACTS_TABLE = "facts"         # authoritative key/value memory
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384
TOP_K = 5
MAX_SNIPPET_CHARS = 400

# ========= Ollama / TTS Config (kept like your original) =========
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"      # your default
OLLAMA_TEMPERATURE = 0.0
OLLAMA_MAX_TOKENS = 20
OLLAMA_STREAM = False           # keep exactly like your working original
HTTP_TIMEOUT = (5, 60)          # (connect, read) seconds

# ========= Mic tuning (avoid blocking forever) =========
PHRASE_TIME_LIMIT = 7           # seconds per utterance
AMBIENT_NOISE_DURATION = 0.4    # seconds to calibrate

# ========= Embedding Helpers =========
_model = None
def get_model():
    global _model
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed(text: str) -> List[float]:
    return get_model().encode(text, normalize_embeddings=True).tolist()

def serialize_f32(vec: List[float]) -> bytes:
    return struct.pack("%sf" % len(vec), *vec)

# ========= SQLite Setup =========
def ensure_db():
    conn = sqlite3.connect(DB_FILE)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)  # loads the bundled extension
    conn.enable_load_extension(False)

    # vec0 table must contain ONLY the embedding column
    conn.execute(
        f"""CREATE VIRTUAL TABLE IF NOT EXISTS {VEC_TABLE}
            USING vec0(embedding float[{EMBEDDING_DIMS}])"""
    )

    # normal table holds metadata and shares rowid with vec table
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {META_TABLE} (
                rowid INTEGER PRIMARY KEY,
                ts    REAL NOT NULL,
                role  TEXT NOT NULL,
                text  TEXT NOT NULL
            )"""
    )

    # facts table for authoritative key/value pairs
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {FACTS_TABLE} (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )"""
    )

    conn.commit()
    conn.close()

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
        return []
    ids = ",".join(str(rid) for rid, _ in neighbors)
    meta = conn.execute(
        f"""SELECT rowid, role, text
            FROM {META_TABLE}
            WHERE rowid IN ({ids})"""
    ).fetchall()
    meta_by_id = {rowid: (role, text) for rowid, role, text in meta}
    return [(dist, *meta_by_id.get(rid, ("unknown", ""))) for rid, dist in neighbors]

def format_memory_snippets(hits: list[Tuple[float, str, str]]) -> str:
    if not hits:
        return "None."
    lines = []
    for dist, role, text in hits:
        snippet = (text[:MAX_SNIPPET_CHARS] + "…") if len(text) > MAX_SNIPPET_CHARS else text
        lines.append(f"- ({role}, d={dist:.3f}) {snippet}")
    return "\n".join(lines)

# ========= Facts (authoritative memory) =========
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

def get_all_facts(conn: sqlite3.Connection) -> Dict[str, str]:
    rows = conn.execute(f"SELECT key, value FROM {FACTS_TABLE}").fetchall()
    return {k: v for k, v in rows}

def extract_and_store_facts(conn: sqlite3.Connection, user_text: str) -> None:
    """
    Minimal robust extraction for name/age, including corrections.
    """
    text = user_text.strip()
    low = text.lower()

    # Name: "my name is X" / "call me X"
    m = re.search(r"\bmy name is\s+([A-Z][a-zA-Z\-']+)", text)
    if m:
        upsert_fact(conn, "name", m.group(1))
    else:
        m2 = re.search(r"\bcall me\s+([A-Z][a-zA-Z\-']+)", text, flags=re.IGNORECASE)
        if m2:
            upsert_fact(conn, "name", m2.group(1))

    # Age: "my age is 23"
    m3 = re.search(r"\bmy age is\s+(\d{1,3})\b", low)
    if m3:
        upsert_fact(conn, "age", m3.group(1))
    else:
        # "I'm 23" / "I am 23"
        m4 = re.search(r"\b(i'?m|i am)\s+(\d{1,3})\b", low)
        if m4:
            upsert_fact(conn, "age", m4.group(2))
        else:
            # Correction: "no it's 23"
            m5 = re.search(r"\bno[, ]+\s*it'?s\s+(\d{1,3})\b", low)
            if m5:
                upsert_fact(conn, "age", m5.group(1))

# ========= Prompt builder (updated for assistant identity) =========
def build_rag_prompt(memory_block: str, user_text: str, facts: dict) -> str:
    """
    Builds a perspective-aware RAG prompt without hard-coding specific facts.
    It teaches the model the correct speaker roles and pronoun mapping dynamically.
    """
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
        " - When responding, use pronouns consistent with natural dialogue "
        "(e.g., 'your' for USER attributes, 'my' for your own attributes).\n"
        "Do not copy or claim any USER fact as your own identity or property.\n\n"
        f"USER facts:\n{facts_block}\n\n"
        f"Recalled conversation snippets:\n{memory_block}\n\n"
        f"USER says: {user_text}\n"
        "Respond in one short, natural sentence addressed to the USER."
    )


# ========= Main loop =========
def listen_and_recognize():
    print(f"Hello, I'm running Python version {sys.version}")
    ensure_db()  # create schema once

    # Persistent recognizer and TTS engine (avoid per-turn init/teardown)
    recognizer = sr.Recognizer()
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('volume', 1.0)

    print("Say something...")
    while True:
        try:
            # LISTEN (bounded time)
            with sr.Microphone() as source:
                print("Listening...")
                recognizer.adjust_for_ambient_noise(source, duration=AMBIENT_NOISE_DURATION)
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=PHRASE_TIME_LIMIT)

            # STT
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")

            # Open fresh DB connection per turn
            conn = sqlite3.connect(DB_FILE)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)

            # Update authoritative facts
            extract_and_store_facts(conn, text)

            # RAG recall BEFORE storing current turn
            hits = recall(conn, text, TOP_K)
            facts = get_all_facts(conn)
            memory_block = format_memory_snippets(hits)

            # Build prompt (facts override memory)
            constrained_prompt = build_rag_prompt(memory_block, text, facts)

            # Call Ollama
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
                print("Response from Ollama:", text_response)
            else:
                print("Error:", response.status_code, response.text)
                text_response = f"Error {response.status_code}"

            # Persist both sides
            remember(conn, "user", text)
            remember(conn, "assistant", text_response)

            conn.close()

            # TTS
            tts_engine.say(text_response)
            tts_engine.runAndWait()
            tts_engine.stop()

            time.sleep(0.15)

        except sr.UnknownValueError:
            print("Sorry, I couldn't understand what you said.")
            time.sleep(0.2)
            continue
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            time.sleep(0.3)
            continue
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            time.sleep(0.3)
            continue

    try:
        tts_engine.stop()
    except:
        pass

if __name__ == "__main__":
    listen_and_recognize()
