# chat_memory_rag_voice
A Python application AI-powered voice assistant that listens to your speech, generates intelligent replies using a local Ollama LLM (like Gemma 3), speaks the answers back, and unlike a typical chatbot it also remembers past conversations using Retrieval-Augmented Generation (RAG) with SQLite-vec.

🧠 Chat Memory RAG Voice Assistant

Chat Memory RAG Voice Assistant is a Python-based, AI-powered voice assistant that listens to your speech, generates intelligent replies using a local Ollama LLM (such as Gemma 3), speaks the answers aloud, and — unlike a typical chatbot — remembers past conversations using Retrieval-Augmented Generation (RAG) with SQLite-vec. The assistant converts your speech into text, stores and embeds each exchange in a lightweight local SQLite vector database, retrieves semantically related memories during each new query, and builds a context-aware prompt for the LLM. It then generates a concise, natural response and reads it back through text-to-speech. This system runs entirely on your own machine, keeping all data private and enabling full offline operation after the models are downloaded.

⚙️ Build & Run (VS Code on Windows)

Install prerequisites

Python 3.11+

Ollama
 (start it with ollama serve)

Visual Studio Code

Clone or open the project folder

git clone https://github.com/<your-repo-name>.git
cd ChatMemoryRAGVoiceAssistant


Create and activate a virtual environment

python -m venv .venv
.\.venv\Scripts\activate


Install required libraries

pip install --upgrade pip
pip install -r requirements.txt


Run inside VS Code

Open the folder in VS Code

Select the .venv Python interpreter (Ctrl+Shift+P → Python: Select Interpreter)

Create a simple launch configuration (Run → Add Configuration → Python File)

Press F5 or click ▶ Run to start

Start interacting

Speak into your microphone

The assistant will listen, think, respond, and remember

Features: Local + private, vector-based long-term memory, natural voice input/output, and full offline capability after setup.