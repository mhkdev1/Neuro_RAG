from pathlib import Path

# Diretório principal do projeto
BASE_DIR = Path(__file__).resolve().parent

# Diretórios de entrada e armazenamento
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Configurações do Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "embeddinggemma"
GENERATION_MODEL = "qwen3:4b"

# Configurações do RAG
COLLECTION_NAME = "neuro_rag"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

# Tempo limite das requisições ao Ollama, em segundos
OLLAMA_TIMEOUT = 120