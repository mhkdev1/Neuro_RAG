from hashlib import sha256
from pathlib import Path

import chromadb
import fitz
import requests

from config import (
    DOCUMENTS_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    OLLAMA_TIMEOUT,
)


EMBEDDING_BATCH_SIZE = 16


def split_text(text: str) -> list[str]:
    """Divide um texto em chunks, mantendo alguma sobreposição."""

    text = " ".join(text.split())

    if not text:
        return []

    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE deve ser maior que zero.")

    if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP deve ser menor que CHUNK_SIZE.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # Tenta terminar o chunk em um espaço, evitando cortar palavras.
        if end < len(text):
            break_at = text.rfind(" ", start, end)

            if break_at > start:
                end = break_at

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def extract_chunks_from_pdf(pdf_path: Path) -> list[dict]:
    """Extrai o texto de cada página e cria chunks com suas fontes."""

    records = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text")

            page_chunks = split_text(page_text)

            for chunk_index, chunk in enumerate(page_chunks, start=1):
                unique_text = (
                    f"{pdf_path.name}|{page_number}|"
                    f"{chunk_index}|{chunk}"
                )

                chunk_id = sha256(
                    unique_text.encode("utf-8")
                ).hexdigest()

                records.append(
                    {
                        "id": chunk_id,
                        "text": chunk,
                        "metadata": {
                            "source": pdf_path.name,
                            "page": page_number,
                            "chunk_index": chunk_index,
                        },
                    }
                )

    return records


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Solicita ao Ollama os embeddings de uma lista de textos."""

    url = f"{OLLAMA_BASE_URL}/api/embed"

    try:
        response = requests.post(
            url,
            json={
                "model": EMBEDDING_MODEL,
                "input": texts,
            },
            timeout=OLLAMA_TIMEOUT,
        )

        response.raise_for_status()

    except requests.exceptions.ConnectionError as error:
        raise RuntimeError(
            "Não foi possível conectar ao Ollama. "
            "Verifique se ele está instalado e em execução."
        ) from error

    except requests.exceptions.Timeout as error:
        raise RuntimeError(
            "O Ollama demorou demais para responder."
        ) from error

    except requests.exceptions.HTTPError as error:
        details = (
            error.response.text
            if error.response is not None
            else str(error)
        )
        raise RuntimeError(
            f"Erro HTTP retornado pelo Ollama: {details}"
        ) from error

    data = response.json()
    embeddings = data.get("embeddings")

    if embeddings is None or len(embeddings) != len(texts):
        raise RuntimeError(
            "O número de embeddings retornados não corresponde "
            "ao número de textos enviados."
        )

    return embeddings


def main() -> None:
    """Executa a ingestão de todos os PDFs da pasta de documentos."""

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"Nenhum PDF encontrado em: {DOCUMENTS_DIR}")
        print("Coloque pelo menos um PDF nessa pasta.")
        return

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )

    for pdf_path in pdf_files:
        print(f"\nProcessando: {pdf_path.name}")

        records = extract_chunks_from_pdf(pdf_path)

        if not records:
            print(
                "Nenhum texto extraído. "
                "O PDF pode estar vazio ou conter apenas imagens."
            )
            continue

        for start in range(0, len(records), EMBEDDING_BATCH_SIZE):
            batch = records[start:start + EMBEDDING_BATCH_SIZE]

            texts = [record["text"] for record in batch]
            embeddings = generate_embeddings(texts)

            collection.upsert(
                ids=[record["id"] for record in batch],
                documents=texts,
                metadatas=[record["metadata"] for record in batch],
                embeddings=embeddings,
            )

            processed = min(
                start + len(batch),
                len(records),
            )

            print(f"Trechos processados: {processed}/{len(records)}")

    print("\nIndexação concluída.")
    print(f"Total de trechos na coleção: {collection.count()}")


if __name__ == "__main__":
    main()