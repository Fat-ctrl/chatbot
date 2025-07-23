import time
from google import genai
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

gemini_client = genai.Client(api_key=gemini_api_key)

def get_embeddings_batch(
    texts: list[str],
    task_type="QUESTION_ANSWERING",
    model_for_embedding: str = "gemini-embedding-001",
    max_retries: int = 5,
    retry_delay: int = 60
) -> list[list[float]] | None:
    """
    Generates embeddings for a batch of texts using Gemini API with retry and delay on 429 errors.

    Args:
        texts: A list of strings to embed.
        task_type: The task type for the embedding model.
        max_retries: Number of times to retry on 429 errors.
        retry_delay: Seconds to wait before retrying.

    Returns:
        A list of embedding vectors (list of floats), or None if a non-retryable error occurs.
    """
    if not texts:
        return []
    attempt = 0
    while attempt <= max_retries:
        try:
            response = gemini_client.models.embed_content(
                model=model_for_embedding,
                contents=texts,
                config={
                    "task_type": task_type,
                }
            )
            return response.embeddings
        except Exception as e:
            if hasattr(e, "code") and getattr(e, "code") == 429:
                print(f"429 RESOURCE_EXHAUSTED: Waiting {retry_delay} seconds before retrying... (attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                attempt += 1
                continue
            elif "RESOURCE_EXHAUSTED" in str(e):
                print(f"429 RESOURCE_EXHAUSTED: Waiting {retry_delay} seconds before retrying... (attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                attempt += 1
                continue
            else:
                print(f"An unexpected error occurred during embedding: {e}")
                return None
    print("Embedding batch failed after maximum retries.")
    return None

# Load text from markdown file 
def load_markdown_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

# Simple chunking 
def chunk_text(text: str, max_length: int = 3072) -> list[str]:
    """Split text into chunks of max_length characters without breaking lines."""
    lines = text.splitlines()
    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 <= max_length:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


