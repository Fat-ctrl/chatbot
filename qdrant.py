from qdrant_client import QdrantClient, models
import time
from tqdm import tqdm
import chunker
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

database_url = os.getenv("QDRANT_URL")

def create_collection(
    collection_name: str = "OptiBot", 
    vector_size: int = 3072, 
    distance_metric: models.Distance = models.Distance.COSINE
):
    """
    Create a Qdrant collection with specified parameters.
    
    Args:
        collection_name: Name of the collection to create.
        vector_size: Size of the embedding vectors.
        distance_metric: Distance metric for the collection.
    
    Returns:
        QdrantClient instance with the created collection.
    """
    
    qdrant_client = QdrantClient(database_url)

    # In case someone tries running the whole notebook again they would want to create the collection again

    try:
        qdrant_client.delete_collection(collection_name=collection_name)
        print(f"Existing collection '{collection_name}' deleted.")
    except Exception as e:
        print(f"Error deleting collection (it might not exist): {e}")

    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=distance_metric
            )
        )
        print(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        print(f"Error creating collection: {e}")
    
    return qdrant_client

def create_payload(
    qdrant_client,
    collection_name: str = "OptiBot",
    articles_dir: str = "./articles",
    batch_size: int = 25,
    qdrant_batch_size: int = 3072
):
    """
    Chunk all markdown files in the articles folder, embed, and upsert to Qdrant.
    """
    # List all markdown files in the articles directory
    articles_path = Path(articles_dir)
    md_files = list(articles_path.glob("*.md"))
    print(f"Found {len(md_files)} markdown files in {articles_dir}")

    points_to_upsert_buffer = []
    total_chunks = 0
    total_upserted = 0
    chunk_id = 1

    for md_file in tqdm(md_files, desc="Processing Markdown Files"):
        text = chunker.load_markdown_text(str(md_file))
        chunks = chunker.chunk_text(text)
        chunk_texts = [c for c in chunks if c.strip()]
        if not chunk_texts:
            continue

        # Process in batches for embedding
        for i in range(0, len(chunk_texts), batch_size):
            batch_chunks = chunk_texts[i : i + batch_size]
            batch_embeddings = chunker.get_embeddings_batch(batch_chunks, task_type="QUESTION_ANSWERING")

            if batch_embeddings and len(batch_embeddings) == len(batch_chunks):
                for j, embedding in enumerate(batch_embeddings):
                    point = models.PointStruct(
                        id=chunk_id,
                        vector=embedding.values if hasattr(embedding, "values") else embedding,
                        payload={
                            "file": md_file.name,
                            "chunk_index": i + j,
                            "text": batch_chunks[j]
                        }
                    )
                    points_to_upsert_buffer.append(point)
                    chunk_id += 1
                total_chunks += len(batch_chunks)
            else:
                print(f"Embedding failed for {md_file.name} batch {i}-{i+len(batch_chunks)}")

            # Upsert if buffer is large enough
            if len(points_to_upsert_buffer) >= qdrant_batch_size:
                try:
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=points_to_upsert_buffer,
                        wait=False
                    )
                    total_upserted += len(points_to_upsert_buffer)
                    points_to_upsert_buffer = []
                except Exception as e:
                    print(f"Error upserting chunk to Qdrant: {e}")
                    points_to_upsert_buffer = []
                    time.sleep(5)

    # Upsert any remaining points
    if points_to_upsert_buffer:
        print(f"Upserting final {len(points_to_upsert_buffer)} points.")
        try:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points_to_upsert_buffer,
                wait=True
            )
            total_upserted += len(points_to_upsert_buffer)
        except Exception as e:
            print(f"Error upserting final chunk: {e}")

    print("Batch embedding and indexing finished.")
    print(f"Total chunks processed: {total_chunks}")
    print(f"Total points upserted: {total_upserted}")

def ask_qdrant(
    qdrant_client,
    query: str,
    collection_name: str = "OptiBot",
    top_k: int = 5,
    task_type: str = "QUESTION_ANSWERING"
):
    """
    Search Qdrant for the most relevant article chunks to the query.

    Args:
        qdrant_client: The QdrantClient instance.
        query: The user's question.
        collection_name: The Qdrant collection name.
        top_k: Number of results to return.
        task_type: Embedding task type.

    Returns:
        List of dicts with file name, chunk index, and text.
    """
    # Embed the query
    query_embedding = chunker.get_embeddings_batch([query], task_type=task_type)
    if not query_embedding or not query_embedding[0]:
        print("Failed to embed query.")
        return []

    # Search Qdrant
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding[0].values if hasattr(query_embedding[0], "values") else query_embedding[0],
        limit=top_k,
        with_payload=True
    )

    # Format results
    results = []
    for hit in search_result:
        payload = hit.payload
        results.append({
            "file": payload.get("file"),
            "chunk_index": payload.get("chunk_index"),
            "text": payload.get("text"),
            "score": hit.score
        })
    return results