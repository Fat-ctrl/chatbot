import os
import hashlib
import json
from datetime import datetime
import scraper
import chunker, qdrant

HASH_DB = "article_hashes.json"
LOG_FILE = "job_log.txt"

def compute_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def load_hash_db():
    if os.path.exists(HASH_DB):
        with open(HASH_DB, "r") as f:
            return json.load(f)
    return {}

def save_hash_db(db):
    with open(HASH_DB, "w") as f:
        json.dump(db, f, indent=2)

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now().isoformat()} {msg}\n")

def main():
    # Scrape articles (will save .md files)
    scraper.main() 

    # Load previous hashes
    hash_db = load_hash_db()
    added, updated, skipped = 0, 0, 0

    # Prepare Qdrant
    client = qdrant.create_collection()
    articles_path = scraper.save_location
    md_files = [f for f in os.listdir(articles_path) if f.endswith(".md")]
    delta_files = []

    for fname in md_files:
        fpath = os.path.join(articles_path, fname)
        text = chunker.load_markdown_text(fpath)
        h = compute_hash(text)
        if fname not in hash_db:
            log(f"Added: {fname}")
            added += 1
            delta_files.append(fname)
        elif hash_db[fname] != h:
            log(f"Updated: {fname}")
            updated += 1
            delta_files.append(fname)
        else:
            skipped += 1

        hash_db[fname] = h

    # Only process new/updated files
    if delta_files:
        for fname in delta_files:
            fpath = os.path.join(articles_path, fname)
            text = chunker.load_markdown_text(fpath)
            chunks = chunker.chunk_text(text)
            embeddings = chunker.get_embeddings_batch(chunks, task_type='QUESTION_ANSWERING')
            # Upsert to Qdrant
            points = []
            for i, emb in enumerate(embeddings):
                points.append(qdrant.models.PointStruct(
                    id=hashlib.sha256(f"{fname}-{i}".encode()).hexdigest(),
                    vector=emb.values if hasattr(emb, "values") else emb,
                    payload={"file": fname, "chunk_index": i, "text": chunks[i]}
                ))
            client.upsert(collection_name="OptiBot", points=points)
    else:
        log("No new or updated articles to upload.")

    save_hash_db(hash_db)
    log(f"Summary: Added={added}, Updated={updated}, Skipped={skipped}")

    print(f"Job complete. See {LOG_FILE} for details.")

if __name__ == "__main__":
    main()