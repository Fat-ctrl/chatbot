# AI-Powered Q&A for Web Documentation

This project provides an end-to-end pipeline for:

- Scraping web documentation
- Chunking and embedding content
- Storing embeddings in a Qdrant vector database
- Serving a chatbot UI (powered by Google Gemini and Gradio) for semantic search and question answering

All components are completely free to use, but please note the following limitations:

- Gemini Free Tier Embeddings: When using the Gemini free tier for text embedding during chunking, a cooldown of ~60 seconds is required after processing several chunks to avoid hitting rate limits.

- API Throttling: Extended or continuous use of the API may result in temporary refusal of further requests. In such cases, additional waiting time is needed before requests are accepted again.

---

## Features

- **Automated Scraping:** Fetches articles from provided source through API.
- **Delta Detection:** Only new or updated articles are processed and embedded.
- **Chunking & Embedding:** Splits articles into manageable chunks and generates embeddings using Google Gemini.
- **Vector Search:** Stores embeddings in Qdrant for fast semantic search.
- **Chatbot UI:** Gradio-based web interface for natural language Q&A over the documentation.
- **Dockerized:** Easily deployable with Docker and Docker Compose.
- **Daily Scheduling:** Designed for scheduled runs (e.g., daily) to keep the knowledge base up to date.
- **Logging:** Tracks added, updated, and skipped articles per run.

---

## Quickstart

### 1. Clone the Repository

```bash
git clone <repository_link>
cd test
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
QDRANT_URL=http://localhost:6333
API_URL=https://<site>/api/v2/help_center/{locale}/articles
```

### 3. Build and Run with Docker Compose

```bash
docker compose up --build
```

- The Gradio chatbot UI will be available at [http://localhost:8080](http://localhost:8080)
- Qdrant vector DB will be available at [http://localhost:6333](http://localhost:6333)

### 4. Usage

- Visit the chatbot UI and ask questions about the documentation.
- The backend will search for relevant content and generate answers using Gemini.

---

## Project Structure

```
.
├── main.py             # Scraper, delta detection, and Qdrant uploader
├── scraper.py          # Scrapes OptiSigns articles and saves as markdown
├── chunker.py          # Text chunking and embedding logic
├── qdrant.py           # Qdrant collection and search utilities
├── chatbot.py          # Chatbot logic (retrieval-augmented generation)
├── gradio_app.py       # Gradio UI for the chatbot
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker build instructions for chatbot container
├── docker-compose.yml  # Multi-container orchestration
└── .env.sample         # API keys and config example .env
```

---

## Scheduling (Production)

- Use DigitalOcean App Platform or any scheduler to run `python main.py` daily for fresh data.
- The chatbot UI can run continuously as a web service.

---

## Logs

- Each run logs added, updated, and skipped articles in `job_log.txt`.
- You can view logs in the container or via your cloud platform's log viewer.

---

## Requirements

- Python 3.11+
- Docker & Docker Compose
- Google Gemini API key

---

## License

MIT License

---

## Acknowledgements

- [Qdrant](https://qdrant.tech/)
- [Google Gemini](https://ai.google.dev/)
- [Gradio](https://gradio.app/)
- Inspiration from [Gemini API Cookbook](https://github.com/google-gemini/cookbook/blob/main/examples/qdrant/Movie_Recommendation.ipynb)
---
