import qdrant
from qdrant_client import QdrantClient
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# # Initialize Gemini client (make sure your API key is set)
gemini_client = genai.Client(api_key=gemini_api_key)  # Replace with your key

def ask_with_context(
    qdrant_client: QdrantClient,
    question: str,
    collection_name: str = "OptiBot",
    top_k: int = 5,
    task_type: str = "QUESTION_ANSWERING",
    model: str = "gemini-2.5-flash"
) -> str:
    """
    Search Qdrant for relevant context and ask Gemini to answer the question.
    """
    # 1. Search Qdrant for relevant chunks
    results = qdrant.ask_qdrant(
        qdrant_client,
        query=question,
        collection_name=collection_name,
        top_k=top_k,
        task_type=task_type
    )
    if not results:
        return "Sorry, I couldn't find relevant information."

    # 2. Build context from Qdrant results
    context = "\n\n".join(
        f"File: {r['file']} | Chunk: {r['chunk_index']}\n{r['text']}" for r in results
    )

    # 3. Compose prompt for Gemini
    prompt = ( 
        "You are OptiBot, the customer-support bot for OptiSigns.com."
        "- Tone: helpful, factual, concise."
        "- Only answer using the uploaded docs."
        "- Max 5 bullet points; else link to the doc."
        "- Cite up to 3 \"Article URL:\" lines per reply.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    # 4. Get answer from Gemini
    response = gemini_client.models.generate_content(
        model=model,
        contents=[prompt]
    )
    # The response object may vary depending on the Gemini SDK version
    # Adjust the following line if needed:
    return response.text if hasattr(response, "text") else str(response)

# Example usage:
if __name__ == "__main__":
    client = QdrantClient("http://localhost:6333")
    question = input("Ask your question: ")
    answer = ask_with_context(client, question)
    print("\nGemini Answer:\n", answer)