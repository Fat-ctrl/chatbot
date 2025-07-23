import gradio as gr
from qdrant_client import QdrantClient
import chatbot
import os
from dotenv import load_dotenv
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_client = QdrantClient(QDRANT_URL)

def chat_interface(question):
    # Input validation: check for empty input and length limit
    if not question or not question.strip():
        return "Please enter a valid question."
    if len(question) > 500:
        return "Your question is too long. Please limit it to 500 characters."
    # Basic sanitization: remove leading/trailing whitespace
    sanitized_question = question.strip()
    answer = chatbot.ask_with_context(qdrant_client, sanitized_question)
    return answer

demo = gr.Interface(
    fn=chat_interface,
    inputs=gr.Textbox(lines=2, label="Ask OptiBot a question about OptiSigns articles", max_lines=4, max_length=500),
    outputs=gr.Textbox(label="OptiBot Answer"),
    title="OptiBot Chatbot",
    description="Ask questions about OptiSigns documentation. Answers are generated using Gemini and Qdrant search."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)