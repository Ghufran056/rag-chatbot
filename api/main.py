from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schema import ChatRequest, ChatResponse
from src.chatbot_agents.textbook_agent import ask


app = FastAPI(
    title="Embodied Intelligence Textbook API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://ghufran056.github.io",
    ],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
def chat(message: ChatRequest):
    if not message.question.strip():
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    response = ask(message.question)

    return ChatResponse(
        answer=response
    )