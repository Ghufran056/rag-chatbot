from openai import AsyncOpenAI
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("OPENROUTER_API_KEY")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://openrouter.ai/api/v1",
)

model = OpenAIChatCompletionsModel(
    model="cohere/north-mini-code:free",
    openai_client=client
)