import asyncio
from agents import ModelSettings
from agents.models.openai_chatcompletions import ModelTracing
from src.chatbot_agents.configuration import model
from src.chatbot_agents.textbook_agent import retrieve_content

prompt = (
    "You are a textbook Q&A assistant. Answer the user's question using only the provided textbook context. "
    "If the context does not contain enough information, say 'This topic is not covered in the textbook.'\n\n"
    "Question: what is this book about\n\nContext:\n"
    + retrieve_content('what is this book about')
)

async def main():
    result = await model.get_response(
        system_instructions="You are a textbook Q&A assistant. Answer only from the provided context.",
        input=prompt,
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
    )
    print(type(result))
    print(result)
    print('output_text', getattr(result, 'output_text', None))
    print('output', getattr(result, 'output', None))
    print('debug', getattr(result, 'debug', None))
    print('to_dict', getattr(result, 'to_dict', lambda: None)())

asyncio.run(main())
