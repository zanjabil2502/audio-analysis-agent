import os

from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI

load_dotenv()

llm = OpenAI(
    model=os.environ.get("LLM_MODEL", "gpt-4.1-mini"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0.1,
)
