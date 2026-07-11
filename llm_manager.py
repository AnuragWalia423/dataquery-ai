# llm_manager.py
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from config import Config

logger = logging.getLogger(__name__)


class LLMManager:
    """Provider-agnostic LLM wrapper with automatic failover.

    Gemini is the primary provider. If a call to Gemini raises for any
    reason (quota exhaustion, rate limiting, transient API/service
    errors, auth issues, etc.), the same request is transparently
    retried against Groq instead.

    Callers only ever see `LLMManager.invoke(prompt)` and a normal
    LangChain message response back — they don't need to know, or
    care, which provider actually answered. This keeps the rest of
    the app (sql_agent.py) free of try/except blocks for individual
    providers.
    """

    def __init__(self):
        self.gemini = ChatGoogleGenerativeAI(
            model=Config.DEFAULT_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=Config.TEMPERATURE
        )

        self.groq = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.GROQ_MODEL,
            temperature=Config.TEMPERATURE
        )

        # Track which provider answered the most recent request. Handy
        # for logging/telemetry and for showing in a UI ("answered by: groq").
        self.last_provider = None

    def invoke(self, prompt):
        """Invoke the primary LLM, falling back to the secondary on failure.

        `prompt` can be anything a LangChain chat model accepts: a plain
        string, a list of BaseMessage objects, or the output of
        ChatPromptTemplate.format_messages(...).

        Raises the fallback provider's exception if both providers fail,
        so callers can still surface a real error instead of failing silently.
        """
        try:
            response = self.gemini.invoke(prompt)
            self.last_provider = "gemini"
            return response
        except Exception as e:
            print("=" * 60)
            print("Gemini FAILED")
            print(e)
            print("Falling back to Groq...")
            print("=" * 60)


        try:
            response = self.groq.invoke(prompt)
            self.last_provider = "groq"

            print("=" * 60)
            print("Groq SUCCESS")
            print(response.content)
            print("=" * 60)

            return response
        except Exception as e:
            logger.error(f"Groq fallback also failed: {e}")
            raise