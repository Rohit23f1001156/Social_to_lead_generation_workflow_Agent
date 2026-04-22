import os
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

def _ensure_event_loop():
    """
    Surgical patch for Python 3.13 + LangGraph.
    Ensures the background thread has an active event loop before calling the Gemini SDK.
    """
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

def get_llm_response(user_input: str) -> str:
    """Standard conversational response."""
    _ensure_event_loop() # Apply the patch
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY not found."

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.7, 
            google_api_key=api_key
        )
        message = HumanMessage(content=user_input)
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

def detect_intent(user_input: str) -> str:
    """Strictly classifies user input into 'greeting', 'query', or 'high_intent'."""
    _ensure_event_loop() # Apply the patch
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "error"

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.0,
            google_api_key=api_key
        )

        system_prompt = """You are a strict intent classification engine.
You must classify the user's input into EXACTLY one of these three categories:
- greeting
- query
- high_intent

CRITICAL RULES:
1. Output MUST be ONLY one word (no sentences, no punctuation).
2. Output must be strictly one of the three labels.
3. No explanations allowed.

INTENT DEFINITIONS:
greeting: Simple greetings, casual chat, OR users wanting to cancel, stop, say "no", or exit.
query: Asking about product, features, OR PRICING. Informational intent only.
high_intent: User shows explicit intent to buy, sign up, or get started.

FEW-SHOT EXAMPLES:
Input: "hello"
Output: greeting
Input: "what is the price of the pro plan?"
Output: query
Input: "tell me about your pricing"
Output: query
Input: "I want to sign up"
Output: high_intent
Input: "how does this work?"
Output: query
"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f'Input: "{user_input}"\nOutput:')
        ]
        
        response = llm.invoke(messages)
        intent = response.content.strip().lower()
        
        valid_intents = ["greeting", "query", "high_intent"]
        if intent not in valid_intents:
            return "query"
            
        return intent

    except Exception as e:
        print(f"Intent detection error: {e}")
        return "query"

def rag_response(user_input: str, context: str) -> str:
    """Generates a response strictly based on the provided context."""
    _ensure_event_loop() # Apply the patch
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "error"

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.0,
            google_api_key=api_key
        )

        system_prompt = f"""You are an AutoStream assistant.

STRICT RULES:
- Answer ONLY from context.
- Be concise and factual.
- If unsure → say EXACTLY: "I don't have that information."
- Do NOT invent information.

CONTEXT:
{context}
"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]
        
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        print(f"RAG Response error: {e}")
        return "An error occurred generating the answer."


def validate_lead_data(user_input: str, expected_field: str) -> str:
    """Uses LLM to validate and format lead data. Returns 'INVALID' if garbage."""
    _ensure_event_loop()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: return "INVALID"

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            temperature=0.0, 
            google_api_key=api_key
        )
        
        system_msg = "You are a strict data validation agent."
        user_msg = f"""The user was asked for their {expected_field}.
Input: "{user_input}"

RULES:
1. If it is a valid {expected_field}, extract and format it properly (e.g., 'YT' -> 'YouTube', 'IG' -> 'Instagram', 'X' -> 'X (formerly Twitter)'). Output ONLY the formatted value.
2. If the user is arguing, refusing, saying 'no', or typing nonsense, output EXACTLY: INVALID."""

        # Explicitly passing both System and Human messages to satisfy the API requirement
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg)
        ]
        
        res = llm.invoke(messages)
        return res.content.strip()
        
    except Exception as e:
        print(f"[Debug] Validator Error: {e}")
        return "INVALID"