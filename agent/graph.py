from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import our existing brain, RAG, and tools
from utils.llm import detect_intent, get_llm_response, rag_response, validate_lead_data
from rag.retriever import retrieve_context
from tools.lead import mock_lead_capture

# 1. Define the State Structure
class AgentState(TypedDict):
    user_input: str
    intent: Optional[str]
    response: Optional[str]
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]
    stage: Optional[str]

# 2. Define the Nodes (The Actions)
def intent_node(state: AgentState):
    intent = detect_intent(state["user_input"])
    return {"intent": intent}

def greeting_node(state: AgentState):
    response = get_llm_response(state["user_input"])
    return {"response": response}

def rag_node(state: AgentState):
    context = retrieve_context(state["user_input"])
    print(f"\n[Debug] RAG Context Retrieved:\n{context}\n") # Debug visibility for interviews
    response = rag_response(state["user_input"], context)
    return {"response": response}

def lead_node(state: AgentState):
    stage = state.get("stage")
    user_input = state["user_input"]
    
    # Escape hatch: If the user explicitly wants to cancel during the flow
    if user_input.lower().strip() in ['cancel', 'stop', 'exit']:
        return {"stage": None, "intent": None, "response": "No problem! I've cancelled the signup process. Let me know if you need anything else."}

    if stage == "collecting_name":
        valid_name = validate_lead_data(user_input, "Name")
        # 🔥 FIX: Robust checking
        if "INVALID" in valid_name.upper():
            return {"response": "That doesn't look like a valid name. Let's try again, or type 'cancel' to stop."}
        return {"name": valid_name, "stage": "collecting_email", "response": f"Thanks {valid_name}! What's your email address?"}
    
    elif stage == "collecting_email":
        valid_email = validate_lead_data(user_input, "Email address")
        # 🔥 FIX: Robust checking
        if "INVALID" in valid_email.upper():
            return {"response": "That doesn't look like a valid email. Please provide a real email, or type 'cancel'."}
        return {"email": valid_email, "stage": "collecting_platform", "response": "Got it. Finally, which platform do you create content on (e.g., YouTube, Instagram)?"}
    
    elif stage == "collecting_platform":
        valid_platform = validate_lead_data(user_input, "Social Media Platform")
        # 🔥 FIX: Robust checking
        if "INVALID" in valid_platform.upper():
             return {"response": "I didn't recognize that platform. Please type something like YouTube, IG, TikTok, or Twitter."}
        
        # ALL DATA VALIDATED -> EXECUTE TOOL
        mock_lead_capture(state.get("name"), state.get("email"), valid_platform)
        
        return {"platform": valid_platform, "stage": None, "intent": None, "response": "You're all set! Our team will reach out to you shortly."}  
    else: 
        # Initial trigger
        return {"stage": "collecting_name", "response": "🚀 I see you're ready to get started! Let's get you set up. What is your name?"}
# 3. Define Routing Logic (The Edges)
def entry_router(state: AgentState):
    # If we are already in the middle of collecting lead data, skip intent detection!
    stage = state.get("stage")
    if stage in ["collecting_name", "collecting_email", "collecting_platform"]:
        return "lead_node"
    return "intent_node"

def intent_router(state: AgentState):
    # Route based on the intent detected by the LLM
    return state.get("intent", "query")

# 4. Build and Compile the Graph
builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("intent_node", intent_node)
builder.add_node("greeting_node", greeting_node)
builder.add_node("rag_node", rag_node)
builder.add_node("lead_node", lead_node)

# Add Edges
builder.add_conditional_edges(START, entry_router, {"lead_node": "lead_node", "intent_node": "intent_node"})
builder.add_conditional_edges("intent_node", intent_router, {"greeting": "greeting_node", "query": "rag_node", "high_intent": "lead_node"})

# All paths lead to the end of the turn
builder.add_edge("greeting_node", END)
builder.add_edge("rag_node", END)
builder.add_edge("lead_node", END)

# Compile with memory so it remembers state across turns
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)