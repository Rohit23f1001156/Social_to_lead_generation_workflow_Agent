# Social-to-Lead Agentic Workflow (Inflx / AutoStream)

This system is an intelligent conversational agent that processes social media interactions, qualifies leads, and captures information using LangGraph and RAG. By leveraging multi-turn conversations and stateful memory, it seamlessly guides users from general inquiries to actionable lead capture in a real-world workflow.

## Key Features

- **Intent Detection**: Dynamically categorizes user input into predefined intents (`greeting`, `query`, `high_intent`) to route the conversation effectively.
- **RAG-based Knowledge Retrieval**: Uses FAISS vector storage and custom embeddings to ground responses in specific domain knowledge, minimizing hallucination.
- **Stateful Multi-Turn Conversation**: Maintains context across interactions using threaded memory, ensuring continuity and coherence.
- **Controlled Tool Execution**: Executes lead capture functions safely when a user exhibits high intent, structuring data reliably.
- **Data Validation**: Sanitizes inputs (e.g., automatically resolving shorthand like "YT" to "YouTube") to maintain clean lead records.
- **Streamlit UI Support**: Provides an interactive front-end to visualize the agent's thought process and interactions.

## Tech Stack

- **Python**: Core programming language
- **LangGraph**: Framework for orchestrating the multi-agent state machine
- **LangChain**: Building blocks for LLM interactions
- **Gemini API**: Primary LLM for reasoning and generation
- **FAISS**: Vector database for semantic search and retrieval
- **Streamlit**: Web framework for building the conversational interface
- **Requests**: Used for a custom REST-based embeddings workaround to ensure stability

## How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd social_lead_agent
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

5. **Run the ingestion script**:
   Populate the FAISS vector database with your knowledge base documents:
   ```bash
   python rag/ingest.py
   ```

6. **Run the Streamlit application**:
   Start the interactive interface:
   ```bash
   streamlit run app.py
   ```

## System Architecture

The core of this agentic workflow is built on **LangGraph**, deliberately chosen over linear LangChain pipelines. A conversational AI designed for lead qualification cannot be a simple forward pass; it requires complex, non-deterministic decision-making. LangGraph enables conditional routing based on intent, allowing the system to pivot between casual conversation, specific knowledge retrieval, and structured tool execution.

State management is handled through a carefully designed multi-turn graph state, ensuring that the context of prior messages dictates future actions. By separating the logic into distinct nodes (intent detection, RAG generation, and lead capture), the architecture adheres to a clean separation of concerns. 

To mitigate hallucinations, the agent relies strictly on the RAG pipeline when answering domain-specific queries. The LLM does not generate answers from its parametric memory but synthesizes responses based entirely on context retrieved from FAISS. Furthermore, tool execution is tightly controlled; the lead capture mechanism only triggers when the intent detection node explicitly flags the conversation as `high_intent`, preventing premature or erroneous data collection.

## RAG Pipeline Explanation

The Retrieval-Augmented Generation (RAG) pipeline operates in distinct stages to ensure highly relevant context injection:
1. **Document Processing**: Raw JSON data is parsed, mapped to Document objects, and systematically chunked to respect LLM context window constraints.
2. **Embeddings Generation**: Text chunks are converted into high-dimensional vector representations capturing semantic meaning.
3. **FAISS Vector Storage**: These vectors are indexed in a local FAISS database for extremely fast nearest-neighbor lookups.
4. **Semantic Retrieval**: Unlike keyword search, the system queries the FAISS index using vector similarity, surfacing chunks that conceptually match the user's query even if exact phrasing differs.
5. **Context Injection**: The retrieved documents are formatted and injected directly into the LLM's system prompt, grounding its response in verified facts.

## Engineering Challenge

A critical issue encountered during development was a persistent `504 Deadline Exceeded` gRPC timeout error when calling the Google Generative AI embeddings service, specifically isolated to macOS environments running Python 3.13. Instead of forcing a downgrade to an older Python version, the architecture was refactored to bypass the problematic gRPC transport layer. A custom, REST-based embedding class was implemented using the standard Python `requests` library to directly interact with the Gemini API endpoints. This architectural pivot completely eliminated the timeout issues, resulting in a significantly more stable and robust ingestion pipeline without sacrificing dependency modernization.

## WhatsApp Integration Strategy

To scale this agent beyond a local Streamlit interface, the architecture is designed to support seamless integration with the Meta WhatsApp Business API:
- **Webhook Setup**: A lightweight backend (e.g., FastAPI or Flask) will expose webhook endpoints to receive incoming messages securely.
- **Message Ingestion**: Payloads from Meta are parsed to extract user text and metadata.
- **Stateful Memory**: The user's WhatsApp phone number inherently serves as the unique `thread_id`, perfectly aligning with LangGraph's checkpointer to maintain conversation history persistently.
- **Response Delivery**: The graph's output is routed back through the Meta API to deliver the message to the user.
- This design unlocks scalable, multi-user conversations, allowing the agent to handle parallel leads asynchronously.

## Demo Flow

A typical interaction with the demo application follows this trajectory:
1. **Greeting**: The user initiates contact. The agent identifies the `greeting` intent and responds politely.
2. **Pricing/Product Query**: The user asks a specific question. The agent detects a `query`, triggers the RAG pipeline, and provides an accurate answer based on the knowledge base.
3. **High Intent Detection**: The user expresses a desire to purchase or engage further. The intent node routes to the lead capture flow.
4. **Tool Execution**: The system executes the lead capture tool, prompting the user for necessary details and structuring the output cleanly.

## Future Improvements

- **Database Integration**: Transition lead storage from volatile memory/logs to a robust database (e.g., PostgreSQL, MongoDB).
- **Authentication**: Implement secure login for an admin dashboard to view captured leads.
- **Deployment Strategy**: Containerize the application using Docker and deploy to a cloud provider (AWS/GCP/Render) for high availability.
- **Multi-Agent Orchestration**: Expand the graph to include specialized sub-agents (e.g., a dedicated scheduling agent or a support agent).
- **Production WhatsApp Rollout**: Fully implement the Meta webhook integration for live customer testing.
