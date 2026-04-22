import streamlit as st
from agent.graph import graph

# 1. Page Configuration
st.set_page_config(
    page_title="AutoStream Agent",
    page_icon="🚀",
    layout="centered"
)

st.title("🚀 AutoStream Lead Agent")
st.caption("Powered by LangGraph & Gemini 2.5 Flash")

# 2. Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you with AutoStream today?"}
    ]

# 3. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # 🔥 FIX: Escape dollar signs to prevent green math rendering
        safe_content = message["content"].replace("$", r"\$")
        st.markdown(safe_content)

# 4. Handle User Input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Display user message instantly
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Save user message to state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 5. Process through LangGraph
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                # Using a fixed thread_id to maintain state memory across the session
                config = {"configurable": {"thread_id": "streamlit_user_1"}}
                result = graph.invoke({"user_input": user_input}, config=config)
                
                response_text = result["response"]
                safe_response = response_text.replace("$", r"\$")
                st.markdown(safe_response)
                
                # Save assistant response to state (we can save the normal text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})