import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv

from lang_memgpt_local.chat import Chat

load_dotenv()

# Define available users
USERS = {
    "cat": {"icon": "🐱", "id": "cat-user-001"},
    "dog": {"icon": "🐶", "id": "dog-user-002"},
    "lion": {"icon": "🦁", "id": "lion-user-003"},
    "fox": {"icon": "🦊", "id": "fox-user-004"},
    "horse": {"icon": "🐴", "id": "horse-user-005"}
}

# Initialize session state for all chat histories if they don't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Track the previously selected user
if "previous_user" not in st.session_state:
    st.session_state.previous_user = None

# Initialize session state for thread IDs if they don't exist
if "thread_ids" not in st.session_state:
    st.session_state.thread_ids = {user: str(uuid.uuid4()) for user in USERS.keys()}

# Initialize chat instances if they don't exist
if "chats" not in st.session_state:
    st.session_state.chats = {user: Chat(USERS[user]["id"], st.session_state.thread_ids[user]) for user in USERS.keys()}

st.title("MemGPT Chat Demo")

# User selection in sidebar with icons
selected_user = st.sidebar.selectbox(
    "Select User",
    options=list(USERS.keys()),
    format_func=lambda x: f"{USERS[x]['icon']} {x.capitalize()}",
)

if st.session_state.previous_user != selected_user:
    st.session_state.thread_ids[selected_user] = str(uuid.uuid4())
    st.session_state.chats[selected_user] = Chat(USERS[selected_user]["id"], st.session_state.thread_ids[selected_user])
    st.session_state.chat_history = []
    st.session_state.previous_user = selected_user
    st.rerun()

# Display current user's chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input(f"Message {USERS[selected_user]['icon']} {selected_user.capitalize()}..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to current user's chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Function to handle streaming response
    async def generate_response():
        response_text = ""
        # Display assistant message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            # Stream the response
            async for token in st.session_state.chats[selected_user].stream_response(prompt):
                response_text += token
                message_placeholder.markdown(response_text + "▌")
                await asyncio.sleep(0.01)  # Small sleep to allow UI to update
            message_placeholder.markdown(response_text)

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

    # Run the async function to get and display the streaming response
    asyncio.run(generate_response())
