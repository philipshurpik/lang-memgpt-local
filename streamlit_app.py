import asyncio
import uuid
import streamlit as st
from lang_memgpt_local.chat import Chat

# Define available users
USERS = {
    "cat": {"icon": "🐱", "id": "cat-user-001"},
    "dog": {"icon": "🐶", "id": "dog-user-002"},
    "lion": {"icon": "🦁", "id": "lion-user-003"},
    "fox": {"icon": "🦊", "id": "fox-user-004"},
    "horse": {"icon": "🐴", "id": "horse-user-005"}
}

# Initialize session state for all chat histories if they don't exist
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {user: [] for user in USERS.keys()}

# Initialize session state for thread IDs if they don't exist
if "thread_ids" not in st.session_state:
    st.session_state.thread_ids = {user: str(uuid.uuid4()) for user in USERS.keys()}

# Initialize chat instances if they don't exist
if "chats" not in st.session_state:
    st.session_state.chats = {
        user: Chat(USERS[user]["id"], st.session_state.thread_ids[user]) 
        for user in USERS.keys()
    }

st.title("MemGPT Chat Demo")

# User selection in sidebar with icons
selected_user = st.sidebar.selectbox(
    "Select User",
    options=list(USERS.keys()),
    format_func=lambda x: f"{USERS[x]['icon']} {x.capitalize()}",
)

# Button to start new conversation for current user
if st.sidebar.button("New Conversation"):
    st.session_state.thread_ids[selected_user] = str(uuid.uuid4())
    st.session_state.chats[selected_user] = Chat(
        USERS[selected_user]["id"], 
        st.session_state.thread_ids[selected_user]
    )
    st.session_state.chat_histories[selected_user] = []
    st.rerun()

# Display current user's chat history
for message in st.session_state.chat_histories[selected_user]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input(f"Message {USERS[selected_user]['icon']} {selected_user.capitalize()}..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to current user's chat history
    st.session_state.chat_histories[selected_user].append(
        {"role": "user", "content": prompt}
    )

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # Since we're using asyncio, we need to run it in a coroutine
        response = asyncio.run(st.session_state.chats[selected_user](prompt))
        message_placeholder.markdown(response)
    
    # Add assistant response to current user's chat history
    st.session_state.chat_histories[selected_user].append(
        {"role": "assistant", "content": response}
    )
