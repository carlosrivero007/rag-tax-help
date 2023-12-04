# Imports
import sqlite3
from datetime import datetime, timedelta
import uuid
import os
import streamlit as st
import pinecone
import tiktoken
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone


st.set_page_config(page_title="Enhanced Chatbot", layout="wide")
# Streamlit App Setup and Functions
def setup_app():
    """Set up the Streamlit app."""
    initialize_session_state()
   
    api_key = configure_sidebar()
    handle_chat_interface(api_key)

# Database file
DB_FILE = 'chat_history.db'
# User Inputs for API Keys and Pinecone Configuration
def get_user_input():
    st.sidebar.header("Configuration")
    PINECONE_API_KEY = st.sidebar.text_input("Enter your Pinecone API Key", type="password", key="pinecone_api_key")
    PINECONE_ENVIRONMENT = st.sidebar.text_input("Enter your Pinecone Environment", key="pinecone_environment")
    PINECONE_INDEX_NAME = st.sidebar.text_input("Enter your Pinecone Index Name", key="pinecone_index_name")
    OPENAI_API_KEY = st.sidebar.text_input("Enter your OpenAI API Key", type="password", key="openai_api_key")

    return PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, OPENAI_API_KEY


PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, OPENAI_API_KEY = get_user_input()

# Initialize Pinecone if API Key is provided
if PINECONE_API_KEY:
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index = pinecone.Index(PINECONE_INDEX_NAME)

# Initialize Langchain Embeddings if OpenAI API Key is provided
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Initialize Vectorstore
    text_field = "text"
    vectorstore = Pinecone(index, embeddings.embed_query, text_field)
# Database Functions
# ------------------

def create_connection():
    """Create a database connection to the SQLite database."""
    try:
        return sqlite3.connect(DB_FILE)
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

def create_table():
    """Create the chat_history table if it doesn't exist."""
    conn = create_connection()
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp DATETIME NOT NULL
                )
            """)
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")
    finally:
        if conn:
            conn.close()

def save_chat(session_id, message, response):
    """Save a chat message and response to the database."""
    conn = create_connection()
    timestamp = datetime.now()
    try:
        with conn:
            conn.execute("""
                INSERT INTO chat_history (session_id, message, response, timestamp)
                VALUES (?, ?, ?, ?)
            """, (session_id, message, response, timestamp))
    except sqlite3.Error as e:
        print(f"Error saving chat: {e}")
    finally:
        if conn:
            conn.close()

def get_chat_history(session_id):
    """Retrieve the chat history for a given session."""
    conn = create_connection()
    try:
        with conn:
            cur = conn.cursor()
            cur.execute("SELECT message, response FROM chat_history WHERE session_id = ?", (session_id,))
            return cur.fetchall()
    except sqlite3.Error as e:
        print(f"Error retrieving chat history: {e}")
    finally:
        if conn:
            conn.close()

def delete_old_records():
    """Delete records older than a specified time period."""
    conn = create_connection()
    try:
        with conn:
            conn.execute("DELETE FROM chat_history WHERE timestamp < ?", (datetime.now() - timedelta(days=1),))
    except sqlite3.Error as e:
        print(f"Error deleting old records: {e}")
    finally:
        if conn:
            conn.close()

# ---------------------------------
# Global variable for conversation history
conversation_history = []

def initialize_session_state():
    """Initialize the session state variables."""
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(uuid.uuid4())

def configure_sidebar():
    """Configure the sidebar of the Streamlit app."""
    with st.sidebar:
        st.image("https://your-sidebar-image-url.jpg", width=150)
        st.header("Chat History")
        history = get_chat_history(st.session_state['session_id'])
        for idx, (message, response) in enumerate(history):
            st.text_area(f"Session {idx}", f"Q: {message}\nA: {response}", height=100)
        api_key = st.text_input("Enter your OpenAI API Key", type="password")
        return api_key

def handle_chat_interface(api_key):
    """Handle the main chat interface."""
    with st.container():
        st.write("Enter your query below:")
        use_rag = st.checkbox("Use RAG for context-aware responses")
        user_input = st.text_input("Type your message here", key="chat_input")
        send_button = st.button("Send")

        if send_button and api_key:
            process_and_display_response(api_key, user_input, use_rag)

def process_and_display_response(api_key, user_input, use_rag):
    """Process the user input and display the response."""
    if use_rag:
        # Process RAG query
        rag_response = process_query(api_key, user_input, use_rag)
        st.write(rag_response)
    else:
        # Process normal query
        openai_response = get_openai_stream_response(api_key, user_input)
        if openai_response:
            save_chat(st.session_state['session_id'], user_input, openai_response)
            st.write(openai_response)
        else:
            st.error("Error in processing your message.")

def process_query(api_key, query, use_context=False, top_k=3):
    """Process a query using either normal or RAG mode."""
    if use_context:
        query = query.lstrip("RAG").strip()
        query_vector = embeddings.embed_query([query])[0]
        similar_items = vectorstore.search(query_vector, top_k=top_k)
        context = "\n".join([item['metadata']['text'] for item in similar_items])
        query = context + "\n" + "\n".join(conversation_history) + "\n" + query

    return get_openai_stream_response(api_key, query)

def get_openai_stream_response(api_key, message):
    """Get a response from OpenAI API using the streaming feature."""
    try:
        client = OpenAI(api_key=api_key)
        stream = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": message}],
            stream=True
        )
        response_content = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response_content += chunk.choices[0].delta.content
        return response_content
    except Exception as e:
        st.error(f"Stream response error: {e}")
        return None



# Main Script
# -----------

if __name__ == "__main__":
    create_table()
    setup_app()
