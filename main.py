import sqlite3
from datetime import datetime, timedelta
import uuid
from openai import OpenAI
import streamlit as st

# Database file
DB_FILE = "chat_history.db"

# Database Functions
# ------------------

def create_connection():
    """Create a database connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table():
    """Create the table for storing chat histories."""
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
        print(e)
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
        print(e)
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
        print(e)
    finally:
        if conn:
            conn.close()

def delete_old_records():
    """Delete records older than 24 hours."""
    conn = create_connection()
    try:
        with conn:
            conn.execute("DELETE FROM chat_history WHERE timestamp < ?", (datetime.now() - timedelta(days=1),))
    except sqlite3.Error as e:
        print(e)
    finally:
        if conn:
            conn.close()

# Initialize the database table
create_table()

# Streamlit and OpenAI API Interaction
# ------------------------------------

def get_openai_stream_response(api_key, message):
    print("Entering get_openai_stream_response")
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
                response_part = chunk.choices[0].delta.content
                print(response_part, end="")
                response_content += response_part
        return response_content
    except Exception as e:
        error_message = f"Stream response error: {e}"
        st.error(error_message)
        print(error_message)
        return None


def setup_app():
    print("Setting up the app")
    st.set_page_config(page_title="Tax Help Now!", layout="wide")

    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(uuid.uuid4())
        print(f"New session ID: {st.session_state['session_id']}")

    with st.sidebar:
        st.image("https://s30383.pcdn.co/wp-content/uploads/2023/02/MOS-using-chatbot-OC23DD.jpg", width=150)
        st.header("Chat History")
        history = get_chat_history(st.session_state['session_id'])
        print(f"Chat history: {history}")
        for idx, (message, response) in enumerate(history):
            st.text_area(f"Session {idx}", f"Q: {message}\nA: {response}", height=100)
        api_key = st.text_input("Enter your OpenAI API Key", type="password")

    with st.container():
        chat_history = st.empty()
        chat_input = st.text_input("Type your message here", key="chat_input")
        send_button = st.button("Send")

    if send_button and api_key:
        print(f"Send button clicked with API key: {api_key}")
        user_message = chat_input
        openai_response = get_openai_stream_response(api_key, user_message)
        if openai_response:
             print("Received response from OpenAI, updating database")
             save_chat(st.session_state['session_id'], user_message, openai_response)
             updated_history = get_chat_history(st.session_state['session_id'])
             chat_history.markdown("\n".join(f"**You:** {m}\n**Bot:** {r}" for m, r in updated_history))
        else:
            st.error("Error in processing your message.")


setup_app()
