import vertexai
import streamlit as st
from vertexai.preview import generative_models
from vertexai.preview.generative_models import GenerativeModel, Part, Content, ChatSession

project = "sample-gemini"
vertexai.init(project=project)

config = generative_models.GenerationConfig(
    temperature=0.4
)

# Load model with config
model = GenerativeModel(
    model_name="gemini-pro",
    generation_config=config
)
chat = model.start_chat()

# Helper function to display and send Streamlit messages
def llm_function(chat: ChatSession, query):
    response = chat.send_message(query)
    output = response.candidates[0].content.parts[0].text

    with st.chat_message("model"):
        st.markdown(output)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )
        st.session_state.messages.append(
            {
                "role": "model",
                "content": output
            }
        )

st.title("Gemini Explorer")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display and load to chat history
for index, message in enumerate(st.session_state.messages):
    content = Content(
        role=message["role"],
        parts=[Part.from_text(message["content"])]
    )

    if index != 0:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    chat.history.append(content)

# For initial message startup
if len(st.session_state.messages) == 0:
    initial_message = "Welcome to Gemini Explorer! How can I assist you today?"

    st.session_state.messages.append(
        {
            "role": "bot", 
            "content": initial_message
            }
        )
    
    content = Content(role="bot", parts=[Part.from_text(initial_message)])
    chat.history.append(content)
    with st.spinner(text="Processing..."):
        llm_function(chat, initial_message)

# Capture user input
query = st.chat_input("Gemini Explorer")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    llm_function(chat, query)
