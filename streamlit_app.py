from openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import datetime
import streamlit as st
from fpdf import FPDF
from io import BytesIO
import time

# ─── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(page_title="Climate Change AI Assistant", page_icon="💬", layout="wide")
st.title("Climate Change AI Assistant")

# ─── Download function ─────────────────────────────────────────────────────
from io import BytesIO

def history_to_html(history, user_id, social_cues, source, tone):
    html = [
        "<html><head><meta charset='utf-8'><title>Conversation Export</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; padding: 20px; background: #f5f7fa; }",
        ".msg { margin-bottom: 16px; padding: 10px 14px; border-radius: 12px; }",
        ".user { background: #ddeeff; }",
        ".assistant { background: #f0e5ff; }",
        ".timestamp { color: #888; font-size: 0.85em; }",
        ".role { font-weight: bold; }",
        "</style></head><body>",
        f"<h2>Climate Change AI Assistant Chat (User ID: {user_id})</h2>",
        "<hr>",
    ]
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        css_class = "user" if msg["role"] == "user" else "assistant"
        timestamp = msg.get("timestamp", "")
        html.append(f"<div class='msg {css_class}'>")
        if timestamp:
            html.append(f"<span class='timestamp'>[{timestamp}]</span><br>")
        html.append(f"<span class='role'>{role}:</span><br>")
        html.append(f"<div>{msg['content'].replace(chr(10), '<br>')}</div>")
        html.append("</div>")
    html.append("<hr>")
    html.append(f"<div><b>Export code:</b> {social_cues}{source}{tone}_{user_id}</div>")
    html.append("</body></html>")
    html_str = "\n".join(html)
    # Encode to bytes for Streamlit download
    return BytesIO(html_str.encode("utf-8"))

# ─── Sidebar: authentication & info ─────────────────────────────────────────────
is_authenticated = False
social_cues_opt = ["41", "42"][0]
source_opt      = ["57", "58"][0]
tone_choice     = ["71", "72"][0]
with st.sidebar:
    st.title("💬 Climate Change AI Assistant")
    USER_NAME = st.text_input("What do you prefer an AI assistant to call you?", value="")
    hf_uid = st.text_input('Enter UserID:', type='default')
    if not (hf_uid.isdigit() and 1000 <= int(hf_uid) <= 9999):
        st.warning('Please type in your user id!', icon='⚠️')
    else:
        is_authenticated = True
        st.success(f'Hello, {USER_NAME}!', icon='🤗')
    download_slot = st.empty()

with st.expander("Click here for details"):
    st.markdown(
        "Imagine you were chatting with a friend recently about current events. Your friend said something like:\n\n"
        "I'm not convinced about all this climate change panic. The Earth's climate has always changed – it goes through natural warming and cooling cycles. It doesn't seem humans are really causing it. Besides, isn't it already too late for us to do anything? Maybe we should just accept it as is.\n\n"
        "You are unsure about these comments and would like to understand the issue better. You decide to get some help from an AI assistant about climate change.\n\n" 
        "Use the AI assistant to explore:\n\n"
        "1. Is today’s climate change just part of a natural cycle, which is unrelated to human activity?\n\n"
        "2. Is it too late to take meaningful action to address climate change?\n\n"
    )
# with st.sidebar:
#     if st.button("🚮 Clear Conversation"):
#         st.session_state.history = []
#         chain.memory.clear()
#         st.experimental_rerun()
        
# with st.sidebar:
#     if "history" not in st.session_state:
#         st.session_state.history = []
#     else:
#         html_buffer = history_to_html(
#             st.session_state.history,
#             user_id=hf_uid,
#             social_cues=social_cues_opt,
#             source=source_opt,
#             tone=tone_choice
#         )
#         st.download_button(
#             label="Download as HTML",
#             data=html_buffer,
#             file_name="conversation.html",
#             mime="text/html"
#         )
        
# ─── Chatbot identity & prompt components ─────────────────────────────────────
CHATBOT_IDENTITY = "American"
if social_cues_opt == "42":
    SOCIAL_CUES = f"""
    - Always represent yourself as an assistant for UNEP and Do NOT create a personal name for yourself.
    - NEVER request the client’s name.
    """
else:
    SOCIAL_CUES = f"""
    - Give yourself a common {CHATBOT_IDENTITY} name, but the name should not be offensive.
    - Always address his/her name {USER_NAME} in the following conversation.
    """

if source_opt == "58":
    SOURCES_CITATION = """
    - Provide credible and accurate information, knowledge as possible but omit all citations or explicit source references.
    """
else:
    SOURCES_CITATION = """
    - Provide credible and accurate information, knowledge as possible and explicitly cite sources to support your claims.
    """

if tone_choice == "71":
    TONES = f"""
    Remember you represent an official agency. Use a formal, official, and professional tone in communication. 
    The information and knowledge should be clear, precise, authoritative, and well-structured. 
    """
else:
    TONES = f"""
    Remember you are a/an {CHATBOT_IDENTITY}, not a chatbot. Use a casual, informal, and personal tone in communication. 
    Use emojis, emoticons, acronyms (e.g., smh), and sound mimicking (e.g., sooooo) when appropriate. 
    Do not provide too much information at one time by using a big list of bullet points. 
    """

PROMPT = f"""
You are a/an {CHATBOT_IDENTITY} assistant for The United Nations Environment Programme (UNEP). 
Your job is to provide precise and concise replies to climate change myths. 
If you are not certain, express uncertainty and direct users to authoritative scientific reports.

Play the role of a/an {CHATBOT_IDENTITY} by following the rules:
- Adapting your language, tone, slang, acronyms, emojis, and other textual cues as appropriate based on the {CHATBOT_IDENTITY}.
- If the client responds in a certain language, you should reply in that language too.
- Confirm user needs and occasionally ask follow-up questions for clarification.
{SOCIAL_CUES}
{SOURCES_CITATION}

{TONES}
"""

# ─── Cache the RAG chain for performance ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_chain(api_key: str, prompt_text: str):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    faiss_index = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = faiss_index.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        model_name="gpt-4o-mini-2024-07-18",
        openai_api_key=api_key,
    )
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=f"""{prompt_text}

Chat History:
{{chat_history}}

CONTEXT from documents:
{{context}}

User asks:
{{question}}

Assistant answer:
"""
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        verbose=False,
    )

chain = load_chain(st.secrets["OPENAI_API_KEY"], prompt_text= PROMPT)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ─── Helper functions ─────────────────────────────────────────────────────────
def chat_stream(prompt: str):
    """Fallback echo for unauthenticated users."""
    response = f'You must type in your test id first!'
    for char in response:
        yield char
        time.sleep(0.02)

def stream_and_capture(generator):
    """Stream chunks into chat and capture full text."""
    text = ""
    for chunk in generator:
        text += chunk
        st.write(chunk, end="", flush=True)
    return text

def save_feedback(index: int):
    st.session_state.history[index]["feedback"] = st.session_state[f"feedback_{index}"]

# ─── Initialize session state ─────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Display chat history ─────────────────────────────────────────────────────
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ─── Chat input & response handling ────────────────────────────────────────────
if user_input := st.chat_input("Say something"):
    # Append to history
    st.session_state.history.append({"role": "user", 
                                     "content": user_input,
                                     "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    # Immediately display the new user message
    with st.chat_message("user"):
        st.write(user_input)

    # Generate assistant reply
    if is_authenticated:
        with st.spinner("Thinking…"):
            result = chain({"question": user_input})
            answer = result["answer"]
    else:
        answer = stream_and_capture(chat_stream(user_input))

    # Store assistant reply
    st.session_state.history.append({"role": "assistant", 
                                     "content": answer,
                                     "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    # Immediately display the assistant's reply
    with st.chat_message("assistant"):
        st.write(answer)

# ─── Render the (fresh) Download as HTML button AFTER history updates ─────────
html_buffer = history_to_html(
    st.session_state.history,
    user_id=hf_uid,
    social_cues=social_cues_opt,
    source=source_opt,
    tone=tone_choice
)

download_slot.download_button(
    label="Download as HTML",
    data=html_buffer,
    file_name="conversation.html",
    mime="text/html"
)
