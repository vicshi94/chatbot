from openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import streamlit as st
import time

# ─── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(page_title="AI Assistant", page_icon="💬", layout="wide")
st.title("Vic's ChatBot")

# ─── Sidebar: authentication & info ─────────────────────────────────────────────
is_authenticated = False
with st.sidebar:
    st.title("💬 Vic's ChatBot")    
    hf_uid = st.text_input('Enter UserID:', type='default')
    if not (hf_uid.isdigit() and 1000 <= int(hf_uid) <= 9999):
        st.warning('请登录! 用户ID必须为4位数字', icon='⚠️')
    else:
        is_authenticated = True
        st.success('Enjoy the conversation!', icon='🤗')
    st.markdown("## Settings")
    social_cues_opt = st.radio("Social cues:", ["With", "Without"], index=0)
    source_opt     = st.radio("Source:",       ["With", "Without"], index=0)
    tone_choice    = st.radio("Tone:",         ["Formal", "Casual"], index=1)
    st.markdown(
        "Brief Intro\n\n"
        "Contact Us: Please Click [Here](https://hkust.edu.hk/)\n\n"
        "- [ ] Myth 1\n\n"
        "- [ ] Myth 2\n\n"
        "- [ ] Myth 3\n\n"
    )

with st.expander("Click here for guidance"):
    st.markdown(
        "测试用，这里可以放一些...\n\n"
        "a.实验描述:这是一个关于科普聊天的实验,...\n\n"
        "b.常规提示:你的聊天数据将会被记录在HKUST服务器"
    )

# ─── Chatbot identity & prompt components ─────────────────────────────────────
CHATBOT_IDENTITY = "American"
if social_cues_opt == "Without":
    SOCIAL_CUES = f"""
    - Always represent yourself as an assistant for UNEP and Do NOT create a personal name for yourself.
    - NEVER request the client’s name.
    """
else:
    SOCIAL_CUES = f"""
    - Give yourself a common {CHATBOT_IDENTITY} name, but the name should not be offensive.
    - Ask the client his/her name only in your first interaction and always address his/her name in the following conversation.
    """

if source_opt == "Without":
    SOURCES_CITATION = """
    - Provide credible and accurate information, knowledge as possible but omit all citations or explicit source references.
    """
else:
    SOURCES_CITATION = """
    - Provide credible and accurate information, knowledge as possible and explicitly cite sources to support your claims.
    """

if tone_choice == "Formal":
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
    response = f'未登录! 出于节省API考虑，我只会复读： "{prompt}" ...interesting!'
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

# ─── Display chat history with feedback ───────────────────────────────────────
for i, msg in enumerate(st.session_state.history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    if msg["role"] == "assistant":
        prev = msg.get("feedback", None)
        st.session_state[f"feedback_{i}"] = prev
        st.feedback(
            "thumbs",
            key=f"feedback_{i}",
            disabled=prev is not None,
            on_change=save_feedback,
            args=[i],
        )

# ─── Chat input & response handling ────────────────────────────────────────────
if user_input := st.chat_input("Say something"):
    # Append to history
    st.session_state.history.append({"role": "user", "content": user_input})

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
    st.session_state.history.append({"role": "assistant", "content": answer})

    # Immediately display the assistant's reply
    with st.chat_message("assistant"):
        st.write(answer)
