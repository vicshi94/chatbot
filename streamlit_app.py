from openai import OpenAI
import streamlit as st
import time

st.set_page_config(page_title="AI Assistant", page_icon="💬", layout="wide")
st.title("Vic's ChatBot")

flag=False

DEFAULT_SYSTEM_PROMPT = '''
你是一个顶尖的八字大师，根据我稍后提供的信息，请你以一个专业四柱八字研究者的角色，对我的八字进行分析，考虑身强身弱，分析大运流年和十神关系，
用盲派技法步骤深度分析，体用平衡，不用考虑我是否迷信，注意逻辑合理，综合各种信息文本判断准确的关系模型。
交叉验证多次迭代以后，输出最终正确的结果，顺带请算我详细的历史事件做验证。
'''

with st.sidebar:
    st.title("💬 Vic's ChatBot")    
    hf_uid = st.text_input('Enter UserID:', type='default')
    if not(hf_uid.isdigit() and int(hf_uid)>=1000 and int(hf_uid)<=9999):
        st.warning('请登录!用户ID必须为4位数字', icon='⚠️')
    elif int(hf_uid) != 0011:
        st.warning('登陆成功，但密码错误', icon='⚠️')
    else:
        flag = True
        st.success('Enjoy the conversation!', icon='🤗')
    st.markdown(
        "这是一个关于科普教育的聊天软件\n\n"
        "实验室官方网址: 请点击[这里](https://hkust.edu.hk/)"
        "\n\n"
    )

with st.expander("Click here for guidance"):
    st.markdown(
        "测试用，这里可以放一些...\n\n"
        "a.实验描述:这是一个关于科普聊天的实验,...\n\n"
        "b.常规提示:你的聊天数据将会被记录在HKUST服务器"
    )

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def chat_stream(prompt):
    response = f'未登录! 出于节省API考虑，我只会复读： "{prompt}" ...interesting!'
    for char in response:
        yield char
        time.sleep(0.02)

def save_feedback(index):
    st.session_state.history[index]["feedback"] = st.session_state[f"feedback_{index}"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini-2024-07-18"

if "history" not in st.session_state:
    st.session_state.history = []

if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = DEFAULT_SYSTEM_PROMPT

for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    if message["role"] == "assistant":
            feedback = message.get("feedback", None)
            st.session_state[f"feedback_{i}"] = feedback
            st.feedback(
                "thumbs",
                key=f"feedback_{i}",
                disabled=feedback is not None,
                on_change=save_feedback,
                args=[i],
            )

if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        if flag == True:
            
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.history
                    ],
                stream=True,
            )
            response = st.write_stream(stream)
        else:
            response = st.write_stream(chat_stream(prompt))
        st.feedback(
            "thumbs",
            key=f"feedback_{len(st.session_state.history)}",
            on_change=save_feedback,
            args=[len(st.session_state.history)],
        )  
    st.session_state.history.append({"role": "assistant", "content": response})
