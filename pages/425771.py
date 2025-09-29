import streamlit as st
from streamlit_app import render_app

st.set_page_config(page_title="425771", page_icon="💬", layout="wide")
render_app("42", "57", "71")
