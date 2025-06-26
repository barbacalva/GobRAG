#!/usr/bin/env python3

import json
import requests
import streamlit as st


API_URL = "http://localhost:8000/ask"
HEADERS = {"Content-Type": "application/json",
           "x-api-key": "demo123"}
TIMEOUT = 60  # in seconds


def stream_answer(question: str, top_k: int = 4):
    payload = json.dumps({"question": question, "top_k": top_k})
    try:
        response = requests.post(API_URL,
                                 data=payload,
                                 headers=HEADERS,
                                 timeout=TIMEOUT,
                                 stream=True)
    except requests.RequestException as e:
        raise RuntimeError(f"Error sending request: {e}")

    if response.status_code != 200:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        msg = detail or response.text or response.reason
        raise RuntimeError(f"Error {response.status_code}: {msg}")

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            break
        yield data


st.set_page_config(page_title="GobRAG", page_icon="⚖️")
st.title("⚖️  GobRAG – Chat jurídico BOE")

if "messages" not in st.session_state:
    st.session_state.messages = []


for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)


if question := st.chat_input("Haz tu consulta…"):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append(("user", question))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        partial = ""

        try:
            for token in stream_answer(question):
                partial += token
                placeholder.markdown(partial + "▌")
            placeholder.markdown(partial)
        except RuntimeError as err:
            placeholder.markdown(f"⚠️ **Error:** {err}")
            partial = f"⚠️ Error: {err}"

    st.session_state.messages.append(("assistant", partial))
