import streamlit as st

from retrieve_context import retrieve_context
from prompting import generate_answer

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 AI RAG Chatbot")

st.write("Ask any question about the uploaded documents.")

question = st.text_input("Enter your question:")

if st.button("Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching..."):

            context, sources = retrieve_context(question)

            answer = generate_answer(
                question,
                context,
                sources
            )

        st.subheader("Answer")

        st.write(answer)