import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Fine-Tuning Demo App")

if "data" not in st.session_state:
    st.session_state.data = []

st.header("Add Training Data")

question = st.text_input("Question")
answer = st.text_input("Answer")

if st.button("Add Pair"):
    if question and answer:
        st.session_state.data.append((question, answer))
        st.success("Training pair added!")

if st.session_state.data:
    df = pd.DataFrame(st.session_state.data, columns=["Question", "Answer"])
    st.dataframe(df)

st.header("Test Model")

prompt = st.text_input("Enter a prompt")

if st.button("Generate Response"):
    if len(st.session_state.data) == 0:
        st.warning("Add training data first.")
    else:
        questions = [q for q, a in st.session_state.data]
        answers = [a for q, a in st.session_state.data]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(questions)

        prompt_vec = vectorizer.transform([prompt])
        similarities = cosine_similarity(prompt_vec, X)

        best_match = similarities.argmax()
        st.write("Response:")
        st.success(answers[best_match])
