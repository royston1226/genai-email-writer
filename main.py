import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from utils import clean_text


def streamlit_app(llm, clean_text):
    st.title("Cold Email Generator")
    url_imput = st.text_input("Enter a URL:", value="https://careers.nike.com/software-engineer/job/R-61123")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader([url_imput])
            data = clean_text(loader.load().pop().page_content)
            jobs = llm.extract_jobs(data)

            for job in jobs:
                skills = job.get("skills", [])
                email = llm.write_mail(job)
                st.code(email, language="markdown")
        except Exception as e:
            st.error(e)

if __name__ == "__main__":
    chain = Chain()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    streamlit_app(chain, clean_text)






