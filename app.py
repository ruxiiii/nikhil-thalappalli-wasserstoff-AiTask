import streamlit as st
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from htmlTemplates import css, bot_template, user_template



load_dotenv()

def get_blog_content(url):
    stop_processing = False
    content = []
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html5lib')
    paragraphs = soup.find_all('p')

    for p in paragraphs:
        if stop_processing:
            break
        p_text = (p.get_text())
        content.append(p_text)
        if "Email Newsletter" in p.text:
            stop_processing = True
            continue
        links = p.find_all('a')
        for link in links:
            link_text=(link.get_text())
            content.append(link_text)

    content = ''.join(content)
    return content




def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        # separator="\n",
        chunk_size=150,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_relevant_excerpts(user_question, vectorstore):

  relevent_docs = vectorstore.similarity_search(user_question)
  return relevent_docs


def groq_chat(client, prompt,  model, response_format):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format=response_format
    )
    return completion.choices[0].message.content


def main():
    st.set_page_config(page_title="Chat with your Worpress Blogs",
                       page_icon=":Wassertoff ChatBot:")
    st.write(css, unsafe_allow_html=True)
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Your WordPress QA Bot")
    user_question = st.text_input("Ask a question about the blog:")
    st.write(user_template.replace("{{MSG}}", 'Hello Bot'), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", 'Hello Human'), unsafe_allow_html=True)


    if user_question and st.session_state.vectorstore:
        relevant_excerpts = get_relevant_excerpts(user_question, st.session_state.vectorstore)
        excerpt_final = ""
        for excerpt in relevant_excerpts:
            excerpt_final += excerpt.page_content


        groq_api_key = os.getenv('GROQ_API_KEY')
        model = "llama3-70b-8192"

        

        with open('base_prompt.txt', 'r') as file:
            base_prompt = file.read()

        client = Groq(
            api_key=groq_api_key
        )

        full_prompt =  base_prompt.format(user_question=user_question, provided_excerpts=excerpt_final)

        llm_response =  groq_chat(client, full_prompt, model, None )
        # st.write(llm_response)
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": llm_response})

        for i, message in enumerate(st.session_state.chat_history):
            # st.write(message)
            # st.write(i,message)
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message['content']), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message['content']), unsafe_allow_html=True)
        

    with st.sidebar:
        st.subheader("Blog URL")
        blog_url = st.text_input("Enter the WordPress blog URL here:")
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_content = get_blog_content(blog_url)

                text_chunks = get_text_chunks(raw_content)
                st.write("Text Chunks:", text_chunks)

                st.session_state.vectorstore = get_vectorstore(text_chunks)





if __name__ == "__main__":
    main()