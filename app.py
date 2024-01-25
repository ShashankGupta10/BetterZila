import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.cohere import Cohere
import PyPDF2

def get_pdf_text():
    text = ""
    with open("48lawsofpower.pdf", 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    print(text)
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = CohereEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def main():
    
    st.title("Chat with PDF :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if st.button("Process"):
        if user_question:
            with st.spinner("Processing"):
                raw_text = get_pdf_text()
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                llm = Cohere(cohere_api_key=st.secrets["COHERE_API_KEY"])
                conversation_chain = RetrievalQA.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(),
                )
                prompt = f"You will be asked a question by the user.\n \
                    The answer should be 3 - 4 sentences according to the context provided.\n \
                    The answer should be in your own words.\n \
                    The answer should be easy to understand.\n \
                    The answer should be grammatically correct.\n \
                    The answer should be relevant to the question.\n \
                    The question is {user_question}."
                answer = conversation_chain.invoke({"query": prompt})
                print(answer)
                st.info(answer["result"])


if __name__ == '__main__':
    main()
