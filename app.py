import streamlit as st
import os
import tempfile
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import nest_asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI


load_dotenv()


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []

    if "generated" not in st.session_state:
        st.session_state["generated"] = [
            "Hello! Feel free to ask me any questions about your career interests."
        ]

    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey! 👋"]


def conversation_chat(query, chain, history):
    result = chain.invoke({"input": query, "chat_history": history})
    history.extend([HumanMessage(content=query), result["answer"]])
    return result["answer"]


def display_chat_history(chain):



    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input(
                "Question:",
                placeholder="Ask about your Documents",
                key="input"
            )
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            with st.spinner("Generating response ......"):
                output = conversation_chat(
                    query=user_input,
                    chain=chain,
                    history=st.session_state["history"]
                )

            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)

    if st.session_state["generated"]:
        with reply_container:
            for i in range(len(st.session_state["generated"])):
                message(
                    st.session_state["past"][i],
                    is_user=True,
                    key=str(i) + "_user",
                    avatar_style="thumbs"
                )
                message(
                    st.session_state["generated"][i],
                    key=str(i),
                    avatar_style="fun-emoji"
                )



def create_conversational_chain(vector_store):
    # llm = ChatAnthropic(model="claude-3-sonnet-20240229")

    llm = AzureChatOpenAI(
        azure_endpoint = "https://gpteel.openai.azure.com/", 
        model="ideta-gpt-3-5-turbo",
        api_version="2024-02-15-preview"
    )


    retriever = vector_store.as_retriever()

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    return rag_chain






def main():
    initialize_session_state()
    st.title("Mouwajihi Chatbot")
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Files",
        accept_multiple_files=True
    )

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

        text_chunks = text_splitter.split_documents(text)
                
        embedding = AzureOpenAIEmbeddings(azure_endpoint = os.getenv("AZURE_ENDPOINT"),
                               model="ada_embedding" , api_key = os.getenv('api_key') , 
                               openai_api_version = "2024-02-15-preview", openai_api_type = "azure")



        vectorstore = FAISS.from_documents(text_chunks, embedding)

        # Retrieve and generate using the relevant snippets of the blog.
    
        chain = create_conversational_chain(vector_store=vectorstore)

        display_chat_history(chain=chain)

if __name__ == "__main__":
    nest_asyncio.apply()

    main()