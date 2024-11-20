import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import json
from dotenv import load_dotenv

# Load API keys from environment variables
load_dotenv()
hf_api_key=st.secrets['HF_Token']
groq_api_key=st.secrets['GROQ_API_KEY']

# Set up Streamlit page
st.set_page_config(page_title="RESEARCH PAPER CHATBOT", page_icon="📰")
st.title("📰 You Got This")

# Guide
with st.sidebar:
    st.markdown("[Go Back](http)", unsafe_allow_html=True)
    st.markdown("**Guide to Use ** .")
    st.write("""
    1. Feel Relaxed and don’t stress more\n
    2. Enter a Session ID for multiple conversations.\n
    3. Upload PDF files using the "Choose A PDF file" button.\n
    4. Wait for the app to process the uploaded PDF(s).\n
    5. Enter your question in the "Your question:" input box.\n
    6. View the Assistant's response based on the PDF content.\n
    7. Click the Chat History expander to view the conversation history.\n
    """)

os.environ['HF_Token'] = os.getenv("HF_Token")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Suggest related research papers
def suggest_research_papers(query: str):
    api_wrapper = ArxivAPIWrapper(top_k_results=5)
    arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper)
    papers = arxiv_tool.run(query)
    return papers

# Session ID input
session_id = st.text_input("Enter Session ID (for multiple conversations):", value="default_session")
llm=ChatGroq(groq_api_key=groq_api_key,model_name="Gemma2-9b-It")


# Check if session history exists in session state
if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader("Choose A PDF file", type="pdf", accept_multiple_files=True)

# Process uploaded PDF files
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

    # Split and create embeddings for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()    
    st.success(f"Processed {uploaded_file.name} successfully!")

    # Create retrieval chain
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
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

    # System prompt for answering questions
    system_prompt = (
        '''You are an assistant designed to answer questions based on research papers. 
        When a question is asked, use the relevant sections of the uploaded research to provide a thorough, detailed answer.
        Your response should explain the context in sufficient but more also in concise way
        detail to make the information clear and understandable. 
        If the answer isn't available in the provided context, please don't answer by yourself; indicate that you don’t have enough information to provide an answer.
        "\n\n
        {context}"
        '''
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Your question:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": session_id}
            },
        )
        st.write("Assistant:", response['answer'])
        papers = suggest_research_papers(uploaded_file.name)

        if papers:
            st.write("Here are some related research papers:")
            with st.expander("Click Here for Recomandations:"):
              st.write(papers)

        # Retrieve and display chat history
        if 'store' in st.session_state:
            session_history = st.session_state.store.get(session_id, [])
            with st.expander("### Chat History:"):
                for message_tuple in session_history:
                    if isinstance(message_tuple, tuple) and isinstance(message_tuple[1], list):
                        for msg in message_tuple[1]:
                            if isinstance(msg, HumanMessage):
                                st.write(f"**User 👱🏼**: {msg.content}")
                            elif isinstance(msg, AIMessage):
                                st.write(f"**AI 🤖**: {msg.content}")
                    else:
                        if isinstance(message_tuple, HumanMessage):
                            st.write(f"**User 👱🏼**: {message_tuple.content}")
                        elif isinstance(message_tuple, AIMessage):
                            st.write(f"**AI 🤖**: {message_tuple.content}")

        # Chat history download functionality
        def format_chat_history(session_id):
            """Format the chat history as a string or JSON for download."""
            if session_id in st.session_state.store:
                session_history = st.session_state.store[session_id]
                chat_history_list = []

                for message_tuple in session_history:
                    if isinstance(message_tuple, tuple) and isinstance(message_tuple[1], list):
                        for msg in message_tuple[1]:
                            if isinstance(msg, HumanMessage):
                                chat_history_list.append(f"**User 👱🏼**: {msg.content}")
                            elif isinstance(msg, AIMessage):
                                chat_history_list.append(f"**AI 🤖**: {msg.content}")
                    else:
                        if isinstance(message_tuple, HumanMessage):
                            chat_history_list.append(f"**User 👱🏼**: {message_tuple.content}")
                        elif isinstance(message_tuple, AIMessage):
                            chat_history_list.append(f"**AI 🤖**: {message_tuple.content}")

                chat_history_string = "\n".join(chat_history_list)
                chat_history_json = json.dumps(chat_history_list, indent=4)

                return chat_history_string, chat_history_json
            else:
                return "No chat history available.", ""

        # Add a download button to download chat history as a text file or JSON
        chat_history_string, chat_history_json = format_chat_history(session_id)

        if chat_history_string != "No chat history available.":
            st.download_button(
                label="Download Chat History (Text)",
                data=chat_history_string,
                file_name=f"chat_history_{session_id}.txt",
                mime="text/plain"
            )

            # JSON download button
            st.download_button(
                label="Download Chat History (JSON)",
                data=chat_history_json,
                file_name=f"chat_history_{session_id}.json",
                mime="application/json"
            )
        else:
            st.write("No chat history to download.")
