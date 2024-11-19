## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.schema import HumanMessage, AIMessage

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


hf_api_key=st.secrets['HF_Token']
groq_api_key=st.secrets['GROQ_API_KEY']
st.set_page_config(page_title="RESEARCH PAPER CHATBOT", page_icon="📰")
st.title("📰 You Got This")
with st.sidebar:
    # if st.button('Click to go to OpenAI'):
    # Redirect the user to the URL when the button is clicked
    st.markdown("[Go Back](http)", unsafe_allow_html=True)
    st.markdown("**Guide to Use ** .")
    st.write("""
1.Enter your Groq API Key in the "Enter your Groq API key:" input box.\n
2.Enter a Session ID for multiple conversations.\n
3.Upload PDF files using the "Choose A PDF file" button.\n
4.Wait for the app to process the uploaded PDF(s).\n
5.Enter your question in the "Your question:" input box.\n
6.View the Assistant's response based on the PDF content.\n
7.Click the Chat History expander to view the conversation history.\n
.""")
   

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",api_key=hf_api_key)


## set up Streamlit 
st.write("Upload Pdf's and chat with their content")



llm=ChatGroq(groq_api_key=groq_api_key,model_name="Gemma2-9b-It")

    ## chat interface

session_id=st.text_input("Enter Session ID (for multiple conversations):",value="default_session")
    ## statefully manage chat history

if 'store' not in st.session_state:
        st.session_state.store={}

uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
    ## Process uploaded  PDF's
if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
         with st.spinner(f"Processing {uploaded_file.name}..."):

            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name


            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

    # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()    
        st.success(f"Processed {uploaded_file.name} successfully!")

        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
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
        
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        ## Answer question

        # Answer question
        system_prompt = (
            '''You are an assistant designed to answer questions based on research papers. 
            When a question is asked, use the relevant sections of the uploaded research to provide a thorough, detailed answer.
            Your response should explain the context in sufficient but more also in concised way
              detail to make the information clear and understandable. 
            If the answer isn't available in the provided context,please dont answer by yourself indicate that you don’t have enough information to provide an answer
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
        
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  # constructs a key "abc123" in `store`.
            )
            # st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            if 'store' in st.session_state:
                # Retrieve the session history
                session_history = st.session_state.store.get(session_id, [])
                
                
                with st.expander("### Chat History:"):
                
                    # Iterate through the session history to extract message content
                    for message_tuple in session_history:
                        # Check if message_tuple is a tuple and its second element is a list
                        if isinstance(message_tuple, tuple) and isinstance(message_tuple[1], list):
                            # Iterate over the list of messages and extract content
                            for msg in message_tuple[1]:
                                # Check if the message is a HumanMessage or AIMessage
                                if isinstance(msg, HumanMessage):
                                    st.write(f"**User**: {msg.content}")
                                elif isinstance(msg, AIMessage):
                                    st.write(f"**AI**: {msg.content}")
                        else:
                            # In case it's not a tuple, handle accordingly (if needed)
                            if isinstance(message_tuple, HumanMessage):
                                st.write(f"**User**: {message_tuple.content}")
                            elif isinstance(message_tuple, AIMessage):
                                st.write(f"**AI**: {message_tuple.content}")
else:
    st.warning("Please enter the GRoq API Key")










