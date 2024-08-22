import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os

# Access the API key in codespaced
#from dotenv import load_dotenv
# Load environment variables from .env file
#load_dotenv()
#api_key = os.getenv("OPENAI_API_KEY")

api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Docerative AI", layout="wide")

# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle cases where extract_text returns None
    return text

# Function to chunk the extracted text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to create the conversational chain with memory
def get_conversational_chain(vectorstore, memory):
    model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return qa_chain

def main():
    st.header("AI clone chatbot💁")

    # Initialize memory and vectorstore in session state
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Initialize chat messages in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show intro markdown only before processing
    if "intro_shown" not in st.session_state:
        st.session_state.intro_shown = False

    if not st.session_state.intro_shown:
        st.markdown("""
        ## PDF Q&A System: Get instant insights from your Documents

        This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Generative AI model GPT 3.5 turbo. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

        ### How It Works

        Follow these simple steps to interact with the chatbot:

        1. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

        2. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
        """)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True, key="pdf_uploader")

        if st.button("Submit & Process", key="process_button"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vectorstore = get_vector_store(text_chunks)
                    st.success("Done")
                    # Once processing is done, hide the intro
                    st.session_state.intro_shown = True
            else:
                st.error("Please upload at least one PDF file.")

    # Display the conversation in the chat format
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])

    # Capture user input
    user_input = st.chat_input("Ask a Question from the PDF Files", key="user_question_input")
    # Process user input when it is submitted
    if user_input and st.session_state.vectorstore:
        # Get the conversational chain
        chain = get_conversational_chain(st.session_state.vectorstore, st.session_state.memory)
        
        # Generate the bot's response
        response = chain({"question": user_input})
        bot_response = response["answer"]
        
        if bot_response:  # Check if there's a valid response
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": bot_response})

        # Display the new messages in the chat format
        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write(bot_response)

        # Rerun the app with a clean input field
        st.query_params.clear()

if __name__ == "__main__":
    main()
