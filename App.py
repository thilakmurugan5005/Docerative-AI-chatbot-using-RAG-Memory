import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
import os
from llama_index.core import Document
from llama_index.core import get_response_synthesizer
from llama_index.core import DocumentSummaryIndex
from llama_index.llms.openai.base import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext

from llama_index.core.memory import ChatMemoryBuffer

api_key = st.secrets["OPENAI_API_KEY"]




st.set_page_config(page_title="Docurative AI", layout="wide")

def get_doc_title(pdf_docs):
    doc_title = []
    for title in pdf_docs:
        doc_title.append(title.name)
    #print(doc_title)
    return doc_title



# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    doc = []
    doc_title = get_doc_title(pdf_docs)
    for i,pdf in enumerate(pdf_docs):
        pdf_text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() or ""  # Handle cases where extract_text returns None

        doc.append(Document(doc_id=doc_title[i],text=pdf_text))

    return doc


# Function to chunk the extracted text
def summary_embeddings(all_docs):
    chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo")
    splitter = SentenceSplitter(chunk_size=1024)
    
    response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=True
    )
    doc_summary_index = DocumentSummaryIndex.from_documents(
    all_docs,
    llm=chatgpt,
    transformations=[splitter],
    response_synthesizer=response_synthesizer,
    show_progress=False,
    )

    doc_summary_index.storage_context.persist("testing_pdf")

    return


def get_conversational_chain (doc_summary_index,memory,user_input):
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=1)
    chat_engine = doc_summary_index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    llm=llm,
    context_prompt=(
        "You are a chatbot, able to have normal interactions, respond to the user question based on the context."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
        "If you are not able to answer based on the context please say Please ask question related to the PDF" 
        ),
    verbose=False,
    )

    response_instance = chat_engine.chat(user_input)

    return response_instance


def main():
    st.header("Docurative-AI Chatbot")

    # Initialize memory and vectorstore in session state
    if "memory" not in st.session_state:
        st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    
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

        This chatbot is built using the Advanced Retrieval-Augmented Generation (RAG) framework, leveraging Generative AI model GPT 3.5 turbo.

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
                    all_docs = get_pdf_text(pdf_docs)
                    #print(all_docs)
                    summary_embeddings(all_docs)
                    # rebuild storage context
                    
                    storage_context = StorageContext.from_defaults(persist_dir="testing_pdf")
                    # Use the StorageContext to load the index
                    index = load_index_from_storage(storage_context)
                    st.session_state.vectorstore = index
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
        response = get_conversational_chain(st.session_state.vectorstore,st.session_state.memory,user_input)
    
        bot_response = response.response
        print("Bot :",bot_response)
        
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
