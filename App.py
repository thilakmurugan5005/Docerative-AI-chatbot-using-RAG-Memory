import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from llama_index.core import Document
from llama_index.llms.openai.base import OpenAI
from llama_index.core.memory import ChatMemoryBuffer

api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Docurative AI", layout="wide")

# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    pdf_text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() or ""  # Handle cases where extract_text returns None
    return pdf_text

# Function to summarize text using OpenAI LLM
def generate_summary(text):
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0)
    prompt = f"Summarize the following text in detail:\n\n{text}"
    response = llm(prompt)
    return response

# Function to handle chatbot conversation
def get_conversational_chain(doc_summary_index, memory, user_input):
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
            "If you are not able to answer based on the context please say Please ask a question related to the PDF."
        ),
        verbose=False,
    )
    response_instance = chat_engine.chat(user_input)
    return response_instance

def main():
    st.header("Docurative-AI Chatbot")

    # Initialize memory in session state
    if "memory" not in st.session_state:
        st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

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
                    # Extract text from PDF
                    pdf_text = get_pdf_text(pdf_docs)

                    # Generate summary using the LLM
                    summary = generate_summary(pdf_text)

                    # Append the summary as the first assistant response
                    st.session_state.messages.append({"role": "assistant", "content": summary})

                    # Once processing is done, hide the intro
                    st.session_state.intro_shown = True
                    st.success("Documents processed. Summary generated.")
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
    if user_input and "vectorstore" in st.session_state:
        # Get the conversational chain
        response = get_conversational_chain(st.session_state.vectorstore, st.session_state.memory, user_input)

        bot_response = response.response

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
