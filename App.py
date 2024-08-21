import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory  # Import memory
from langchain.chains import ConversationalRetrievalChain

api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="PDF Q&A System", layout="wide")

st.markdown("""
## PDF Q&A System: Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Generative AI model GPT 3.5 turbo. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works

Follow these simple steps to interact with the chatbot:

1. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

2. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key,
                                  model="text-embedding-3-small")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Adding memory to the conversation chain
def get_conversational_chain(vectorstore,memory):
    prompt_template = """
    Use the context below to answer the question as accurately as possible. If the answer is not in the context, respond with
    'The answer is not available in the context.' Ensure that your response is accurate and concise.

    Context: {context}

    Question: {question}

    Answer:
    """
    model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key,
                       temperature=0)

    # ConversationalRetrievalChain setup with memory
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return qa_chain


def user_input(user_question, vectorstore, memory):
    chain = get_conversational_chain(vectorstore, memory)

    # Retrieve response using the conversational retrieval chain
    response = chain({"question": user_question})

    # Display the output
    st.write("Reply: ", response["answer"])


def main():
    st.header("AI clone chatbot💁")

    # Initialize memory to keep track of conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question:
        # Load the vectorstore and initiate conversation
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        chain = get_conversational_chain(vectorstore, memory)

        # Process the user question
        response = chain({"question": user_question})

        # Display the response
        st.write("Reply: ", response["answer"])

        # Optionally: Show the conversation history
        st.write("Conversation History:", memory.load_memory_variables({})["chat_history"])

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True, key="pdf_uploader")

        if st.button("Submit & Process", key="process_button"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()

