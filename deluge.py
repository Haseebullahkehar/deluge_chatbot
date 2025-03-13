import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ.get("OPENAI_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

if not tavily_api_key:
    st.error("Tavily API key not found. Please set the TAVILY_API_KEY environment variable.")
    st.stop()

# Load and process document
loader = WebBaseLoader("https://www.zoho.com/deluge/help/")
loaded_document = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=5)
chunks = text_splitter.split_documents(loaded_document)

# Generate embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_db = FAISS.from_documents(chunks, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 1})

# Define prompt template
template = """You are an expert in the Deluge programming language, specializing in generating Deluge scripts and providing in-depth explanations. Your task is to assist users with Deluge scripting by following these guidelines:

Understand the user's query and generate the appropriate Deluge code.
Provide a step-by-step explanation of the code, making it easy to understand.
Ensure accuracy by strictly adhering to Deluge syntax and best practices.
If the query is unrelated to Deluge scripting, politely decline by responding with:
"I specialize in Deluge scripting only. Please ask questions related to Deluge.

{context}

Chat History:
{chat_history}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Real-time search tool
search = TavilySearchResults(api_key=tavily_api_key, max_results=1)
tools = [search]
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key)
agent_executor = create_react_agent(llm, tools)

# Streamlit UI
st.set_page_config(
    page_title="Deluge Code Generator",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar with profile links and project description
with st.sidebar:
    st.image("data-1/zoho.JPG", width=150)  # Add Deluge logo
    st.markdown("""
        # Welcome to Deluge Code Generator
        Your virtual assistant for generating and understanding Deluge code snippets. 
        Powered by advanced AI, this assistant is here to help you with Zoho Deluge scripting.
    """)

    # Button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()

# Main chat interface
st.title("Deluge Code Generator")
st.write("Ask me anything about Deluge scripting language!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Enter your query about Deluge scripting:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare chat history for context
    chat_history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages[-5:]]  # Last 5 messages
    )

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({
                "messages": [HumanMessage(content=chat_history + "\nUser: " + user_input)]
            })
            answer = response["messages"][-1].content if "messages" in response else "No response received."
            st.markdown(answer)

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
