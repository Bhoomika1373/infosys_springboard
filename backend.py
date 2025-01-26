import ollama
from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# Load the PDF
def response(user_query):
    loader = PyPDFLoader("healthyheart.pdf")
    data=loader.load()
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
    persist_directory="./chroma_storage"
    )
    retriever = vectorstore.as_retriever()
    
    docs = vectorstore.similarity_search(user_query)

    template = """
    <|context|>
    You are an Medical Assistant that follows the instructions and generate the accurate response based on the query and the context provided.
    Please be truthful and give direct answers.

    <|user|>
    {user_query}

    <|assistant|>
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Local LLM0
    ollama_llm = "tinyllama"
    model_local = ChatOllama(model=ollama_llm)

    # Chain
    chain = (
    {"context": retriever, "user_query": RunnablePassthrough()}
    | prompt
    | model_local
    | StrOutputParser() 
    )
    response=chain.invoke(user_query)
    return(response)