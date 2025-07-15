from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from file_handler import get_documents_from_files

def get_retriever_from_source(source_type, source_input):
    """
    URL 또는 파일로부터 문서를 로드하고, 텍스트를 분할하여 retriever를 생성합니다.
    """
    documents = []
    if source_type == "URL":
        loader = WebBaseLoader(source_input)
        documents = loader.load()
    elif source_type == "Files":
        documents = get_documents_from_files(source_input)

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def get_conversational_rag_chain(retriever, system_prompt):
    """
    RAG 기능을 수행하는 체인을 생성합니다. (시스템 프롬프트 적용)
    """
    template = f"""{system_prompt}

Answer the user's question based on the context provided below.
If you don't know the answer, just say that you don't know. Don't make up an answer.

Context:
{{context}}

Question:
{{input}}
"""
    rag_prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    return create_retrieval_chain(retriever, document_chain)


def get_default_chain(system_prompt):
    """
    기본적인 대화형 체인을 생성합니다. (시스템 프롬프트 적용)
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{question}")]
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    return prompt | llm | StrOutputParser()
