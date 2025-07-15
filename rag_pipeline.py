from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# 키워드 검색을 위한 BM25Retriever와 두 검색 결과를 합치는 EnsembleRetriever를 import 합니다.
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from file_handler import get_documents_from_files

def get_retriever_from_source(source_type, source_input):
    """
    URL 또는 파일로부터 문서를 로드하고, 텍스트를 분할하여
    키워드 검색과 의미 검색을 결합한 EnsembleRetriever를 생성합니다.
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
    
    if not splits:
        return None

    # 1. 키워드 기반 검색기(BM25Retriever)를 초기화합니다.
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5

    # 2. 의미 기반 검색기(FAISS)를 초기화합니다.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 3. 두 검색기를 결합하는 앙상블 검색기를 생성합니다.
    # 이 방식은 Cohere API 없이도 안정적인 검색 성능을 제공합니다.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )
    
    return ensemble_retriever


def get_conversational_rag_chain(retriever, system_prompt):
    """
    RAG 기능을 수행하는 체인을 생성합니다. (시스템 프롬프트 적용)
    창의적인 요청을 처리하기 위한 지침이 포함된 프롬프트를 사용합니다.
    """
    template = f"""{system_prompt}

You are an AI assistant. Your task is to answer the user's request based on the provided "Context".
The "Context" is a collection of text snippets from a document or a URL.

**Instructions:**
1. First, carefully read the user's request and the provided "Context".
2. If the user is asking a direct question (e.g., "Who is...?", "What is...?"), find the answer within the "Context" and provide a clear, concise response.
3. If the user is making a creative request (e.g., "write a script", "summarize this in a poem", "create a marketing slogan"), use the main ideas, themes, and key information from the "Context" as the raw material for your creative work. **Do not look for information about the creative task itself (e.g., script writing) in the context.**
4. If the "Context" does not contain relevant information to fulfill the user's request, you must respond with: "죄송합니다. 제공된 내용만으로는 요청하신 작업을 수행할 수 없습니다."

**Context:**
{{context}}

**User's Request:**
{{input}}
"""
    rag_prompt = ChatPromptTemplate.from_template(template)
    # 창의적인 작업을 위해 온도를 약간 높여 답변의 다양성을 확보합니다.
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7) 
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    return create_retrieval_chain(retriever, document_chain)


def get_default_chain(system_prompt):
    """
    기본적인 대화형 체인을 생성합니다. (시스템 프롬프트 적용)
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{question}")]
    )
    # 창의적인 작업을 위해 온도를 약간 높여 답변의 다양성을 확보합니다.
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    return prompt | llm | StrOutputParser()
