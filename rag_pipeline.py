from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# 키워드 검색, 앙상블 검색, LLM 재평가를 위한 도구들을 import 합니다.
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from file_handler import get_documents_from_files

def get_retriever_from_source(source_type, source_input):
    """
    URL 또는 파일로부터 문서를 로드하고, 텍스트를 분할하여
    정확도를 극대화한 최종 RAG Retriever를 생성합니다.
    """
    documents = []
    if source_type == "URL":
        # WebBaseLoader가 가끔 빈 문서를 반환하는 경우를 대비한 예외 처리
        try:
            loader = WebBaseLoader(source_input)
            documents = loader.load()
        except Exception as e:
            print(f"Error loading URL: {e}")
            return None
    elif source_type == "Files":
        documents = get_documents_from_files(source_input)

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)
    
    if not splits:
        return None

    # 1. 키워드 기반 검색기(BM25Retriever)를 초기화합니다.
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 10 # 후보군을 10개로 늘려 더 넓게 탐색

    # 2. 의미 기반 검색기(FAISS)를 초기화합니다.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 3. 두 검색기를 결합하는 앙상블 검색기를 생성합니다. (1차 검색)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )

    # 4. 검색된 문서를 압축(재평가)할 LLM 기반 압축기를 생성합니다.
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    
    # 5. 압축기와 앙상블 검색기를 결합한 최종 Retriever를 생성합니다. (2차 필터링)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )
    
    return compression_retriever


def get_conversational_rag_chain(retriever, system_prompt):
    """
    RAG 기능을 수행하는 체인을 생성합니다. (시스템 프롬프트 적용)
    """
    template = f"""{system_prompt}

Answer the user's question or perform the requested task based on the context provided below.
If the context is empty or not relevant, say that you cannot fulfill the request with the given information.

Context:
{{context}}

User's Request:
{{input}}
"""
    rag_prompt = ChatPromptTemplate.from_template(template)
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
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    return prompt | llm | StrOutputParser()
