# rag_pipeline.py - 버전 1 수정: BM25 + Cohere Rerank

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import BM25Retriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_core.documents import Document as LangChainDocument

from file_handler import get_documents_from_files

def get_retriever_from_source(source_type, source_input):
    """
    BM25로 후보군을 찾고 Cohere Rerank로 재정렬하는 Retriever를 생성합니다.
    """
    documents = []
    if source_type == "URL":
        try:
            loader = WebBaseLoader(source_input)
            documents = loader.load()
        except Exception as e:
            print(f"Error loading URL: {e}")
            return None
    elif source_type == "Files":
        txt_files = [f for f in source_input if f.name.endswith('.txt')]
        other_files = [f for f in source_input if not f.name.endswith('.txt')]

        for txt_file in txt_files:
            content = txt_file.getvalue().decode('utf-8')
            documents.append(LangChainDocument(page_content=content, metadata={"source": txt_file.name}))

        if other_files:
            llama_documents = get_documents_from_files(other_files)
            if llama_documents:
                documents.extend([LangChainDocument(page_content=doc.text, metadata=doc.metadata) for doc in llama_documents])

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    if not splits:
        return None

    # --- 핵심 변경 부분 ---
    # 1. 1차 후보군을 넓게 찾기 위해 BM25 Retriever를 생성하고 k값을 20으로 설정합니다.
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 20
    
    # 2. Cohere Rerank 압축기를 생성합니다. 최종 5개를 선택합니다.
    compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=5)
    
    # 3. BM25 Retriever와 Rerank 압축기를 결합하여 최종 Retriever를 생성합니다.
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=bm25_retriever
    )
    
    return compression_retriever


def get_conversational_rag_chain(retriever, system_prompt):
    """
    (모든 버전에서 동일) 전달받은 Retriever와 LLM을 연결하여 RAG 체인을 생성합니다.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    rag_prompt_template = f"""{system_prompt}

Answer the user's request based on the provided "Context".

**Context:**
{{context}}

**User's Request:**
{{input}}
"""
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
    
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    
    return create_retrieval_chain(retriever, document_chain)


def get_default_chain(system_prompt):
    """
    (모든 버전에서 동일) 문서가 없을 때 사용되는 기본 대화 체인입니다.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{question}")]
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    return prompt | llm | StrOutputParser()
