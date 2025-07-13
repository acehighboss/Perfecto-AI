import os
import tempfile
import streamlit as st
import asyncio
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument
from llama_parse import LlamaParse

# LlamaParse를 비동기적으로 실행하기 위한 헬퍼 함수
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def get_documents_from_url(url: str):
    """URL에서 문서를 로드합니다."""
    try:
        loader = WebBaseLoader(url)
        return loader.load()
    except Exception as e:
        st.error(f"URL을 처리하는 중 오류가 발생했습니다: {e}")
        return None

def get_documents_from_files(uploaded_files):
    """업로드된 파일에서 LlamaParse를 사용하여 문서를 로드합니다."""
    temp_file_paths = []
    try:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_paths.append(tmp_file.name)
        
        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            verbose=True,
        )

        llama_index_documents = run_async(parser.aload_data(temp_file_paths))
        
        # [수정] LangChain 문서 형식으로 변환 시, 파일 이름과 페이지 번호를 메타데이터에 추가합니다.
        langchain_documents = []
        for doc in llama_index_documents:
            # LlamaParse가 반환하는 메타데이터에서 파일명과 페이지 번호를 추출
            file_name = doc.metadata.get("file_name", "N/A")
            page_label = doc.metadata.get("page_label", "N/A")
            
            # 메타데이터를 재구성하여 LangChain Document 객체 생성
            new_metadata = {"source": file_name, "page": page_label}
            langchain_documents.append(LangChainDocument(page_content=doc.text, metadata=new_metadata))
            
        return langchain_documents

    except Exception as e:
        st.error(f"LlamaParse로 파일을 처리하는 중 오류가 발생했습니다: {e}")
        return None
    finally:
        for path in temp_file_paths:
            os.remove(path)

def get_vector_store(source_input, source_type: str):
    """소스(파일 또는 URL)로부터 문서를 로드하고 벡터 저장소를 생성합니다."""
    with st.spinner(f"{source_type}을(를) 분석하고 벡터 저장소를 생성 중입니다..."):
        documents = []
        if source_type == "URL":
            documents = get_documents_from_url(source_input)
        elif source_type == "Files":
            documents = get_documents_from_files(source_input)

        if not documents:
            st.warning("문서에서 내용을 추출하지 못했습니다.")
            return None

        # [수정] Chunk Size를 줄이고 Overlap을 늘려 더 촘촘하게 문서를 분할합니다.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore
