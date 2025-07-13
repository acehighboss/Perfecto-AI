import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_upstage.document_loaders import UpstageDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

def get_documents_from_url(url: str):
    """URL에서 문서를 로드합니다."""
    try:
        loader = WebBaseLoader(url)
        return loader.load()
    except Exception as e:
        st.error(f"URL을 처리하는 중 오류가 발생했습니다: {e}")
        return None

def get_documents_from_files(uploaded_files):
    """업로드된 파일에서 문서를 로드합니다. (UpstageDocumentLoader 사용)"""
    temp_files = []
    documents = []
    try:
        for uploaded_file in uploaded_files:
            # 임시 파일로 저장하여 경로를 얻습니다.
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_files.append(tmp_file.name)
        
        # UpstageDocumentLoader를 사용하여 파일들을 한 번에 로드합니다.
        loader = UpstageDocumentLoader(temp_files, api_key=os.getenv("UPSTAGE_API_KEY"))
        documents = loader.load()
    except Exception as e:
        st.error(f"파일을 처리하는 중 오류가 발생했습니다: {e}")
        return None
    finally:
        # 임시 파일들을 삭제합니다.
        for file_path in temp_files:
            os.remove(file_path)
    return documents

def get_vector_store(source_input, source_type: str):
    """소스(파일 또는 URL)로부터 문서를 로드하고 벡터 저장소를 생성합니다."""
    if source_type == "URL":
        documents = get_documents_from_url(source_input)
    elif source_type == "Files":
        documents = get_documents_from_files(source_input)
    else:
        documents = []

    if not documents:
        st.warning("문서에서 내용을 추출하지 못했습니다.")
        return None

    # 텍스트를 의미 있는 단위로 분할합니다.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # FAISS 벡터 저장소를 생성하고 반환합니다.
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore
