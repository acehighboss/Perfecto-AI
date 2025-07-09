# 파일 처리 모듈
# file_handler.py (LlamaParse 제거 버전)

import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

def get_documents_from_files(uploaded_files):
    """
    업로드된 파일들을 확장자에 맞는 기본 로더를 사용하여 처리합니다.
    """
    all_documents = []
    # 파일 처리 중 스피너를 여기에 추가
    with st.spinner("파일을 파싱하고 있습니다..."):
        for uploaded_file in uploaded_files:
            # 임시 파일로 저장하여 경로를 얻음
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            loader = None
            try:
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_file_path)
                elif uploaded_file.name.endswith(".docx"):
                    loader = Docx2txtLoader(tmp_file_path)
                elif uploaded_file.name.endswith(".txt"):
                    loader = TextLoader(tmp_file_path, encoding='utf-8')
                
                if loader:
                    all_documents.extend(loader.load_and_split())

            except Exception as e:
                st.error(f"'{uploaded_file.name}' 파일 처리 중 오류 발생: {e}")
            finally:
                # 임시 파일 삭제
                os.remove(tmp_file_path)

    return all_documents
