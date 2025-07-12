# file_handler.py (Upstage Loader 수정 버전)

import os
import tempfile
import streamlit as st
# [수정 1] 올바른 클래스 이름인 UpstageDocumentLoader를 import 합니다.
from langchain_upstage import UpstageDocumentLoader

def get_documents_from_files(uploaded_files):
    """
    업로드된 파일들을 UpstageDocumentLoader를 사용하여
    Markdown 형식으로 변환하고 구조를 분석합니다.
    """
    all_documents = []
    
    # 여러 파일을 효율적으로 처리하기 위해 임시 파일 경로를 먼저 리스트에 담습니다.
    temp_file_paths = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_paths.append(tmp_file.name)
    
    if temp_file_paths:
        try:
            # [수정 2] UpstageDocumentLoader를 초기화하고 한 번에 모든 파일을 처리합니다.
            # 이 로더는 파일을 받아 서버로 보낸 뒤, 분석된 결과를 반환합니다.
            loader = UpstageDocumentLoader(
                file_path=temp_file_paths,
                api_key=os.getenv("UPSTAGE_API_KEY"),
                split="page"
            )
            docs = loader.load()
            all_documents.extend(docs)
        except Exception as e:
            st.error(f"파일 처리 중 오류 발생: {e}")
        finally:
            # 모든 임시 파일을 삭제합니다.
            for path in temp_file_paths:
                os.remove(path)
    
    return all_documents
