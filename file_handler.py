# file_handler.py (Upstage Loader 버전)

import os
import tempfile
import streamlit as st
from langchain_upstage import UpstageLayoutAnalysisLoader

def get_documents_from_files(uploaded_files):
    """
    업로드된 파일들을 UpstageLayoutAnalysisLoader를 사용하여
    Markdown 형식으로 변환하고 구조를 분석합니다.
    """
    all_documents = []
    # UpstageLayoutAnalysisLoader를 초기화합니다.
    # 이 로더는 파일을 받아 서버로 보낸 뒤, 분석된 결과를 반환합니다.
    layzer = UpstageLayoutAnalysisLoader(
        api_key=os.getenv("UPSTAGE_API_KEY"), split="page"
    )

    for uploaded_file in uploaded_files:
        # 임시 파일로 저장하여 경로를 얻음
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # 단일 파일에 대해 load()를 호출합니다.
            docs = layzer.load(file_path=tmp_file_path)
            all_documents.extend(docs)
        except Exception as e:
            st.error(f"'{uploaded_file.name}' 파일 처리 중 오류 발생: {e}")
        finally:
            os.remove(tmp_file_path)
    
    return all_documents
