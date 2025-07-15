import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

def get_documents_from_files(uploaded_files):
    """
    업로드된 파일 리스트(PDF, DOCX, TXT)에서 문서를 로드합니다.
    각 파일을 임시 저장소에 저장한 후, 파일 형식에 맞는 로더를 사용하여 처리합니다.
    """
    all_documents = []
    for uploaded_file in uploaded_files:
        # 임시 파일로 저장하여 경로를 얻음
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            loader = None
            # 파일 확장자에 따라 적절한 로더 선택
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_file_path)
            elif uploaded_file.name.endswith(".docx"):
                loader = Docx2txtLoader(tmp_file_path)
            elif uploaded_file.name.endswith(".txt"):
                loader = TextLoader(tmp_file_path, encoding='utf-8')
            
            if loader:
                all_documents.extend(loader.load())
        
        finally:
            # 처리 후 임시 파일 삭제
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            
    return all_documents
