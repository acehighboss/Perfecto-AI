import os
import tempfile
import asyncio
import streamlit as st
from langchain_core.documents import Document as LangChainDocument
from langchain_community.document_loaders import WebBaseLoader
from llama_parse import LlamaParse

class FileHandler:
    def __init__(self):
        self.llama_api_key = st.secrets["LLAMA_CLOUD_API_KEY"]
    
    def read_txt_file(self, file):
        """TXT 파일 읽기"""
        try:
            # 파일 내용을 바이트로 읽고 UTF-8로 디코딩
            content = file.read()
            if isinstance(content, bytes):
                # 여러 인코딩 시도
                encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1']
                for encoding in encodings:
                    try:
                        text = content.decode(encoding)
                        return text
                    except UnicodeDecodeError:
                        continue
                # 모든 인코딩 실패 시 에러 처리
                text = content.decode('utf-8', errors='ignore')
                st.warning("일부 문자가 올바르게 읽히지 않을 수 있습니다.")
                return text
            else:
                return content
        except Exception as e:
            st.error(f"TXT 파일 읽기 오류: {e}")
            return ""

    async def parse_with_llamaparse(self, uploaded_files):
        """LlamaParse를 사용한 파일 파싱 (PDF, DOCX만)"""
        parser = LlamaParse(
            api_key=self.llama_api_key,
            result_type="markdown",
            verbose=True,
        )
        
        parsed_data = []
        for file in uploaded_files:
            # TXT 파일은 LlamaParse 사용하지 않고 직접 처리
            if file.name.endswith('.txt'):
                text = self.read_txt_file(file)
                if text:
                    # 임시 Document 객체 생성
                    class TempDoc:
                        def __init__(self, text, metadata):
                            self.text = text
                            self.metadata = metadata
                    
                    parsed_data.append(TempDoc(text, {"source": file.name, "type": "txt"}))
                continue
            
            # PDF, DOCX 파일은 LlamaParse 사용
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                documents = await parser.aload_data(tmp_file_path)
                for doc in documents:
                    # 메타데이터에 파일 타입 추가
                    if hasattr(doc, 'metadata'):
                        doc.metadata["type"] = "llamaparse"
                    else:
                        doc.metadata = {"source": file.name, "type": "llamaparse"}
                parsed_data.extend(documents)
            except Exception as e:
                st.error(f"LlamaParse 처리 중 오류 발생 ({file.name}): {e}")
            finally:
                os.remove(tmp_file_path)
        
        return parsed_data

    def get_documents_from_files(self, uploaded_files):
        """파일에서 문서 추출 - TXT 파일 지원"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        llama_index_documents = loop.run_until_complete(
            self.parse_with_llamaparse(uploaded_files)
        )
        
        if llama_index_documents:
            langchain_documents = []
            for doc in llama_index_documents:
                # 문서 타입별 메타데이터 설정
                metadata = doc.metadata if hasattr(doc, 'metadata') else {"source": "unknown"}
                
                langchain_doc = LangChainDocument(
                    page_content=doc.text, 
                    metadata=metadata
                )
                langchain_documents.append(langchain_doc)
            
            # 처리된 파일 정보 표시
            txt_files = [doc for doc in llama_index_documents if doc.metadata.get("type") == "txt"]
            llamaparse_files = [doc for doc in llama_index_documents if doc.metadata.get("type") == "llamaparse"]
            
            if txt_files:
                st.info(f"📄 TXT 파일 {len(txt_files)}개 직접 처리 완료")
            if llamaparse_files:
                st.info(f"🔍 PDF/DOCX 파일 {len(llamaparse_files)}개 LlamaParse 처리 완료")
            
            return langchain_documents
        return []

    def get_documents_from_url(self, url):
        """URL에서 문서 추출"""
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            if not documents:
                st.warning("URL에서 콘텐츠를 추출하지 못했습니다.")
                return []
            
            st.success(f"URL에서 {len(documents)}개 문서를 성공적으로 로드했습니다.")
            return documents
            
        except Exception as e:
            st.error(f"URL 로딩 오류: {e}")
            return []
