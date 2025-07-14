import os
import tempfile
import asyncio
import streamlit as st
from langchain_core.documents import Document as LangChainDocument
from langchain_community.document_loaders import WebBaseLoader
from llama_parse import LlamaParse
import PyPDF2
import docx
from io import BytesIO

class FileHandler:
    def __init__(self):
        self.llama_api_key = st.secrets["LLAMA_CLOUD_API_KEY"]
    
    def read_pdf_file(self, file):
        """PDF 파일 읽기 (빠른 대체 방법)"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"PDF 파일 읽기 오류: {e}")
            return ""

    def read_docx_file(self, file):
        """DOCX 파일 읽기"""
        try:
            doc = docx.Document(BytesIO(file.read()))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"DOCX 파일 읽기 오류: {e}")
            return ""

    def read_txt_file(self, file):
        """TXT 파일 읽기"""
        try:
            return file.read().decode('utf-8')
        except Exception as e:
            st.error(f"TXT 파일 읽기 오류: {e}")
            return ""

    async def parse_with_llamaparse(self, uploaded_files):
        """LlamaParse를 사용한 파일 파싱 (타임아웃 설정)"""
        parser = LlamaParse(
            api_key=self.llama_api_key,
            result_type="markdown",
            verbose=False,  # 로그 출력 최소화
        )
        
        parsed_data = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # 타임아웃 설정 (30초)
                documents = await asyncio.wait_for(
                    parser.aload_data(tmp_file_path), 
                    timeout=30.0
                )
                parsed_data.extend(documents)
            except asyncio.TimeoutError:
                st.warning(f"LlamaParse 타임아웃 ({file.name}), 기본 파서 사용")
                # 기본 파서로 대체
                text = self._fallback_parse(file)
                if text:
                    class TempDoc:
                        def __init__(self, text, metadata):
                            self.text = text
                            self.metadata = metadata
                    parsed_data.append(TempDoc(text, {"source": file.name}))
            except Exception as e:
                st.warning(f"LlamaParse 오류 ({file.name}): {e}, 기본 파서 사용")
                text = self._fallback_parse(file)
                if text:
                    class TempDoc:
                        def __init__(self, text, metadata):
                            self.text = text
                            self.metadata = metadata
                    parsed_data.append(TempDoc(text, {"source": file.name}))
            finally:
                os.remove(tmp_file_path)
        
        return parsed_data

    def _fallback_parse(self, file):
        """기본 파서 (빠른 처리)"""
        if file.name.endswith('.pdf'):
            return self.read_pdf_file(file)
        elif file.name.endswith('.docx'):
            return self.read_docx_file(file)
        elif file.name.endswith('.txt'):
            return self.read_txt_file(file)
        return ""

    def get_documents_from_files(self, uploaded_files):
        """파일에서 문서 추출 (성능 최적화)"""
        # 파일 크기 체크
        total_size = sum(file.size for file in uploaded_files)
        if total_size > 10 * 1024 * 1024:  # 10MB 이상
            st.warning("파일이 큽니다. 기본 파서를 사용하여 빠르게 처리합니다.")
            # 큰 파일은 기본 파서 사용
            documents = []
            for file in uploaded_files:
                text = self._fallback_parse(file)
                if text:
                    documents.append(LangChainDocument(
                        page_content=text,
                        metadata={"source": file.name}
                    ))
            return documents
        
        # 작은 파일은 LlamaParse 사용
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        llama_index_documents = loop.run_until_complete(
            self.parse_with_llamaparse(uploaded_files)
        )
        
        if llama_index_documents:
            langchain_documents = [
                LangChainDocument(
                    page_content=doc.text, 
                    metadata=doc.metadata if hasattr(doc, 'metadata') else {"source": "unknown"}
                )
                for doc in llama_index_documents
            ]
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
