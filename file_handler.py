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
            # TXT 파일은 LlamaParse 사용하지 않고 직접 처리 (빠른 처리)
            if file.name.endswith('.txt'):
                text = self.read_txt_file(file)
                if text:
                    class TempDoc:
                        def __init__(self, text, metadata):
                            self.text = text
                            self.metadata = metadata
                    
                    parsed_data.append(TempDoc(text, {"source": file.name, "type": "txt"}))
                continue
            
            # PDF, DOCX 파일 처리
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # 빠른 처리를 위해 간단한 파일은 기본 파서 사용
                if file.name.endswith('.pdf'):
                    text = self.read_pdf_file(file)
                elif file.name.endswith('.docx'):
                    text = self.read_docx_file(file)
                else:
                    text = ""
                
                if text:
                    class TempDoc:
                        def __init__(self, text, metadata):
                            self.text = text
                            self.metadata = metadata
                    
                    parsed_data.append(TempDoc(text, {"source": file.name, "type": "basic"}))
                else:
                    # 기본 파서 실패 시에만 LlamaParse 사용
                    documents = await parser.aload_data(tmp_file_path)
                    for doc in documents:
                        if hasattr(doc, 'metadata'):
                            doc.metadata["type"] = "llamaparse"
                        else:
                            doc.metadata = {"source": file.name, "type": "llamaparse"}
                    parsed_data.extend(documents)
                    
            except Exception as e:
                st.error(f"파일 처리 중 오류 발생 ({file.name}): {e}")
            finally:
                os.remove(tmp_file_path)
        
        return parsed_data

    def get_documents_from_files(self, uploaded_files):
        """파일에서 문서 추출"""
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
                metadata = doc.metadata if hasattr(doc, 'metadata') else {"source": "unknown"}
                
                langchain_doc = LangChainDocument(
                    page_content=doc.text, 
                    metadata=metadata
                )
                langchain_documents.append(langchain_doc)
            
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
