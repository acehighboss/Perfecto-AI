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
        """PDF 파일 읽기"""
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
        """LlamaParse를 사용한 파일 파싱"""
        parser = LlamaParse(
            api_key=self.llama_api_key,
            result_type="markdown",
            verbose=True,
            premium_mode=True,
        )
        
        parsed_data = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                documents = await parser.aload_data(tmp_file_path)
                parsed_data.extend(documents)
            except Exception as e:
                st.error(f"LlamaParse 처리 중 오류 발생 ({file.name}): {e}")
                # LlamaParse 실패 시 기본 파서 사용
                if file.name.endswith('.pdf'):
                    text = self.read_pdf_file(file)
                elif file.name.endswith('.docx'):
                    text = self.read_docx_file(file)
                elif file.name.endswith('.txt'):
                    text = self.read_txt_file(file)
                else:
                    text = ""
                
                if text:
                    # 임시 Document 객체 생성
                    class TempDoc:
                        def __init__(self, text, metadata):
                            self.text = text
                            self.metadata = metadata
                    
                    parsed_data.append(TempDoc(text, {"source": file.name}))
            finally:
                os.remove(tmp_file_path)
        
        return parsed_data

    def get_documents_from_files(self, uploaded_files):
        """파일에서 문서 추출"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            llama_index_documents = loop.run_until_complete(self.parse_with_llamaparse(uploaded_files))
            
            if llama_index_documents:
                langchain_documents = [
                    LangChainDocument(
                        page_content=doc.text, 
                        metadata=doc.metadata if hasattr(doc, 'metadata') else {"source": "unknown"}
                    )
                    for doc in llama_index_documents
                ]
                return langchain_documents
        except Exception as e:
            st.error(f"파일 처리 오류: {e}")
            return []

    def get_documents_from_url(self, url):
        """URL에서 문서 추출 - 오류 처리 강화"""
        try:
            # 필요한 라이브러리 확인
            try:
                import bs4
                import requests
            except ImportError as e:
                st.error(f"필수 라이브러리 누락: {e}")
                st.info("requirements.txt에 beautifulsoup4, requests를 추가해주세요.")
                return []
        
            # WebBaseLoader 사용
            from langchain_community.document_loaders import WebBaseLoader
        
            # User-Agent 설정으로 차단 방지
            loader = WebBaseLoader(
                url,
                header_template={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
        
            documents = loader.load()
        
            if not documents:
                st.warning("URL에서 콘텐츠를 추출하지 못했습니다.")
                return []
        
            st.success(f"URL에서 {len(documents)}개 문서를 성공적으로 로드했습니다.")
            return documents
        
        except Exception as e:
            st.error(f"URL 로딩 오류: {e}")
        
            # 대안 방법 제시
            st.info("""
            **대안 방법:**
            1. 웹페이지 내용을 복사하여 텍스트 파일로 저장 후 업로드
            2. 웹페이지를 PDF로 저장 후 업로드
            3. 다른 URL 시도
            """)
            
            return []

    def get_namu_wiki_content(self, url):
        """나무위키 전용 로더"""
        try:
            import requests
            from bs4 import BeautifulSoup
        
            # 나무위키 접근을 위한 헤더 설정
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
        
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 나무위키 본문 추출
        content_div = soup.find('div', class_='wiki-content')
        if content_div:
            text = content_div.get_text(strip=True)
            return [{"page_content": text, "metadata": {"source": url}}]
        else:
            st.warning("나무위키 본문을 찾을 수 없습니다.")
            return []
            
        except Exception as e:
            st.error(f"나무위키 접근 오류: {e}")
            return []


