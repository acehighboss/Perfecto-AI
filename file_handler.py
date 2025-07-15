import os
import requests
import tempfile
from typing import List, Optional
from io import BytesIO
import streamlit as st
from llama_parse import LlamaParse
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup

class FileHandler:
    def __init__(self, llama_api_key: str):
        self.llama_parser = LlamaParse(api_key=llama_api_key)
    
    def extract_text_from_url(self, url: str) -> Optional[str]:
        """URL에서 텍스트 추출"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 불필요한 태그 제거
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            st.error(f"URL 처리 중 오류 발생: {str(e)}")
            return None
    
    def extract_text_from_pdf(self, file_content: bytes) -> Optional[str]:
        """PDF에서 텍스트 추출 (LlamaParser 우선, 실패시 PyPDF2)"""
        try:
            # LlamaParser 시도
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_file.flush()
                
                try:
                    documents = self.llama_parser.load_data(tmp_file.name)
                    text = "\n".join([doc.text for doc in documents])
                    os.unlink(tmp_file.name)
                    return text
                except:
                    # LlamaParser 실패시 PyPDF2 사용
                    os.unlink(tmp_file.name)
                    
            # PyPDF2 fallback
            pdf_file = BytesIO(file_content)
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
            
        except Exception as e:
            st.error(f"PDF 처리 중 오류 발생: {str(e)}")
            return None
    
    def extract_text_from_docx(self, file_content: bytes) -> Optional[str]:
        """Word 문서에서 텍스트 추출"""
        try:
            doc_file = BytesIO(file_content)
            doc = Document(doc_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Word 문서 처리 중 오류 발생: {str(e)}")
            return None
    
    def extract_text_from_txt(self, file_content: bytes) -> Optional[str]:
        """텍스트 파일에서 텍스트 추출"""
        try:
            text = file_content.decode('utf-8')
            return text
        except UnicodeDecodeError:
            try:
                text = file_content.decode('cp949')
                return text
            except Exception as e:
                st.error(f"텍스트 파일 인코딩 오류: {str(e)}")
                return None
        except Exception as e:
            st.error(f"텍스트 파일 처리 중 오류 발생: {str(e)}")
            return None
    
    def process_file(self, uploaded_file) -> Optional[str]:
        """업로드된 파일 처리"""
        file_extension = uploaded_file.name.split('.')[-1].lower()
        file_content = uploaded_file.read()
        
        if file_extension == 'pdf':
            return self.extract_text_from_pdf(file_content)
        elif file_extension in ['docx', 'doc']:
            return self.extract_text_from_docx(file_content)
        elif file_extension == 'txt':
            return self.extract_text_from_txt(file_content)
        else:
            st.error(f"지원하지 않는 파일 형식: {file_extension}")
            return None
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """텍스트를 청크로 분할"""
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            if end > text_length:
                end = text_length
            
            chunk = text[start:end]
            
            # 문장 경계에서 자르기
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                cut_point = max(last_period, last_newline)
                
                if cut_point > start + chunk_size // 2:
                    chunk = text[start:cut_point + 1]
                    end = cut_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= text_length:
                break
        
        return [chunk for chunk in chunks if chunk]
