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
            
            # 테이블 데이터 추출
            for table in doc.tables:
                text += "\n--- 테이블 ---\n"
                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells])
                    text += row_text + "\n"
                text += "--- 테이블 끝 ---\n\n"
            
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

    def get_universal_parsing_instruction(self, document_type="general"):
        """문서 유형에 따른 범용 파싱 지시사항"""
        base_instruction = """
        이 문서를 정확하게 파싱해주세요. 다음 사항에 특히 주의하여 처리하세요:
        
        **테이블 처리:**
        1. 모든 테이블의 구조를 정확히 보존하세요
        2. 헤더와 데이터의 관계를 명확히 하세요
        3. 숫자, 텍스트, 날짜 등 모든 데이터 타입을 정확히 파싱하세요
        4. 병합된 셀의 정보도 누락하지 마세요
        5. 테이블 캡션이나 제목이 있다면 포함하세요
        
        **데이터 정확성:**
        1. 단위 정보(%, 원, 달러, kg, 개 등)를 누락하지 마세요
        2. 날짜 형식을 정확히 보존하세요
        3. 소수점, 천단위 구분자를 정확히 유지하세요
        4. 특수 문자나 기호의 의미를 보존하세요
        
        **구조 보존:**
        1. 섹션별 구분을 명확히 하세요
        2. 목록이나 계층 구조를 유지하세요
        3. 각주나 참고사항도 포함하세요
        """
        
        type_specific = {
            "financial": """
            **재무 문서 특화:**
            - 분기별, 연도별 데이터 구분
            - 부문별, 계정별 분류 정보
            - 증감률, 비율 정보
            """,
            "research": """
            **연구 문서 특화:**
            - 실험 데이터와 결과 테이블
            - 통계 수치와 p-value
            - 그래프나 차트 설명
            """,
            "inventory": """
            **재고/물류 문서 특화:**
            - 품목별, 창고별 분류
            - 수량, 단가, 금액 정보
            - 입출고 날짜와 담당자
            """,
            "hr": """
            **인사 문서 특화:**
            - 직급, 부서별 분류
            - 급여, 평가 등급 정보
            - 날짜와 기간 정보
            """,
            "sales": """
            **영업 문서 특화:**
            - 제품별, 지역별 분류
            - 매출액, 수량, 단가 정보
            - 고객사별 실적 데이터
            """
        }
        
        return base_instruction + type_specific.get(document_type, "")

    async def parse_with_llamaparse(self, uploaded_files, document_type="general"):
        """LlamaParse를 사용한 범용 파일 파싱"""
        parser = LlamaParse(
            api_key=self.llama_api_key,
            result_type="markdown",
            verbose=True,
            premium_mode=True,
            parsing_instruction=self.get_universal_parsing_instruction(document_type)
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
                    class TempDoc:
                        def __init__(self, text, metadata):
                            self.text = text
                            self.metadata = metadata
                    
                    parsed_data.append(TempDoc(text, {"source": file.name}))
            finally:
                os.remove(tmp_file_path)
        
        return parsed_data

    def get_documents_from_files(self, uploaded_files, document_type="general"):
        """파일에서 문서 추출"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            llama_index_documents = loop.run_until_complete(
                self.parse_with_llamaparse(uploaded_files, document_type)
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
        except Exception as e:
            st.error(f"파일 처리 오류: {e}")
            return []

    def get_documents_from_url(self, url):
        """URL에서 문서 추출"""
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
            st.info("""
            **대안 방법:**
            1. 웹페이지 내용을 복사하여 텍스트 파일로 저장 후 업로드
            2. 웹페이지를 PDF로 저장 후 업로드
            3. 다른 URL 시도
            """)
            return []
