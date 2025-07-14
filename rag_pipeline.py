import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_upstage import UpstageEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
import re
import nltk
from nltk.tokenize import sent_tokenize

# NLTK 데이터 다운로드 (초기 실행 시 한 번만)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SmartTextSplitter:
    """문단/문장 기반 지능형 텍스트 분할기"""
    
    def __init__(self, max_chunk_size=2000, overlap_sentences=2):
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences
    
    def detect_table_boundaries(self, text):
        """표 경계 감지"""
        table_patterns = [
            r'\|.*\|',  # 마크다운 테이블
            r'─+',      # 표 구분선
            r'┌.*┐',    # 박스 테이블
            r'^\s*\d+\.\s*\d+',  # 숫자 데이터 패턴
            r'^\s*[가-힣]+\s*\|\s*[가-힣]+',  # 한글 테이블 패턴
        ]
        
        table_regions = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            for pattern in table_patterns:
                if re.search(pattern, line):
                    # 표 시작/끝 찾기
                    start = max(0, i - 2)
                    end = min(len(lines), i + 10)
                    table_regions.append((start, end))
                    break
        
        return table_regions
    
    def split_by_paragraphs(self, text):
        """문단별 분할"""
        # 표 영역 감지
        table_regions = self.detect_table_boundaries(text)
        
        # 문단 분할 (빈 줄 기준)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 표가 포함된 문단은 분할하지 않음
            is_table_paragraph = self._is_table_content(paragraph)
            
            if is_table_paragraph:
                # 현재 청크가 있으면 저장
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_size = 0
                
                # 표 문단은 독립적으로 저장
                chunks.append(paragraph)
                continue
            
            # 일반 문단 처리
            paragraph_size = len(paragraph)
            
            if current_size + paragraph_size > self.max_chunk_size and current_chunk:
                # 현재 청크 저장
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
                current_size = paragraph_size
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_size += paragraph_size
        
        # 마지막 청크 저장
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def split_by_sentences(self, text):
        """문장별 분할 (문단 분할이 어려운 경우)"""
        # 표 영역 감지
        table_regions = self.detect_table_boundaries(text)
        
        # 문장 분할
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        sentence_buffer = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 표 관련 문장 감지
            is_table_sentence = self._is_table_content(sentence)
            
            if is_table_sentence:
                # 현재 청크가 있으면 저장
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_size = 0
                    sentence_buffer = []
                
                # 표 관련 문장들을 모아서 처리
                sentence_buffer.append(sentence)
                continue
            
            # 표 문장 버퍼가 있으면 처리
            if sentence_buffer:
                table_chunk = " ".join(sentence_buffer)
                chunks.append(table_chunk)
                sentence_buffer = []
            
            # 일반 문장 처리
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                # 오버랩을 위해 마지막 몇 문장 유지
                overlap_text = self._get_overlap_text(current_chunk)
                chunks.append(current_chunk.strip())
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_size = len(current_chunk)
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_size += sentence_size
        
        # 마지막 처리
        if sentence_buffer:
            table_chunk = " ".join(sentence_buffer)
            chunks.append(table_chunk)
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _is_table_content(self, text):
        """텍스트가 표 내용인지 판단"""
        table_indicators = [
            r'\|.*\|',  # 마크다운 테이블
            r'─+',      # 표 구분선
            r'┌.*┐',    # 박스 테이블
            r'^\s*\d+\s*\|\s*\d+',  # 숫자 테이블
            r'^\s*[가-힣]+\s*\|\s*[가-힣]+',  # 한글 테이블
            r'^\s*\d+\.\d+\s*\d+\.\d+',  # 숫자 데이터 패턴
            r'(매출|수익|비용|자산|부채|자본).*\d+',  # 재무 데이터
            r'\d+\s*(억|만|천|원|달러|\$|%)',  # 단위가 있는 숫자
            r'(분기|연도|월별|일별).*\d+',  # 시계열 데이터
        ]
        
        for pattern in table_indicators:
            if re.search(pattern, text):
                return True
        
        # 숫자 밀도 체크 (숫자가 많으면 표일 가능성 높음)
        numbers = re.findall(r'\d+', text)
        if len(numbers) > 3 and len(text) < 200:
            return True
        
        return False
    
    def _get_overlap_text(self, text):
        """오버랩을 위한 마지막 문장들 추출"""
        sentences = sent_tokenize(text)
        if len(sentences) <= self.overlap_sentences:
            return ""
        
        overlap_sentences = sentences[-self.overlap_sentences:]
        return " ".join(overlap_sentences)
    
    def split_documents(self, documents):
        """문서 분할 메인 함수"""
        split_docs = []
        
        for doc in documents:
            text = doc.page_content
            
            # 1차: 문단별 분할 시도
            try:
                chunks = self.split_by_paragraphs(text)
                
                # 문단별 분할이 효과적이지 않으면 문장별 분할
                if len(chunks) < 2 or any(len(chunk) > self.max_chunk_size * 1.5 for chunk in chunks):
                    chunks = self.split_by_sentences(text)
                
            except Exception:
                # 실패 시 문장별 분할
                chunks = self.split_by_sentences(text)
            
            # Document 객체 생성
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    split_docs.append(Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            'chunk_id': i,
                            'chunk_type': 'table' if self._is_table_content(chunk) else 'text'
                        }
                    ))
        
        return split_docs

class RAGPipeline:
    def __init__(self):
        self.upstage_api_key = st.secrets["UPSTAGE_API_KEY"]
        self.google_api_key = st.secrets["GOOGLE_API_KEY"]
        self.embeddings = UpstageEmbeddings(
            api_key=self.upstage_api_key,
            model="solar-embedding-1-large"
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0,
            google_api_key=self.google_api_key
        )
        self.text_splitter = SmartTextSplitter(max_chunk_size=1500, overlap_sentences=2)

    def get_universal_table_prompt(self, system_prompt):
        """범용 표 해석 프롬프트"""
        return f"""{system_prompt}

다음 컨텍스트를 바탕으로 사용자의 질문에 답변해주세요.
컨텍스트는 문단별/문장별로 의미적으로 분할되어 있으며, 표 구조가 보존되어 있습니다.

**지능형 분할 시스템 특징:**
- 표 내용은 분할되지 않고 완전한 형태로 제공됩니다
- 문단별 분할로 의미적 연관성이 높습니다
- 문장별 오버랩으로 맥락이 보존됩니다

**범용 표 해석 지침:**

1. **표 구조 분석:**
   - 완전한 표 구조를 활용하여 정확한 해석
   - 헤더와 데이터의 관계를 명확히 파악
   - 병합된 셀이나 다단계 헤더 고려
   - 표 제목과 캡션의 맥락 활용

2. **데이터 정확성:**
   - 숫자 데이터와 단위를 정확히 매칭
   - 날짜/시간 형식을 정확히 해석
   - 백분율과 비율 정보 정확히 처리
   - 통화 단위 명시

3. **의미적 연관성:**
   - 문단별 분할로 관련 정보가 함께 제공됨
   - 표와 관련된 설명 텍스트 연결
   - 시계열 데이터의 변화 추이 분석
   - 카테고리별 비교 분석

4. **답변 형식:**
   - 질문에 직접적으로 답변
   - 표 형태 정리 시 마크다운 테이블 사용
   - 단위와 함께 정확한 수치 제공
   - 출처 명시 필수

**특별 지침:**
- 표 청크는 완전한 형태로 제공되므로 정확한 해석 가능
- 문단별 맥락을 활용하여 포괄적 답변 제공
- 확실하지 않은 정보는 "확실하지 않음"으로 명시
- 계산이 필요한 경우 과정을 설명

컨텍스트:
{context}
"""

    def create_retriever(self, documents):
        """문서에서 검색기 생성 - 지능형 분할 적용"""
        if not documents:
            st.warning("문서에서 내용을 추출하지 못했습니다.")
            return None

        # 지능형 텍스트 분할 적용
        splits = self.text_splitter.split_documents(documents)
        
        if not splits:
            st.warning("문서 분할에 실패했습니다.")
            return None
        
        # 분할 결과 정보 표시
        table_chunks = [doc for doc in splits if doc.metadata.get('chunk_type') == 'table']
        text_chunks = [doc for doc in splits if doc.metadata.get('chunk_type') == 'text']
        
        st.info(f"📊 지능형 분할 완료: 표 청크 {len(table_chunks)}개, 텍스트 청크 {len(text_chunks)}개")
        
        # FAISS 벡터스토어 생성
        try:
            vectorstore = FAISS.from_documents(splits, self.embeddings)
        except Exception as e:
            st.error(f"벡터스토어 생성 오류: {e}")
            return None

        # 검색기 설정 - 표 데이터를 위해 더 많은 문서 검색
        base_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 12,  # 더 많은 문서 검색 (표 + 관련 텍스트)
                "score_threshold": 0.15  # 낮은 임계값으로 관련 정보 포함
            }
        )
        
        # 압축 검색기 사용
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return compression_retriever

    def create_conversational_rag_chain(self, retriever, system_prompt):
        """대화형 RAG 체인 생성"""
        enhanced_template = self.get_universal_table_prompt(system_prompt)
        
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", enhanced_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        
        document_chain = create_stuff_documents_chain(self.llm, rag_prompt)
        return create_retrieval_chain(retriever, document_chain)

    def create_default_chain(self, system_prompt):
        """기본 체인 생성"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}"),
        ])
        return prompt | self.llm | StrOutputParser()

    def format_chat_history(self, messages):
        """채팅 히스토리 포맷팅"""
        chat_history = []
        for msg in messages[:-1]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))
        return chat_history

    def validate_table_response(self, user_input, ai_answer, source_documents):
        """표 관련 답변 검증"""
        table_indicators = [
            "표", "테이블", "table", "매출", "실적", "데이터", "수치", "금액", 
            "비율", "퍼센트", "%", "분기", "연도", "부문", "항목", "합계", 
            "평균", "최대", "최소", "증가", "감소", "비교", "순위"
        ]
        
        is_table_question = any(indicator in user_input.lower() for indicator in table_indicators)
        
        if is_table_question:
            # 표 청크가 있는지 확인
            table_chunks = [doc for doc in source_documents 
                          if doc.metadata.get('chunk_type') == 'table']
            
            if table_chunks and ("확실하지 않음" in ai_answer or "해당 정보 없음" in ai_answer):
                return {
                    "warning": "⚠️ 표 데이터가 완전한 형태로 제공되었지만 해석에 어려움이 있을 수 있습니다.",
                    "suggestion": "더 구체적인 질문을 하거나 '표 형태로 정리해달라'고 요청해보세요."
                }
            
            if not table_chunks:
                return {
                    "warning": "⚠️ 표 관련 질문이지만 관련 표 데이터를 찾지 못했습니다.",
                    "suggestion": "문서에 해당 표가 있는지 확인하거나 다른 키워드로 질문해보세요."
                }
        
        return None
