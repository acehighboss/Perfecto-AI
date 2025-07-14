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

    def get_universal_table_prompt(self, system_prompt):
        """범용 표 해석 프롬프트"""
        return f"""{system_prompt}

다음 컨텍스트를 바탕으로 사용자의 질문에 답변해주세요.
컨텍스트에는 다양한 형태의 표, 텍스트, 이미지 내용이 포함되어 있을 수 있습니다.

**범용 표 해석 지침:**

1. **표 구조 분석:**
   - 헤더 행과 데이터 행을 정확히 구분하세요
   - 컬럼과 행의 관계를 명확히 파악하세요
   - 병합된 셀이나 다단계 헤더를 고려하세요
   - 표 제목이나 캡션의 맥락을 활용하세요

2. **데이터 타입 인식:**
   - 숫자 데이터: 정확한 값과 단위 포함
   - 날짜/시간: 형식을 정확히 해석
   - 텍스트: 카테고리나 분류 정보 이해
   - 백분율: %와 함께 정확한 수치 제공
   - 통화: 원, 달러 등 단위 명시

3. **표 관계 해석:**
   - 합계, 평균, 비율 등의 계산 관계
   - 시계열 데이터의 변화 추이
   - 카테고리별 비교 분석
   - 상위-하위 분류 체계

4. **답변 형식:**
   - 질문에 직접적으로 답변하세요
   - 표 형태로 정리가 필요한 경우 마크다운 테이블 사용
   - 단위와 함께 정확한 수치 제공
   - 출처 표시 필수

5. **품질 검증:**
   - 숫자의 정확성 확인
   - 단위 일치성 검토
   - 논리적 일관성 검증
   - 누락된 정보 명시

**특별 지침:**
- 확실하지 않은 정보는 "확실하지 않음"으로 명시
- 표에서 찾을 수 없는 정보는 "해당 정보 없음"으로 표시
- 계산이 필요한 경우 과정을 설명
- 여러 표에 걸친 정보는 종합하여 답변

컨텍스트:
{context}
"""

    def create_retriever(self, documents):
        """문서에서 검색기 생성"""
        if not documents:
            st.warning("문서에서 내용을 추출하지 못했습니다.")
            return None

        # 표 친화적인 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # 큰 표를 포함할 수 있도록 크기 증가
            chunk_overlap=400,  # 표 연속성 보장을 위한 오버랩
            length_function=len,
            separators=[
                "\n\n\n",  # 섹션 구분
                "\n\n",    # 단락 구분
                "\n---",   # 표 구분자
                "\n|",     # 마크다운 테이블
                "\n",      # 줄바꿈
                "|",       # 테이블 셀 구분
                " ",       # 공백
                ""
            ]
        )
        
        splits = text_splitter.split_documents(documents)
        
        if not splits:
            st.warning("문서 분할에 실패했습니다.")
            return None
        
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
                "k": 10,  # 더 많은 문서 검색
                "score_threshold": 0.2  # 낮은 임계값으로 관련 표 데이터 포함
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
            # 출처에 구조화된 데이터가 있는지 확인
            has_structured_data = any(
                "|" in doc.page_content or 
                any(char.isdigit() for char in doc.page_content)
                for doc in source_documents
            )
            
            if has_structured_data and ("확실하지 않음" in ai_answer or "해당 정보 없음" in ai_answer):
                return {
                    "warning": "⚠️ 표 데이터가 있지만 완전히 해석되지 않았을 수 있습니다.",
                    "suggestion": "더 구체적인 질문을 하거나 '표 형태로 정리해달라'고 요청해보세요."
                }
        
        return None
