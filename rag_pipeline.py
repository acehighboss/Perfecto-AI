import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import re

class SmartTextSplitter:
    """개선된 문단/문장 기반 지능형 텍스트 분할기"""
    
    def __init__(self, max_chunk_size=1000, overlap_sentences=1):
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences
    
    def flexible_sentence_split(self, text):
        """유연한 문장 분할 - 마침표 없는 문장도 고려"""
        # 명확한 문장 끝 패턴
        sentence_endings = r'[.!?;]["\']?\s+'
        
        # 1차: 명확한 문장 끝으로 분할
        sentences = re.split(sentence_endings, text)
        
        # 2차: 마침표 없는 문장들을 줄바꿈으로 분할
        refined_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 긴 문장이고 줄바꿈이 있으면 줄바꿈 단위로 추가 분할
            if len(sentence) > 200 and '\n' in sentence:
                lines = sentence.split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        refined_sentences.append(line)
            else:
                refined_sentences.append(sentence)
        
        # 3차: 너무 짧은 문장들을 인접한 문장과 병합
        final_sentences = []
        current_sentence = ""
        
        for sentence in refined_sentences:
            if len(sentence) < 50 and current_sentence:
                # 짧은 문장은 이전 문장과 병합
                current_sentence += " " + sentence
            else:
                if current_sentence:
                    final_sentences.append(current_sentence)
                current_sentence = sentence
        
        if current_sentence:
            final_sentences.append(current_sentence)
        
        return final_sentences
    
    def split_by_paragraphs(self, text):
        """문단별 분할 (우선 방법)"""
        # 문단 분할 (빈 줄 기준)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_size = len(paragraph)
            
            # 토큰 제한 확인
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
        """문장별 분할 (토큰 제한 초과 시)"""
        sentences = self.flexible_sentence_split(text)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence)
            
            # 단일 문장이 토큰 제한을 초과하는 경우
            if sentence_size > self.max_chunk_size:
                # 현재 청크 저장
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_size = 0
                
                # 긴 문장을 강제로 분할
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk + " " + word) > self.max_chunk_size:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word
                    else:
                        temp_chunk += " " + word if temp_chunk else word
                
                if temp_chunk:
                    chunks.append(temp_chunk.strip())
                continue
            
            # 토큰 제한 확인
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                # 오버랩 처리
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
        
        # 마지막 청크 저장
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap_text(self, text):
        """오버랩을 위한 마지막 문장들 추출"""
        sentences = self.flexible_sentence_split(text)
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
                
                # 문단별 분할 결과 검증
                oversized_chunks = [chunk for chunk in chunks if len(chunk) > self.max_chunk_size * 1.2]
                
                if oversized_chunks:
                    # 토큰 제한 초과 시 문장별 분할
                    st.info(f"토큰 제한 초과로 문장별 분할 적용 ({len(oversized_chunks)}개 청크)")
                    chunks = self.split_by_sentences(text)
                
            except Exception as e:
                st.warning(f"지능형 분할 실패, 기본 분할 사용: {e}")
                # 실패 시 기본 RecursiveCharacterTextSplitter 사용
                fallback_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=100
                )
                fallback_docs = fallback_splitter.split_documents([doc])
                chunks = [d.page_content for d in fallback_docs]
            
            # Document 객체 생성
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    split_docs.append(Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            'chunk_id': i,
                            'chunk_length': len(chunk)
                        }
                    ))
        
        return split_docs

class RAGPipeline:
    def __init__(self):
        self.google_api_key = st.secrets["GOOGLE_API_KEY"]
        
        # Google 무료 임베딩 모델 사용 (text-embedding-004)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=self.google_api_key
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0,
            google_api_key=self.google_api_key
        )
        
        # 개선된 텍스트 분할기
        self.text_splitter = SmartTextSplitter(max_chunk_size=1000, overlap_sentences=1)

    def get_system_prompt_template(self, system_prompt):
        """시스템 프롬프트 템플릿 - 범용적 사용"""
        template = system_prompt + """

다음 컨텍스트를 바탕으로 사용자의 질문에 정확하게 답변해주세요.

**답변 지침:**
1. 제공된 컨텍스트의 정보를 우선적으로 사용하여 답변하세요
2. 컨텍스트에 관련 정보가 부족한 경우, 일반적인 지식을 활용하여 보충 설명하되 "(일반 지식 기반)"이라고 표시하세요
3. 답변할 때는 참조한 출처를 명시해주세요
4. 정확한 정보만을 제공하고, 추측이나 가정은 피하세요
5. 가능한 한 구체적이고 상세한 답변을 제공하세요

컨텍스트:
{context}
"""
        return template

    def create_retriever(self, documents):
        """하이브리드 검색기 생성 (BM25 + 벡터 검색)"""
        if not documents:
            st.warning("문서에서 내용을 추출하지 못했습니다.")
            return None

        # 개선된 텍스트 분할 적용
        try:
            splits = self.text_splitter.split_documents(documents)
        except Exception as e:
            st.error(f"텍스트 분할 실패: {e}")
            # 기본 분할기로 대체
            fallback_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            splits = fallback_splitter.split_documents(documents)
        
        if not splits:
            st.warning("문서 분할에 실패했습니다.")
            return None
        
        # 분할 결과 정보 표시
        avg_chunk_length = sum(len(doc.page_content) for doc in splits) / len(splits)
        st.info(f"📊 분할 완료: {len(splits)}개 청크, 평균 길이: {avg_chunk_length:.0f}자")
        
        try:
            # 1. 벡터 검색기 생성 (의미적 유사도)
            vectorstore = FAISS.from_documents(splits, self.embeddings)
            vector_retriever = vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={
                    "k": 10,              # 검색 결과 수 증가
                    "score_threshold": 0.15  # 임계값 낮춤 (더 관대하게)
                }
            )
            
            # 2. BM25 검색기 생성 (키워드 매칭)
            bm25_retriever = BM25Retriever.from_documents(splits)
            bm25_retriever.k = 8  # BM25로 8개 문서 검색
            
            # 3. 하이브리드 앙상블 검색기 생성
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.4, 0.6]  # BM25: 40%, 벡터: 60%
            )
            
            st.success("🔍 하이브리드 검색기 (BM25 + 벡터) 생성 완료")
            
            # 4. 압축 검색기 적용 (선택적)
            if len(splits) < 30:  # 작은 문서만 압축 검색 사용
                compressor = LLMChainExtractor.from_llm(self.llm)
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=ensemble_retriever
                )
                return compression_retriever
            else:
                return ensemble_retriever
                
        except Exception as e:
            st.error(f"하이브리드 검색기 생성 오류: {e}")
            # 오류 발생 시 기본 벡터 검색기만 사용
            try:
                vectorstore = FAISS.from_documents(splits, self.embeddings)
                return vectorstore.as_retriever(
                    search_type="similarity", 
                    search_kwargs={"k": 10, "score_threshold": 0.15}
                )
            except Exception as e2:
                st.error(f"기본 검색기 생성도 실패: {e2}")
                return None

    def create_conversational_rag_chain(self, retriever, system_prompt):
        """대화형 RAG 체인 생성"""
        enhanced_template = self.get_system_prompt_template(system_prompt)
        
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
