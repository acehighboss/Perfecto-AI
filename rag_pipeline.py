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
    """문단/문장 기반 지능형 텍스트 분할기"""
    
    def __init__(self, max_tokens=3000, overlap_sentences=2):
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
    
    def estimate_tokens(self, text):
        """토큰 수 추정 (한국어 기준 대략 2.5자 = 1토큰)"""
        return len(text) // 2.5
    
    def split_by_sentences(self, text):
        """문장별 분할"""
        # 한국어와 영어 문장 끝 패턴
        sentence_pattern = r'[.!?;]\s+|[。！？；]\s*|[\n]+'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def split_by_paragraphs(self, text):
        """문단별 분할"""
        # 빈 줄 기준으로 문단 분할
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    def smart_split_documents(self, documents):
        """지능형 문서 분할"""
        split_docs = []
        
        for doc in documents:
            text = doc.page_content
            source = doc.metadata.get('source', 'unknown')
            
            # 1단계: 문단별 분할 시도
            paragraphs = self.split_by_paragraphs(text)
            
            for para_idx, paragraph in enumerate(paragraphs):
                tokens = self.estimate_tokens(paragraph)
                
                # 문단이 토큰 제한 이내라면 그대로 사용
                if tokens <= self.max_tokens:
                    split_docs.append(Document(
                        page_content=paragraph,
                        metadata={
                            'source': source,
                            'chunk_type': 'paragraph',
                            'chunk_index': para_idx,
                            'token_count': tokens
                        }
                    ))
                else:
                    # 문단이 너무 길면 문장별로 분할
                    sentences = self.split_by_sentences(paragraph)
                    current_chunk = ""
                    current_tokens = 0
                    sentence_buffer = []
                    
                    for sentence in sentences:
                        sentence_tokens = self.estimate_tokens(sentence)
                        
                        # 문장이 토큰 제한을 초과하면 강제로 자름
                        if sentence_tokens > self.max_tokens:
                            if current_chunk:
                                split_docs.append(Document(
                                    page_content=current_chunk,
                                    metadata={
                                        'source': source,
                                        'chunk_type': 'sentence_group',
                                        'chunk_index': f"{para_idx}_{len(split_docs)}",
                                        'token_count': current_tokens
                                    }
                                ))
                                current_chunk = ""
                                current_tokens = 0
                            
                            # 긴 문장을 강제로 자름
                            truncated = sentence[:int(self.max_tokens * 2.5)]
                            split_docs.append(Document(
                                page_content=truncated,
                                metadata={
                                    'source': source,
                                    'chunk_type': 'truncated_sentence',
                                    'chunk_index': f"{para_idx}_{len(split_docs)}",
                                    'token_count': self.max_tokens
                                }
                            ))
                            continue
                        
                        # 현재 청크에 추가할 수 있는지 확인
                        if current_tokens + sentence_tokens <= self.max_tokens:
                            current_chunk += " " + sentence if current_chunk else sentence
                            current_tokens += sentence_tokens
                            sentence_buffer.append(sentence)
                        else:
                            # 현재 청크 저장
                            if current_chunk:
                                split_docs.append(Document(
                                    page_content=current_chunk,
                                    metadata={
                                        'source': source,
                                        'chunk_type': 'sentence_group',
                                        'chunk_index': f"{para_idx}_{len(split_docs)}",
                                        'token_count': current_tokens
                                    }
                                ))
                            
                            # 새 청크 시작 (오버랩 적용)
                            overlap_sentences = sentence_buffer[-self.overlap_sentences:] if len(sentence_buffer) > self.overlap_sentences else sentence_buffer
                            overlap_text = " ".join(overlap_sentences)
                            
                            current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                            current_tokens = self.estimate_tokens(current_chunk)
                            sentence_buffer = overlap_sentences + [sentence]
                    
                    # 마지막 청크 저장
                    if current_chunk:
                        split_docs.append(Document(
                            page_content=current_chunk,
                            metadata={
                                'source': source,
                                'chunk_type': 'sentence_group',
                                'chunk_index': f"{para_idx}_{len(split_docs)}",
                                'token_count': current_tokens
                            }
                        ))
        
        return split_docs

class RAGPipeline:
    def __init__(self):
        self.google_api_key = st.secrets["GOOGLE_API_KEY"]
        
        # Google 무료 임베딩 모델 사용
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=self.google_api_key
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0,
            google_api_key=self.google_api_key
        )
        
        # 지능형 텍스트 분할기
        self.text_splitter = SmartTextSplitter(max_tokens=3000, overlap_sentences=2)

    def get_system_prompt_template(self, system_prompt):
        """시스템 프롬프트 템플릿"""
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
        """하이브리드 검색기 생성 (문단/문장 기반)"""
        if not documents:
            st.warning("문서에서 내용을 추출하지 못했습니다.")
            return None

        # 지능형 문서 분할
        splits = self.text_splitter.smart_split_documents(documents)
        
        if not splits:
            st.warning("문서 분할에 실패했습니다.")
            return None
        
        # 분할 결과 통계
        paragraph_chunks = [s for s in splits if s.metadata.get('chunk_type') == 'paragraph']
        sentence_chunks = [s for s in splits if s.metadata.get('chunk_type') == 'sentence_group']
        
        st.info(f"📊 지능형 분할 완료: {len(splits)}개 청크 (문단: {len(paragraph_chunks)}개, 문장그룹: {len(sentence_chunks)}개)")
        
        try:
            # 벡터 검색기 생성
            vectorstore = FAISS.from_documents(splits, self.embeddings)
            vector_retriever = vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={
                    "k": 12,  # 더 많은 문서 검색
                    "score_threshold": 0.1  # 낮은 임계값
                }
            )
            
            # BM25 검색기 생성
            bm25_retriever = BM25Retriever.from_documents(splits)
            bm25_retriever.k = 8
            
            # 하이브리드 검색기
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.4, 0.6]
            )
            
            st.success("🔍 하이브리드 검색기 (문단/문장 기반) 생성 완료")
            return ensemble_retriever
                
        except Exception as e:
            st.error(f"검색기 생성 오류: {e}")
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
