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

    def get_system_prompt_template(self, system_prompt):
        """시스템 프롬프트 템플릿 - 외부 지식 활용 허용"""
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

        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        
        if not splits:
            st.warning("문서 분할에 실패했습니다.")
            return None
        
        st.info(f"📊 분할 완료: {len(splits)}개 청크 생성")
        
        try:
            # 1. 벡터 검색기 생성 (의미적 유사도)
            vectorstore = FAISS.from_documents(splits, self.embeddings)
            vector_retriever = vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={
                    "k": 10,  # 6 → 10으로 증가
                    "score_threshold": 0.15  # 0.3 → 0.15로 감소 (더 관대하게)
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
            st.error(f"검색기 생성 오류: {e}")
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
