import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

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
        """시스템 프롬프트 템플릿 - context 변수 수정"""
        # f-string 사용하지 않고 직접 문자열 연결
        template = system_prompt + """

다음 컨텍스트를 바탕으로 사용자의 질문에 정확하게 답변해주세요.
컨텍스트에는 분석된 문서의 내용이 포함되어 있습니다.

**중요한 지침:**
1. 제공된 컨텍스트의 정보를 반드시 활용하여 답변하세요
2. 컨텍스트에 관련 정보가 있다면 "제공되지 않았습니다"라고 말하지 마세요
3. 컨텍스트의 내용을 바탕으로 구체적이고 상세한 답변을 제공하세요
4. 답변할 때는 참조한 출처를 명시해주세요

컨텍스트:
{context}
"""
        return template

    def create_retriever(self, documents):
        """문서에서 검색기 생성"""
        if not documents:
            st.warning("문서에서 내용을 추출하지 못했습니다.")
            return None

        # SemanticChunker 사용 (Google 임베딩 모델과 함께)
        try:
            text_splitter = SemanticChunker(
                self.embeddings, 
                breakpoint_threshold_type="percentile"
            )
            
            splits = text_splitter.split_documents(documents)
        except Exception as e:
            st.warning(f"SemanticChunker 실패, 기본 분할 사용: {e}")
            # 기본 분할기로 대체
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            splits = text_splitter.split_documents(documents)
        
        if not splits:
            st.warning("문서 분할에 실패했습니다.")
            return None
        
        st.info(f"📊 분할 완료: {len(splits)}개 청크 생성")
        
        # FAISS 벡터스토어 생성
        try:
            vectorstore = FAISS.from_documents(splits, self.embeddings)
        except Exception as e:
            st.error(f"벡터스토어 생성 오류: {e}")
            return None

        # 검색기 설정 - 더 많은 문서 검색
        base_retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 8}  # 더 많은 관련 문서 검색
        )
        
        # 압축 검색기 사용하지 않고 기본 검색기만 사용 (안정성 우선)
        return base_retriever

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
