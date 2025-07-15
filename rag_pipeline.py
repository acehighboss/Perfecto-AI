import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

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

    def get_system_prompt_template(self, system_prompt):
        """시스템 프롬프트 템플릿 - 범용 처리"""
        template = system_prompt + """

다음 컨텍스트를 바탕으로 사용자의 질문에 정확하게 답변해주세요.
컨텍스트에는 분석된 문서의 내용이 포함되어 있습니다.

**중요한 지침:**
1. 제공된 컨텍스트의 정보를 반드시 활용하여 답변하세요
2. 컨텍스트에 관련 정보가 있다면 구체적이고 상세한 답변을 제공하세요
3. 답변할 때는 참조한 출처를 명시해주세요
4. 컨텍스트에 없는 정보는 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 명시하세요
5. 모든 주제와 질문에 대해 동일한 수준의 정확성을 유지하세요

컨텍스트:
{context}
"""
        return template

    def create_retriever(self, documents):
        """문서에서 검색기 생성 - 범용 최적화"""
        if not documents:
            st.warning("문서에서 내용을 추출하지 못했습니다.")
            return None

        # 기본 분할기 사용 (빠른 처리)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
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

        # 검색기 설정 - 범용 최적화
        base_retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 6}  # 적절한 개수로 조정
        )
        
        # 압축 없이 기본 검색기만 사용 (빠른 처리)
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
