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
        
        # Google 임베딩 모델
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
        """시스템 프롬프트 템플릿"""
        template = system_prompt + """

다음 컨텍스트를 바탕으로 사용자의 질문에 정확하게 답변해주세요.

**중요한 답변 규칙:**
1. 반드시 제공된 컨텍스트의 정보만을 사용하여 답변하세요
2. 컨텍스트에서 관련 정보를 찾을 수 없다면 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 명시하세요
3. 절대로 일반 지식이나 외부 정보를 사용하지 마세요
4. 답변할 때는 반드시 참조한 출처를 명시해주세요
5. 추측이나 가정은 하지 마세요

컨텍스트:
{context}
"""
        return template

    def create_retriever(self, documents):
        """문서에서 검색기 생성 - 검색 성능 개선"""
        if not documents:
            st.warning("문서에서 내용을 추출하지 못했습니다.")
            return None

        # 더 작은 청크로 분할 (검색 정확도 향상)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 1000 → 500으로 감소
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
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

        # 검색 설정 대폭 완화
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",  # 임계값 제거
            search_kwargs={
                "k": 15,  # 6 → 15로 증가 (더 많은 문서 검색)
            }
        )
        
        return base_retriever  # 압축 검색기 제거 (정보 손실 방지)

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
            ("system", system_prompt + "\n\n중요: 문서가 제공되지 않았으므로 '문서를 먼저 업로드해주세요'라고 안내하세요."),
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

    def debug_search_results(self, retriever, query):
        """검색 결과 디버깅"""
        try:
            docs = retriever.get_relevant_documents(query)
            st.write(f"🔍 검색된 문서 수: {len(docs)}")
            
            for i, doc in enumerate(docs[:3]):  # 상위 3개만 표시
                st.write(f"**문서 {i+1}:**")
                st.write(doc.page_content[:200] + "...")
                st.write("---")
                
        except Exception as e:
            st.error(f"검색 디버깅 오류: {e}")
