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

    def get_system_prompt_template(self, system_prompt):
        """시스템 프롬프트 템플릿"""
        template = system_prompt + """

다음 컨텍스트를 바탕으로 사용자의 질문에 정확하게 답변해주세요.

**답변 지침:**
1. 제공된 컨텍스트의 정보만을 사용하여 답변하세요
2. 답변할 때는 반드시 참조한 출처를 명시해주세요
3. 컨텍스트에 없는 정보는 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 명시하세요
4. 정확한 정보만을 제공하고, 추측이나 가정은 피하세요
5. 가능한 한 구체적이고 상세한 답변을 제공하세요

컨텍스트:
{context}
"""
        return template

    def create_retriever(self, documents):
        """문서에서 검색기 생성"""
        if not documents:
            st.warning("문서에서 내용을 추출하지 못했습니다.")
            return None

        # 텍스트 분할 (토큰 제한 고려)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
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

        # 검색기 설정
        base_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 6,
                "score_threshold": 0.3
            }
        )
        
        # 압축 검색기 사용 (관련성 높은 정보만 추출)
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return compression_retriever

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
