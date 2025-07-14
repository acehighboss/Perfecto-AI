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

    def create_retriever(self, documents):
        """문서에서 검색기 생성"""
        if not documents:
            st.warning("문서에서 내용을 추출하지 못했습니다.")
            return None

        # RecursiveCharacterTextSplitter 사용
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
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

        # 검색기 설정
        base_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 8,
                "score_threshold": 0.3
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
        template = f"""{system_prompt}

다음 컨텍스트를 바탕으로 사용자의 질문에 답변해주세요.
컨텍스트에는 텍스트, 테이블, 이미지 내용이 마크다운 형식으로 포함되어 있을 수 있습니다.

중요한 지침:
1. 답변할 때는 반드시 참조한 출처를 명시해주세요.
2. 테이블이나 구조화된 데이터는 정확히 해석해주세요.
3. 확실하지 않은 정보는 "확실하지 않음"이라고 명시해주세요.
4. 질문과 관련된 키워드를 컨텍스트에서 찾아 정확한 답변을 제공해주세요.

컨텍스트:
{{context}}
"""
        
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", template),
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

    def stream_response(self, chain, user_input, chat_history):
        """스트리밍 응답 생성"""
        ai_answer = ""
        source_documents = []
        
        for chunk in chain.stream({
            "input": user_input, 
            "chat_history": chat_history
        }):
            if "answer" in chunk:
                ai_answer += chunk["answer"]
                yield chunk["answer"]
            if "context" in chunk and not source_documents:
                source_documents = chunk["context"]
        
        return ai_answer, source_documents

    def stream_default_response(self, chain, user_input, chat_history):
        """기본 체인 스트리밍 응답"""
        ai_answer = ""
        
        for token in chain.stream({
            "question": user_input, 
            "chat_history": chat_history
        }):
            ai_answer += token
            yield token
        
        return ai_answer
