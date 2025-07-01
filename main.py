import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()


# --- 새로운 기능: 키워드 추출 체인 ---
def get_keyword_extraction_chain():
    """
    주어진 텍스트에서 키워드를 추출하는 LLM 체인을 생성합니다.
    """
    # 키워드 추출을 위한 프롬프트
    prompt = ChatPromptTemplate.from_template(
        """
        Analyze the following text and extract the 10 most important and relevant keywords or key phrases.
        The keywords should accurately represent the main topics of the text.
        Please provide the result as a single comma-separated string.

        Example:
        Text: "The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy..."
        Keywords: James Webb Space Telescope, infrared astronomy, space telescope, JWST, cosmic history, Big Bang, galaxy formation, exoplanets, NASA, deep space

        Text to analyze:
        {text}

        Keywords:
        """
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    chain = prompt | llm | StrOutputParser()
    return chain


# --- RAG 파이프라인 관련 함수 (수정됨) ---
@st.cache_resource(show_spinner="웹사이트를 분석하고 키워드를 추출 중입니다...")
def process_url_and_get_components(url):
    """
    URL로부터 문서를 로드, 분할하고 retriever와 원본 문서를 반환합니다.
    """
    # 1. 문서 로드
    loader = WebBaseLoader(url)
    documents = loader.load()

    # 2. 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    # 3. 임베딩 및 벡터 저장소 생성
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)

    # 4. Retriever와 분할된 문서(splits)를 함께 반환
    return vectorstore.as_retriever(search_kwargs={"k": 5}), splits


# --- 기존 체인 생성 함수들 ---
def get_conversational_rag_chain(retriever):
    rag_prompt = ChatPromptTemplate.from_template(
        """Answer the user's question based on the context provided below. If you don't know the answer, just say that you don't know.
        Context: {context}
        Question: {input}"""
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    return create_retrieval_chain(retriever, document_chain)


def get_default_chain():
    prompt = ChatPromptTemplate.from_messages(
        [("system", "당신은 친절한 AI 어시스턴트입니다."), ("user", "{question}")]
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    return prompt | llm | StrOutputParser()


# --- Streamlit 앱 UI 및 로직 ---
st.set_page_config(page_title="Keyword-Extracting RAG Chatbot", page_icon="🔑")
st.title("🔑 키워드 추출 RAG 챗봇")

# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "keywords" not in st.session_state:
    st.session_state.keywords = []

# --- 사이드바 UI ---
with st.sidebar:
    st.header("RAG 설정")
    url_input = st.text_input(
        "분석할 웹사이트 URL을 입력하세요", placeholder="https://example.com"
    )

    if st.button("사이트 분석 및 키워드 추출"):
        if url_input:
            # URL 처리 함수를 호출하여 retriever와 문서를 가져옴
            retriever, documents = process_url_and_get_components(url_input)
            st.session_state.retriever = retriever

            # 키워드 추출 로직
            if documents:
                # 모든 문서의 내용을 하나로 합침
                full_text = " ".join([doc.page_content for doc in documents])
                # 키워드 추출 체인 실행
                keyword_chain = get_keyword_extraction_chain()
                keyword_string = keyword_chain.invoke({"text": full_text})
                # 쉼표로 분리하고 공백 제거하여 리스트로 만듦
                st.session_state.keywords = [
                    keyword.strip() for keyword in keyword_string.split(",")
                ]

            st.success("분석 및 키워드 추출 완료!")
        else:
            st.warning("URL을 입력해주세요.")

    # 추출된 키워드가 있으면 사이드바에 표시
    if st.session_state.keywords:
        st.divider()
        st.subheader("추출된 핵심 키워드")
        # st.chip을 사용하여 키워드를 보기 좋게 표시
        for keyword in st.session_state.keywords:
            st.chip(keyword, icon="🔑")

    st.divider()
    if st.button("대화 초기화"):
        st.session_state.clear()
        st.rerun()

# --- 메인 채팅 화면 ---
# 이전 대화 기록 출력
if "messages" in st.session_state:
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    st.chat_message("user").write(user_input)

    # RAG 기능이 활성화되었는지 확인
    if st.session_state.retriever:
        chain = get_conversational_rag_chain(st.session_state.retriever)
        response = chain.stream({"input": user_input})

        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for chunk in response:
                if "answer" in chunk:
                    ai_answer += chunk["answer"]
                    container.markdown(ai_answer)
    else:
        # RAG 기능이 비활성화된 경우, 기본 체인 사용
        chain = get_default_chain()
        response = chain.stream({"question": user_input})

        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

    # 대화기록 저장
    st.session_state.messages.append(ChatMessage(role="user", content=user_input))
    st.session_state.messages.append(ChatMessage(role="assistant", content=ai_answer))
