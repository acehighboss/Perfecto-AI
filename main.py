import streamlit as st
import os
import tempfile
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

st.set_page_config(page_title="File/URL RAG Chatbot", page_icon="🤖")
st.title("🤖 파일/URL 분석 RAG 챗봇")
st.markdown(
    """
안녕하세요! 이 챗봇은 웹사이트 URL이나 업로드된 파일(PDF, DOCX)의 내용을 분석하고 답변합니다.
왼쪽 사이드바에서 AI의 페르소나와 분석할 대상을 설정해주세요.
"""
)

# --- 문서 로딩 및 처리 관련 함수 ---


def get_documents_from_files(uploaded_files):
    """
    업로드된 파일 리스트에서 문서를 로드합니다.
    """
    all_documents = []
    for uploaded_file in uploaded_files:
        # 임시 파일로 저장하여 경로를 얻음
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = None
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_file_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_file_path)
        # 추가적인 파일 형식 로더를 여기에 추가할 수 있습니다.

        if loader:
            all_documents.extend(loader.load())

        # 임시 파일 삭제
        os.remove(tmp_file_path)

    return all_documents


@st.cache_resource(show_spinner="분석 중입니다...")
def get_retriever_from_source(source_type, source_input, threshold):
    """
    URL 또는 파일로부터 문서를 로드하고, 텍스트를 분할하여 retriever를 생성합니다.
    """
    documents = []
    if source_type == "URL":
        loader = WebBaseLoader(source_input)
        documents = loader.load()
    elif source_type == "Files":
        documents = get_documents_from_files(source_input)

    if not documents:
        return None

    # 임베딩 모델 정의
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # SemanticChunker를 사용하여 의미 기반으로 텍스트 분할
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    splits = text_splitter.split_documents(documents)

    # FAISS 벡터 저장소 생성 및 retriever 반환
    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": threshold},
    )


# --- LangChain 체인 생성 함수 ---


# get_conversational_rag_chain: 대화 기록을 처리하도록 프롬프트 수정
def get_conversational_rag_chain(retriever, system_prompt):
    """
    RAG 체인을 생성합니다. 대화 기록(chat_history)을 컨텍스트에 포함합니다.
    """
    # 시스템 프롬프트에 대화 기록을 활용하라는 내용을 추가할 수 있습니다.
    template = f"""{system_prompt}

Answer the user's question based on the context provided below and the conversation history.
If you don't know the answer, just say that you don't know. Don't make up an answer.

Context:
{{context}}
"""
    # MessagesPlaceholder를 사용하여 대화 기록을 프롬프트에 삽입합니다.
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    return create_retrieval_chain(retriever, document_chain)


# get_default_chain: 대화 기록을 처리하도록 프롬프트 수정
def get_default_chain(system_prompt):
    """
    기본 대화 체인을 생성합니다. 대화 기록(chat_history)을 컨텍스트에 포함합니다.
    """
    # MessagesPlaceholder를 사용하여 대화 기록을 프롬프트에 삽입합니다.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}"),
        ]
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    return prompt | llm | StrOutputParser()


# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "당신은 친절한 AI 어시스턴트입니다. 사용자의 질문에 항상 친절하고 상세하게 답변해주세요."


# --- 사이드바 UI ---
with st.sidebar:
    st.header("⚙️ 설정")
    st.divider()

    st.subheader("🤖 AI 페르소나 설정")
    system_prompt_input = st.text_area(
        "AI의 역할을 설정해주세요.", value=st.session_state.system_prompt, height=150
    )
    st.session_state.system_prompt = system_prompt_input

    st.divider()
    st.subheader("🔎 분석 대상 설정")

    url_input = st.text_input("웹사이트 URL", placeholder="https://example.com")
    uploaded_files = st.file_uploader(
        "파일 업로드 (PDF, DOCX)", type=["pdf", "docx"], accept_multiple_files=True
    )
    st.info("이미지 파일 분석은 현재 지원되지 않습니다.", icon="ℹ️")

    st.subheader("📊 검색 정확도 설정")
    similarity_threshold = st.slider(
        "유사도 임계값 (값이 낮을수록 정확함)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="문서 검색 시, 설정된 값보다 낮은 거리(distance)의 문서만 가져옵니다. 0에 가까울수록 질문과 유사한 내용만 필터링합니다.",
    )

    if st.button("분석 시작"):
        source_type = None
        source_input = None
        if uploaded_files:
            source_type = "Files"
            source_input = uploaded_files
            st.session_state.retriever = get_retriever_from_source(
                source_type, source_input, similarity_threshold
            )
        elif url_input:
            source_type = "URL"
            source_input = url_input
            st.session_state.retriever = get_retriever_from_source(
                source_type, source_input, similarity_threshold
            )
        else:
            st.warning("분석할 URL을 입력하거나 파일을 업로드해주세요.")

        if st.session_state.retriever:
            st.success("분석이 완료되었습니다! 이제 질문해보세요.")

    st.divider()
    if st.button("대화 초기화"):
        st.session_state.clear()
        st.rerun()

# --- 메인 채팅 화면 ---
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("참고한 출처 보기"):
                for i, source in enumerate(message["sources"]):
                    st.info(f"**출처 {i+1}**\n\n{source.page_content}")
                    st.divider()


user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # 체인에 전달할 대화 기록 생성
    # 현재 입력을 제외한 이전 대화 기록을 LangChain이 이해하는 형식으로 변환합니다.
    chat_history = [
        ChatMessage(role=msg["role"], content=msg["content"])
        for msg in st.session_state.messages[:-1]
    ]

    if st.session_state.retriever:
        chain = get_conversational_rag_chain(
            st.session_state.retriever, st.session_state.system_prompt
        )

        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            source_documents = []

            # RAG 체인 호출 시 chat_history 전달
            for chunk in chain.stream(
                {"input": user_input, "chat_history": chat_history}
            ):
                if "answer" in chunk:
                    ai_answer += chunk["answer"]
                    container.markdown(ai_answer)
                if "context" in chunk and not source_documents:
                    source_documents = chunk["context"]

            st.session_state.messages.append(
                {"role": "assistant", "content": ai_answer, "sources": source_documents}
            )

            if source_documents:
                with st.expander("참고한 출처 보기"):
                    for i, source in enumerate(source_documents):
                        st.info(f"**출처 {i+1}**\n\n{source.page_content}")
                        st.divider()

    else:
        chain = get_default_chain(st.session_state.system_prompt)

        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            # 기본 체인 호출 시 chat_history 전달
            for token in chain.stream(
                {"question": user_input, "chat_history": chat_history}
            ):
                ai_answer += token
                container.markdown(ai_answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": ai_answer, "sources": []}
        )
