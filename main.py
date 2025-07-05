import streamlit as st
import os
import tempfile
import asyncio # 비동기 처리를 위해 추가
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# [수정 1: LangChain Document 클래스 import]
from langchain_core.documents import Document as LangChainDocument
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from llama_parse import LlamaParse
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Multimodal RAG Chatbot", page_icon="🤖")
st.title("🤖 멀티모달 파일/URL 분석 RAG 챗봇")
st.markdown(
    """
안녕하세요! 이 챗봇은 웹사이트 URL이나 업로드된 파일(PDF, DOCX)의 내용을 분석하고 답변합니다.
**LlamaParse**를 사용하여 **테이블과 텍스트를 함께 인식**하고 질문에 답할 수 있습니다.
"""
)

def get_documents_from_files_with_llamaparse(uploaded_files):
    """
    업로드된 파일 리스트에서 LlamaParse를 사용하여 LlamaIndex 문서를 로드합니다.
    """
    all_documents = []
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        verbose=True,
    )

    # LlamaParse는 비동기 함수를 사용하므로 이벤트 루프를 통해 실행합니다.
    async def parse_files(files):
        parsed_data = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # aload_data는 비동기 함수이므로 await 사용
                documents = await parser.aload_data(tmp_file_path)
                parsed_data.extend(documents)
            except Exception as e:
                st.error(f"LlamaParse 처리 중 오류 발생: {e}")
            finally:
                os.remove(tmp_file_path)
        return parsed_data

    # Streamlit 환경에서 asyncio 실행
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    all_documents = loop.run_until_complete(parse_files(uploaded_files))
    return all_documents


@st.cache_resource(show_spinner="LlamaParse로 문서를 분석 중입니다...")
def get_retriever_from_source(source_type, source_input, threshold):
    """
    URL 또는 파일로부터 문서를 로드하고, 텍스트를 분할하여 retriever를 생성합니다.
    """
    documents = [] # LangChain 문서를 담을 최종 리스트
    
    if source_type == "URL":
        loader = WebBaseLoader(source_input)
        documents = loader.load()
    elif source_type == "Files":
        # LlamaParse는 LlamaIndex 형식의 문서를 반환합니다.
        llama_index_documents = get_documents_from_files_with_llamaparse(source_input)

        if llama_index_documents:
            # [수정 2: LlamaIndex 문서를 LangChain 문서로 변환]
            # 이 부분이 오류 해결의 핵심입니다.
            langchain_documents = [
                LangChainDocument(page_content=doc.text, metadata=doc.metadata)
                for doc in llama_index_documents
            ]
            documents.extend(langchain_documents)

    if not documents:
        st.warning("문서에서 내용을 추출하지 못했습니다.")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # 이제 SemanticChunker는 page_content 속성을 가진 LangChain 문서를 받게 됩니다.
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    splits = text_splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": threshold},
    )


def get_conversational_rag_chain(retriever, system_prompt):
    template = f"""{system_prompt}

Answer the user's question based on the context provided below and the conversation history.
The context may include text and tables in markdown format. You must be able to understand and answer based on them.
If you don't know the answer, just say that you don't know. Don't make up an answer.

Context:
{{context}}
"""
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


def get_default_chain(system_prompt):
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
    st.session_state.system_prompt = "당신은 문서 분석 전문가 AI 어시스턴트입니다. 주어진 문서의 텍스트와 테이블을 정확히 이해하고 상세하게 답변해주세요."


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
    st.info("LlamaParse는 테이블, 텍스트가 포함된 문서 분석에 최적화되어 있습니다.", icon="ℹ️")

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
            with st.expander("참고한 출처 보기 (마크다운 형식)"):
                for i, source in enumerate(message["sources"]):
                    st.text(f"--- 출처 {i+1} ---")
                    st.markdown(source.page_content)


user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

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
                with st.expander("참고한 출처 보기 (마크다운 형식)"):
                    for i, source in enumerate(source_documents):
                        st.text(f"--- 출처 {i+1} ---")
                        st.markdown(source.page_content)

    else:
        chain = get_default_chain(st.session_state.system_prompt)

        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for token in chain.stream(
                {"question": user_input, "chat_history": chat_history}
            ):
                ai_answer += token
                container.markdown(ai_answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": ai_answer, "sources": []}
        )
