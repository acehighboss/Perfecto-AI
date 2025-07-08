# 메인 애플리케이션
# Streamlit UI를 그리고, 사용자 입력을 받아 다른 모듈의 함수를 호출하여 챗봇의 전체 흐름을 제어합니다.

import streamlit as st
import subprocess
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from rag_pipeline import get_retriever_from_source, get_document_chain, get_default_chain

# --- [수정] Playwright 브라우저 자동 설치 및 디버깅 로직 ---
# 세션 상태를 사용하여 앱 세션당 한 번만 설치를 시도합니다.
if "playwright_installed" not in st.session_state:
    st.set_page_config(page_title="Initial Setup", layout="wide")
    st.title("🛠️ 초기 설정: Playwright 브라우저 설치")
    st.write("챗봇을 실행하기 전에 필요한 Playwright 브라우저를 설치합니다. 이 과정은 처음 한 번만 실행되며, 몇 분 정도 소요될 수 있습니다.")

    with st.spinner("설치 명령을 실행 중입니다..."):
        # subprocess.run을 사용하여 'playwright install' 명령을 실행하고 결과를 캡처합니다.
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install"],
            capture_output=True,
            text=True,
            encoding='utf-8'  # 인코딩 명시
        )
    
    # 설치 과정의 표준 출력(stdout)과 표준 에러(stderr)를 화면에 표시합니다.
    st.subheader("설치 로그")
    st.code(f"Return Code: {result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")

    if result.returncode == 0:
        st.success("브라우저 설치가 성공적으로 완료되었습니다! 앱을 자동으로 다시 시작합니다.")
        st.session_state["playwright_installed"] = True
        # 성공 후 잠시 딜레이를 주어 메시지를 읽을 시간을 줍니다.
        import time
        time.sleep(3)
        st.rerun() # 앱을 새로고침하여 원래의 챗봇 화면을 로드합니다.
    else:
        # STDERR에 'sudo' 관련 메시지가 있어도 STDOUT에 다운로드 성공 메시지가 있으면 성공으로 간주
        if "downloaded" in result.stdout.lower():
             st.success("브라우저 다운로드가 확인되었습니다. 5초 후 앱을 자동으로 다시 시작합니다.")
             st.session_state["playwright_installed"] = True
             time.sleep(5)
             st.rerun()
        st.error("Playwright 브라우저 설치에 실패했습니다. 위의 로그를 확인하여 원인을 파악해주세요.")
        st.stop() # 설치 실패 시 앱 실행을 중단합니다.

# --- (이후 코드는 브라우저 설치가 성공해야만 실행됩니다) ---

# API 키 로드
load_dotenv()

# --- 앱 기본 설정 ---
st.set_page_config(page_title="Multimodal RAG Chatbot", page_icon="🤖")
st.title("🤖 멀티모달 파일/URL 분석 RAG 챗봇")
st.markdown(
    """
안녕하세요! 이 챗봇은 웹사이트 URL이나 업로드된 파일(PDF, DOCX)의 내용을 분석하고 답변합니다.
**LlamaParse**를 사용하여 **테이블과 텍스트를 함께 인식**하고 질문에 답할 수 있습니다.
"""
)

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
    st.info(
        "LlamaParse는 테이블, 텍스트가 포함된 문서 분석에 최적화되어 있습니다.",
        icon="ℹ️",
    )

    if st.button("분석 시작"):
        st.session_state.messages = []
        st.session_state.retriever = None

        source_type = None
        source_input = None
        if uploaded_files:
            source_type = "Files"
            source_input = uploaded_files
        elif url_input:
            source_type = "URL"
            source_input = url_input
        else:
            st.warning("분석할 URL을 입력하거나 파일을 업로드해주세요.")

        if source_input:
            st.session_state.retriever = get_retriever_from_source(
                source_type, source_input
            )
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

    try:
        chat_history = [
            (
                HumanMessage(content=msg["content"])
                if msg["role"] == "user"
                else AIMessage(content=msg["content"])
            )
            for msg in st.session_state.messages[:-1]
        ]

        # [수정] RAG 체인 호출 로직 변경
        if st.session_state.retriever:
            with st.chat_message("assistant"):
                with st.spinner("관련 문서를 검색하고 답변을 생성 중입니다..."):
                    # 1. Retriever를 직접 호출하여 출처 문서를 먼저 가져옵니다.
                    retriever = st.session_state.retriever
                    source_documents = retriever.get_relevant_documents(
                        user_input,
                    )
                    
                    # 2. 답변 생성 체인을 가져옵니다.
                    document_chain = get_document_chain(st.session_state.system_prompt)
                    
                    # 3. 직접 가져온 출처와 사용자 질문으로 답변을 생성합니다.
                    ai_answer = document_chain.invoke({
                        "input": user_input,
                        "chat_history": chat_history,
                        "context": source_documents
                    })
                    
                    # 4. 결과 표시 및 저장
                    st.markdown(ai_answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": ai_answer,
                            "sources": source_documents,
                        }
                    )

                    if source_documents:
                        with st.expander("참고한 출처 보기"):
                            for i, source in enumerate(source_documents):
                                st.info(f"**출처 {i+1}**\n\n{source.page_content}")
                                st.divider()

        else: # RAG가 아닌 일반 대
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

    except Exception as e:
        st.chat_message("assistant").error(
            f"죄송합니다, 답변을 생성하는 중 오류가 발생했습니다.\n\n오류: {e}"
        )
        st.session_state.messages.pop()
