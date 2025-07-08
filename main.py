# 메인 애플리케이션
# Streamlit UI를 그리고, 사용자 입력을 받아 다른 모듈의 함수를 호출하여 챗봇의 전체 흐름을 제어합니다.

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from rag_pipeline import get_retriever_from_source, get_conversational_rag_chain, get_default_chain

# API 키 로드
load_dotenv()

# --- 앱 기본 설정 ---
st.set_page_config(page_title="Multimodal RAG Chatbot", page_icon="🤖")
st.title("🤖 멀티모달 파일/URL 분석 RAG 챗봇")
st.markdown(
    """
안녕하세요! 이 챗봇은 웹사이트 URL이나 업로드된 파일(PDF, DOCX, TXT)의 내용을 분석하고 답변합니다.
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
        "파일 업로드 (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )
    st.info("LlamaParse는 테이블, 텍스트가 포함된 문서 분석에 최적화되어 있습니다.", icon="ℹ️")
    
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
            st.session_state.retriever = get_retriever_from_source(source_type, source_input)
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

    try:
        chat_history = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" 
            else AIMessage(content=msg["content"])
            for msg in st.session_state.messages[:-1]
        ]
        
        if st.session_state.retriever:
            chain = get_conversational_rag_chain(st.session_state.retriever, st.session_state.system_prompt)
            with st.chat_message("assistant"):
                with st.spinner("답변을 생성하고 출처를 확인하는 중입니다..."):
                    full_response = chain.invoke({"input": user_input, "chat_history": chat_history})
                    ai_answer = full_response.get('answer', '답변을 가져오는 데 실패했습니다.')
                    source_documents = full_response.get('context', [])
                    st.markdown(ai_answer)
                    
                    st.session_state.messages.append({"role": "assistant", "content": ai_answer, "sources": source_documents})
                    
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
                for token in chain.stream({"question": user_input, "chat_history": chat_history}):
                    ai_answer += token
                    container.markdown(ai_answer)
                st.session_state.messages.append({"role": "assistant", "content": ai_answer, "sources": []})

    except Exception as e:
        st.chat_message("assistant").error(f"죄송합니다, 답변을 생성하는 중 오류가 발생했습니다.\n\n오류: {e}")
        st.session_state.messages.pop()
