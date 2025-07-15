import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import get_retriever_from_source, get_conversational_rag_chain, get_default_chain

# API KEY 정보로드
load_dotenv()

# --- 페이지 설정 ---
st.set_page_config(page_title="Modular RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG 챗봇")
st.markdown(
    """
안녕하세요! 이 챗봇은 웹사이트 URL이나 업로드된 파일(PDF, DOCX, TXT)의 내용을 분석하고 답변합니다.
왼쪽 사이드바에서 AI의 페르소나와 분석할 대상을 설정하고 '적용' 또는 '분석' 버튼을 눌러주세요.
"""
)

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
    # 페르소나 입력을 위한 text_area
    system_prompt_input = st.text_area(
        "AI의 역할을 설정해주세요.",
        value=st.session_state.system_prompt,
        height=150,
        key="persona_input"
    )
    # 페르소나 적용 버튼
    if st.button("페르소나 적용"):
        st.session_state.system_prompt = system_prompt_input
        st.success("페르소나가 적용되었습니다!")

    st.divider()
    st.subheader("🔎 분석 대상 설정")
    
    url_input = st.text_input("웹사이트 URL", placeholder="https://example.com")
    
    uploaded_files = st.file_uploader(
        "파일 업로드 (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if st.button("분석 시작"):
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

        if source_type:
            with st.spinner("분석 중입니다..."):
                st.session_state.retriever = get_retriever_from_source(source_type, source_input)
            st.success("분석이 완료되었습니다! 이제 질문해보세요.")

    st.divider()
    if st.button("대화 초기화"):
        st.session_state.clear()
        st.rerun()

# --- 메인 채팅 화면 ---
# 이전 대화 기록 출력
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("참고한 출처 보기"):
                for i, source in enumerate(message["sources"]):
                    st.info(f"**출처 {i+1}**\n\n{source.page_content}")
                    st.divider()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    if st.session_state.retriever:
        chain = get_conversational_rag_chain(st.session_state.retriever, st.session_state.system_prompt)
        
        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            source_documents = []
            
            for chunk in chain.stream({"input": user_input}):
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
            for token in chain.stream({"question": user_input}):
                ai_answer += token
                container.markdown(ai_answer)
        
        st.session_state.messages.append(
            {"role": "assistant", "content": ai_answer, "sources": []}
        )
