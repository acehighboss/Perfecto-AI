import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from file_handler import get_vector_store
from rag_pipeline import get_conversational_rag_chain, get_default_chain

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 페이지 설정 ---
st.set_page_config(page_title="Upstage RAG Chatbot", page_icon="🚀")
st.title("🚀 문서/URL 분석 RAG 챗봇")
st.markdown(
    """
안녕하세요! 이 챗봇은 웹사이트 URL이나 업로드된 파일(PDF, DOCX 등)의 내용을 분석하고 답변합니다.
**LlamaParse**를 사용하여 **이미지, 테이블, 텍스트를 함께 인식**하고 질문에 답할 수 있습니다.
"""
)

# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
# [수정] retriever 대신 vector_store를 세션에 저장합니다.
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "당신은 문서 분석 전문가 AI 어시스턴트입니다. 주어진 문서의 텍스트와 테이블을 정확히 이해하고 상세하게 답변해주세요."

# --- 사이드바 UI ---
with st.sidebar:
    st.header("⚙️ 설정")
    st.divider()

    st.subheader("🤖 AI 페르소나 설정")
    system_prompt_input = st.text_area(
        "AI의 역할을 설정해주세요.",
        value=st.session_state.system_prompt,
        height=150,
        key="system_prompt_input_area"
    )
    if st.button("페르소나 적용"):
        st.session_state.system_prompt = system_prompt_input
        st.success("페르소나가 적용되었습니다!")
    
    st.divider()
    st.subheader("🔎 분석 대상 설정")
    url_input = st.text_input("웹사이트 URL", placeholder="https://example.com")
    uploaded_files = st.file_uploader(
        "파일 업로드 (PDF, DOCX 등)", type=["pdf", "docx", "md", "txt"], accept_multiple_files=True
    )
    st.info("LlamaParse는 이미지, 테이블, 텍스트가 포함된 문서 분석에 최적화되어 있습니다.", icon="ℹ️")
    
    if st.button("분석 시작"):
        source_input = None
        
        if uploaded_files:
            source_type = "Files"
            source_input = uploaded_files
        elif url_input:
            source_type = "URL"
            source_input = url_input
        else:
            st.warning("분석할 URL을 입력하거나 파일을 업로드해주세요.")
            st.stop()
        
        st.session_state.messages = []
        st.session_state.vector_store = None

        vector_store = get_vector_store(source_input, source_type)
        if vector_store:
            # [수정] vector_store 자체를 세션 상태에 저장합니다.
            st.session_state.vector_store = vector_store
            st.success("분석이 완료되었습니다! 이제 질문해보세요.")
        else:
            st.error("벡터 저장소 생성에 실패했습니다. 파일을 다시 확인해주세요.")

    st.divider()
    if st.button("대화 초기화"):
        st.session_state.clear()
        st.rerun()

# --- 메인 채팅 인터페이스 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("참고한 출처 보기"):
                for i, source in enumerate(message["sources"]):
                    source_info = f"출처 {i+1} (Source: {source.metadata.get('source', 'N/A')}, Page: {source.metadata.get('page', 'N/A')})"
                    st.markdown(f"**{source_info}**")
                    st.markdown(source.page_content)

user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    chat_history = [
        HumanMessage(content=msg["content"]) if msg["role"] == "user" 
        else AIMessage(content=msg["content"])
        for msg in st.session_state.messages[:-1]
    ]

    try:
        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            source_documents = []

            # [수정] retriever 대신 vector_store의 존재 여부를 확인합니다.
            if st.session_state.vector_store:
                # [수정] 체인 생성 시 retriever가 아닌 vector_store를 전달합니다.
                chain = get_conversational_rag_chain(
                    st.session_state.vector_store, st.session_state.system_prompt
                )
                for chunk in chain.stream({"input": user_input, "chat_history": chat_history}):
                    if "answer" in chunk:
                        ai_answer += chunk["answer"]
                        container.markdown(ai_answer)
                    if "context" in chunk and not source_documents:
                        source_documents = chunk["context"]
            else:
                chain = get_default_chain(st.session_state.system_prompt)
                for token in chain.stream({"question": user_input, "chat_history": chat_history}):
                    ai_answer += token
                    container.markdown(ai_answer)
            
            st.session_state.messages.append(
                {"role": "assistant", "content": ai_answer, "sources": source_documents}
            )

            if source_documents:
                with st.expander("참고한 출처 보기"):
                    for i, source in enumerate(source_documents):
                        source_info = f"출처 {i+1} (Source: {source.metadata.get('source', 'N/A')}, Page: {source.metadata.get('page', 'N/A')})"
                        st.markdown(f"**{source_info}**")
                        st.markdown(source.page_content)

    except Exception as e:
        st.error(f"죄송합니다, 답변을 생성하는 중 오류가 발생했습니다.\n\n오류: {e}")
