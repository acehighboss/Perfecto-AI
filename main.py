# main.py
import streamlit as st
# API 키는 Streamlit의 secrets에서 자동으로 로드됩니다. .env 파일이나 load_dotenv()는 필요 없습니다.
from rag_pipeline import get_retriever_from_source, get_conversational_rag_chain, get_default_chain

# --- 페이지 설정 ---
st.set_page_config(page_title="LlamaParse RAG Chatbot", page_icon="🦙")
st.title("🦙 LlamaParse & Rerank RAG 챗봇")
st.markdown(
    """
    안녕하세요! 이 챗봇은 **LlamaParse**로 문서를 분석하고, **Cohere Rerank**로 답변의 정확도를 높였습니다.
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
    st.info("LLAMA_CLOUD_API_KEY, GOOGLE_API_KEY, COHERE_API_KEY를 Streamlit secrets에 설정해야 합니다.")
    st.divider()
    
    with st.form("persona_form"):
        st.subheader("🤖 AI 페르소나 설정")
        system_prompt_input = st.text_area(
            "AI의 역할을 설정해주세요.",
            value=st.session_state.system_prompt,
            height=150
        )
        if st.form_submit_button("페르소나 적용"):
            st.session_state.system_prompt = system_prompt_input
            st.success("페르소나가 적용되었습니다!")

    st.divider()
    
    with st.form("source_form"):
        st.subheader("🔎 분석 대상 설정")
        url_input = st.text_input("웹사이트 URL", placeholder="https://example.com")
        uploaded_files = st.file_uploader(
            "파일 업로드 (PDF, DOCX 등)",
            accept_multiple_files=True
        )

        if st.form_submit_button("분석 시작"):
            source_type = "Files" if uploaded_files else "URL" if url_input else None
            source_input = uploaded_files or url_input

            if source_type:
                with st.spinner("LlamaParse로 문서를 분석하고 Rerank 모델을 준비 중입니다..."):
                    st.session_state.retriever = get_retriever_from_source(source_type, source_input)
                
                if st.session_state.retriever:
                    st.success("분석이 완료되었습니다! 이제 질문해보세요.")
                else:
                    st.error("분석에 실패했습니다. API 키나 파일/URL 상태를 확인해주세요.")
            else:
                st.warning("분석할 URL을 입력하거나 파일을 업로드해주세요.")

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
                    # Reranker가 추가한 관련성 점수(relevance_score)를 함께 표시
                    relevance_score = source.metadata.get('relevance_score', 'N/A')
                    st.info(f"**출처 {i+1}** (관련성: {relevance_score:.2f})\n\n{source.page_content}")
                    st.divider()

# 사용자의 입력
if user_input := st.chat_input("궁금한 내용을 물어보세요!"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    current_system_prompt = st.session_state.system_prompt

    try:
        with st.chat_message("assistant"):
            if st.session_state.retriever:
                chain = get_conversational_rag_chain(st.session_state.retriever, current_system_prompt)
                
                # Reranker 사용 시 스트리밍 대신 invoke 사용
                response = chain.invoke({"input": user_input})
                ai_answer = response.get("answer", "답변을 생성하지 못했습니다.")
                source_documents = response.get("context", [])
                
                st.markdown(ai_answer)
                
                # 답변과 함께 출처를 표시
                if source_documents:
                    with st.expander("참고한 출처 보기"):
                        for i, source in enumerate(source_documents):
                            relevance_score = source.metadata.get('relevance_score', 'N/A')
                            st.info(f"**출처 {i+1}** (관련성: {relevance_score:.2f})\n\n{source.page_content}")
                            st.divider()
                
                st.session_state.messages.append(
                    {"role": "assistant", "content": ai_answer, "sources": source_documents}
                )

            else: # RAG 기능이 비활성화된 경우
                chain = get_default_chain(current_system_prompt)
                ai_answer = st.write_stream(chain.stream({"question": user_input}))
                st.session_state.messages.append(
                    {"role": "assistant", "content": ai_answer, "sources": []}
                )
    except Exception as e:
        error_message = f"죄송합니다, 답변을 생성하는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요. (오류: {e})"
        st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": []})
