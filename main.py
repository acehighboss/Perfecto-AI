import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import get_retriever_from_source, get_conversational_rag_chain, get_default_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# API KEY 정보로드
load_dotenv()

# --- 새로운 기능: 출처 재평가 및 필터링 함수 ---
def filter_relevant_sources(answer, source_documents):
    """
    LLM을 사용하여 생성된 답변과 직접적으로 관련된 소스 문서만 필터링합니다.
    """
    if not source_documents:
        return []

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    context_str = "\n---\n".join(
        [f"[Document {i+1}]: {doc.page_content}" for i, doc in enumerate(source_documents)]
    )

    prompt_template = """
    You are a helpful assistant. Your task is to identify which of the provided source documents are most relevant to the given answer.

    Here is the answer that was generated:
    ---
    {answer}
    ---

    Here are the source documents that were used as context:
    ---
    {context}
    ---

    Please list the numbers of the documents that directly support or contain the information presented in the answer. List the most relevant documents first. If no documents are relevant, respond with "None".

    Example: 3, 1, 8
    Relevant Document Numbers:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({"answer": answer, "context": context_str})
        if response and response.strip().lower() != 'none':
            indices_str = response.strip().split(',')
            relevant_indices = [int(i.strip()) - 1 for i in indices_str if i.strip().isdigit()]
            # 관련성 순서대로 정렬된 문서를 반환
            filtered_docs = [source_documents[i] for i in relevant_indices if 0 <= i < len(source_documents)]
            return filtered_docs
        return []
    except Exception as e:
        print(f"Error during source filtering: {e}")
        return []

# --- 페이지 설정 ---
st.set_page_config(page_title="Modular RAG Chatbot", page_icon="🤖")
st.title("🤖 모듈화된 RAG 챗봇")

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
            "파일 업로드 (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        if st.form_submit_button("분석 시작"):
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
                
                if st.session_state.retriever:
                    st.success("분석이 완료되었습니다! 이제 질문해보세요.")
                else:
                    st.error("분석에 실패했습니다. URL이나 파일 상태를 확인해주세요.")

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

    current_system_prompt = st.session_state.system_prompt

    try:
        if st.session_state.retriever:
            chain = get_conversational_rag_chain(st.session_state.retriever, current_system_prompt)
            
            with st.chat_message("assistant"):
                container = st.empty()
                ai_answer = ""
                source_documents = []
                
                with st.spinner("답변 생성 중..."):
                    for chunk in chain.stream({"input": user_input}):
                        if "answer" in chunk:
                            ai_answer += chunk["answer"]
                            container.markdown(ai_answer + "▌")
                        if "context" in chunk:
                            source_documents = chunk["context"]
                
                container.markdown(ai_answer)
                
                relevant_sources = []
                if source_documents:
                    with st.spinner("출처 확인 중..."):
                        relevant_sources = filter_relevant_sources(ai_answer, source_documents)
                
                if relevant_sources:
                    with st.expander("참고한 출처 보기"):
                        # 필터링된 출처를 최대 5개까지 표시
                        for i, source in enumerate(relevant_sources[:5]):
                            st.info(f"**출처 {i+1}**\n\n{source.page_content}")
                            st.divider()
                
                st.session_state.messages.append(
                    {"role": "assistant", "content": ai_answer, "sources": relevant_sources[:5]}
                )

        else:
            chain = get_default_chain(current_system_prompt)
            
            with st.chat_message("assistant"):
                container = st.empty()
                ai_answer = ""
                for token in chain.stream({"question": user_input}):
                    ai_answer += token
                    container.markdown(ai_answer + "▌")
                container.markdown(ai_answer)
            
            st.session_state.messages.append(
                {"role": "assistant", "content": ai_answer, "sources": []}
            )
    except Exception as e:
        with st.chat_message("assistant"):
            error_message = f"죄송합니다, 답변을 생성하는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요. (오류: {e})"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": []})
