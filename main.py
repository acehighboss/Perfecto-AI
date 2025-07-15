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
    
    # 모든 문서를 번호가 매겨진 단일 문자열로 결합
    context_str = "\n---\n".join(
        [f"[Document {i+1}]: {doc.page_content}" for i, doc in enumerate(source_documents)]
    )

    # LLM에게 관련 문서를 식별하도록 요청하는 프롬프트
    prompt_template = """
    You are a helpful assistant. Your task is to identify which of the provided source documents are relevant to the given answer.

    Here is the answer that was generated:
    ---
    {answer}
    ---

    Here are the source documents that were used as context:
    ---
    {context}
    ---

    Please list the numbers of the documents that directly support or contain the information presented in the answer. The document numbers should be separated by commas. If no documents are relevant, respond with "None".

    Example: 1, 3, 8
    Relevant Document Numbers:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({
            "answer": answer,
            "context": context_str
        })

        if response and response.strip().lower() != 'none':
            indices_str = response.strip().split(',')
            # 중복을 제거하고 효율적인 조회를 위해 set 사용
            relevant_indices = {int(i.strip()) - 1 for i in indices_str}
            filtered_docs = [doc for i, doc in enumerate(source_documents) if i in relevant_indices]
            return filtered_docs
        else:
            return []
    except Exception:
        # 파싱 오류나 다른 문제가 발생할 경우 안전하게 빈 리스트 반환
        return []

# --- 페이지 설정 ---
st.set_page_config(page_title="Modular RAG Chatbot", page_icon="🤖")
st.title("🤖 모듈화된 RAG 챗봇")
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
    system_prompt_input = st.text_area(
        "AI의 역할을 설정해주세요.",
        value=st.session_state.system_prompt,
        height=150,
        key="persona_input"
    )
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

    current_system_prompt = st.session_state.system_prompt

    if st.session_state.retriever:
        chain = get_conversational_rag_chain(st.session_state.retriever, current_system_prompt)
        
        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            source_documents = []
            
            # 답변 스트리밍
            for chunk in chain.stream({"input": user_input}):
                if "answer" in chunk:
                    ai_answer += chunk["answer"]
                    container.markdown(ai_answer + "▌")
                if "context" in chunk:
                    source_documents = chunk["context"]
            
            container.markdown(ai_answer)

            # 답변 생성 후, 관련성 높은 출처만 필터링 (수정된 부분)
            relevant_sources = []
            if source_documents:
                with st.spinner("출처 확인 중..."):
                    relevant_sources = filter_relevant_sources(ai_answer, source_documents)
            
            # 필터링된 출처 표시
            if relevant_sources:
                with st.expander("참고한 출처 보기"):
                    for i, source in enumerate(relevant_sources):
                        st.info(f"**출처 {i+1}**\n\n{source.page_content}")
                        st.divider()
            
            # 필터링된 출처와 함께 메시지 저장
            st.session_state.messages.append(
                {"role": "assistant", "content": ai_answer, "sources": relevant_sources}
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
