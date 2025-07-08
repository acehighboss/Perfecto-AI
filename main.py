# main.py (디버깅용)

import streamlit as st
import subprocess
import sys
import time

# st.session_state를 사용하여 앱 세션당 한 번만 설치를 시도합니다.
if "playwright_installed" not in st.session_state:
    st.set_page_config(page_title="Playwright Installation Debugger", layout="wide")
    st.title("🛠️ Playwright 설치 디버거")
    st.write("챗봇 실행에 필요한 Playwright 브라우저 설치 과정을 확인합니다.")
    st.info("이 화면은 문제 해결을 위한 디버깅용이며, 문제가 해결되면 원래의 main.py 코드로 되돌릴 것입니다.")

    # --- 명령어 실행 ---
    with st.spinner("`playwright install --with-deps` 명령을 실행 중입니다..."):
        # subprocess.run을 사용하여 'playwright install' 명령을 실행하고 결과를 캡처합니다.
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--with-deps"],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
    
    # --- 결과 출력 ---
    st.subheader("설치 로그")
    st.text("명령어 실행이 완료되었습니다. 아래 로그를 분석하여 원인을 파악합니다.")

    # Return Code 출력
    st.write(f"**Return Code:** `{result.returncode}`")
    if result.returncode == 0:
        st.success("명령어 자체는 성공적으로 종료되었습니다 (Return Code 0).")
    else:
        st.error("명령어 실행에 실패했습니다 (Return Code 1).")

    # STDOUT (표준 출력) 출력
    with st.expander("STDOUT (표준 출력) 보기"):
        st.code(result.stdout)

    # STDERR (표준 에러) 출력
    with st.expander("STDERR (표준 에러) 보기"):
        st.code(result.stderr)

    # --- 최종 진단 ---
    st.subheader("최종 진단")
    if "successfully" in result.stdout.lower():
        st.success("로그 분석 결과: 브라우저 다운로드는 성공한 것으로 보입니다.")
        st.write("이 상태에서 앱을 다시 시작하면 정상 작동할 수 있습니다.")
        st.session_state["playwright_installed"] = True
        if st.button("챗봇 시작하기"):
            st.rerun()
    else:
        st.error("로그 분석 결과: 브라우저 설치에 실패했습니다.")
        st.write("위의 STDERR (표준 에러) 로그에 실패의 원인이 담겨있을 가능성이 높습니다.")
        st.warning("**위 '설치 로그' 섹션의 모든 텍스트를 복사하여 저에게 다시 알려주세요.**")
    
    # 앱이 더 이상 진행되지 않도록 여기서 멈춤
    st.stop()


# --------------------------------------------------------------------------
# 디버깅이 끝나면, 아래의 원래 코드로 복원해야 합니다.
# --------------------------------------------------------------------------

# from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage, AIMessage
# from rag_pipeline import get_retriever_from_source, get_conversational_rag_chain, get_default_chain

# load_dotenv()
# st.set_page_config(...)
# ... (이하 모든 원래 코드) ...
