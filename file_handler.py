import os
import tempfile
from llama_parse import LlamaParse
import asyncio # asyncio import 추가

# LlamaParse를 사용하기 위한 parser 객체 초기화
parser = LlamaParse(result_type="markdown")

# ▼▼▼ [수정] 함수를 async def로 변경하여 비동기 처리 ▼▼▼
async def get_documents_from_files(uploaded_files):
    """
    업로드된 파일 리스트를 LlamaParse를 사용하여 비동기적으로 로드하고 구조화합니다.
    """
    tasks = []
    temp_file_paths = []

    # 1. 파일을 임시 저장하고, 각 파일에 대한 파싱 작업을 리스트에 추가
    for uploaded_file in uploaded_files:
        try:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                temp_file_paths.append(tmp_file_path)
            
            # 비동기 파싱 작업을 태스크 리스트에 추가
            tasks.append(parser.aload_data(tmp_file_path))
        except Exception as e:
            print(f"Error creating temp file for {uploaded_file.name}: {e}")

    # 2. asyncio.gather를 사용하여 모든 파싱 작업을 동시에 실행
    all_documents = []
    if tasks:
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    all_documents.extend(result)
                elif isinstance(result, Exception):
                    print(f"An error occurred during parsing: {result}")
        except Exception as e:
            print(f"Error during parallel parsing: {e}")
            
    # 3. 처리 후 모든 임시 파일 삭제
    for path in temp_file_paths:
        if os.path.exists(path):
            os.remove(path)
            
    return all_documents
