from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from file_handler import get_documents_from_files

def get_retriever_from_source(source_type, source_input):
    """
    URL 또는 파일(LlamaParse 사용)로부터 문서를 로드하고,
    텍스트를 분할하여 EnsembleRetriever를 생성합니다.
    """
    documents = []
    if source_type == "URL":
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            loader = WebBaseLoader(source_input, header_template=headers)
            documents = loader.load()
        except Exception as e:
            print(f"Error loading URL: {e}")
            return None
    elif source_type == "Files":
        # LlamaParse를 통해 구조화된 문서를 가져옵니다.
        documents = get_documents_from_files(source_input)

    if not documents:
        return None

    # LlamaParse가 이미 문서를 잘 구조화했지만, 일관된 처리를 위해 한 번 더 분할합니다.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    if not splits:
        return None

    # 1. 키워드 기반 검색기(BM25Retriever) 초기화
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 7

    # 2. 의미 기반 검색기(FAISS) 초기화
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

    # 3. 앙상블 검색기 생성
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )
    
    return ensemble_retriever


def get_conversational_rag_chain(retriever, system_prompt):
    """
    RAG 기능을 수행하는 체인을 생성합니다.
    """
    template = f"""{system_prompt}

Answer the user's request based on the provided "Context".
The "Context" is a collection of text snippets from a document or a URL.

**Instructions:**
1. First, carefully read the user's request and the provided "Context".
2. If the user is asking a direct question (e.g., "Who is...?", "What is...?"), find the answer within the "Context" and provide a clear, concise response.
3. If the user is making a creative request (e.g., "write a script", "summarize this in a poem", "create a marketing slogan"), use the main ideas, themes, and key information from the "Context" as the raw material for your creative work.
4. If the "Context" does not contain relevant information to fulfill the user's request, you must respond with: "죄송합니다. 제공된 내용만으로는 요청하신 작업을 수행할 수 없습니다."

**Context:**
{{context}}

**User's Request:**
{{input}}
"""
    rag_prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    return create_retrieval_chain(retriever, document_chain)


def get_default_chain(system_prompt):
    """
    기본적인 대화형 체인을 생성합니다.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{question}")]
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    return prompt | llm | StrOutputParser()
