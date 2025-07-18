import nest_asyncio
nest_asyncio.apply()

import bs4
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_core.documents import Document as LangChainDocument
from langchain_core.runnables import RunnableLambda
from langchain_experimental.text_splitter import SemanticChunker
from newspaper import Article

from file_handler import get_documents_from_files

async def get_retriever_from_source_async(source_type, source_input):
    """
    URL 또는 파일로부터 문서를 로드하고,
    Cohere Rerank가 적용된 ContextualCompressionRetriever를 생성합니다.
    """
    documents = []
    if source_type == "URL":
        try:
            # ▼▼▼ [수정] 위키피디아와 일반 사이트를 구분하여 처리하는 하이브리드 로직 ▼▼▼
            if "wikipedia.org" in source_input:
                # --- 위키피디아 전용 정제 로직 (BeautifulSoup) ---
                loader = WebBaseLoader(web_path=source_input)
                raw_docs = await loader.aload() # aload()로 비동기 호출출

                for doc in raw_docs:
                    soup = bs4.BeautifulSoup(doc.page_content, "lxml")
                    content_container = soup.find("div", class_="mw-parser-output")
                    if not content_container: continue

                    unwanted_selectors = [
                        "div.reflist", "ol.references", "div.navbox", "table.navbox",
                        "div.hatnote", "div.rellink", "span.mw-editsection",
                        "div#toc", "div#catlinks"
                    ]
                    for selector in unwanted_selectors:
                        for element in content_container.select(selector):
                            element.decompose()
                    
                    cleaned_text = content_container.get_text(separator="\n", strip=True)
                    if cleaned_text:
                        documents.append(LangChainDocument(page_content=cleaned_text, metadata=doc.metadata))
            else:
                # --- 일반 사이트용 정제 로직 (newspaper3k) ---
                loop = asyncio.get_running_loop()
                article = await loop.run_in_executor(None, lambda: Article(url=source_input, language='ko'))
                await loop.run_in_executor(None, article.download)
                await loop.run_in_executor(None, article.parse)
                
                cleaned_text = article.text
                if cleaned_text:
                    metadata = {"source": source_input}
                    documents = [LangChainDocument(page_content=cleaned_text, metadata=metadata)]
            # ▲▲▲ [수정] 여기까지 ▲▲▲
        except Exception as e:
            print(f"Error processing URL {source_input}: {e}")
            return None
    elif source_type == "Files":
        txt_files = [f for f in source_input if f.name.endswith('.txt')]
        other_files = [f for f in source_input if not f.name.endswith('.txt')]

        for txt_file in txt_files:
            try:
                content = txt_file.getvalue().decode('utf-8')
                doc = LangChainDocument(page_content=content, metadata={"source": txt_file.name})
                documents.append(doc)
            except Exception as e:
                print(f"Error reading .txt file {txt_file.name}: {e}")

        if other_files:
            try:
                llama_documents = await get_documents_from_files(other_files)
                if llama_documents:
                    langchain_docs = [LangChainDocument(page_content=doc.text, metadata=doc.metadata) for doc in llama_documents]
                    documents.extend(langchain_docs)
            except Exception as e:
                print(f"Error parsing files with LlamaParse: {e}")

    if not documents:
        print("Warning: No processable documents found.")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
        buffer_size=1,
        min_chunk_size=100
    )
    splits = text_splitter.split_documents(documents)
    
    if not splits:
        return None

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 10

    # --- 임베딩 병목 해결을 위한 핵심 비동기 로직 ---
    texts_for_embedding = [doc.page_content for doc in splits]
    # aembed_documents를 사용하여 모든 텍스트 청크에 대한 임베딩을 동시에 요청
    embedded_vectors = await embeddings.aembed_documents(texts_for_embedding)
    
    # 미리 계산된 임베딩을 사용하여 FAISS 인덱스를 생성 (네트워크 통신 없음)
    text_embedding_pairs = zip(texts_for_embedding, embedded_vectors)
    vectorstore = FAISS.from_embeddings(text_embeddings=list(text_embedding_pairs), embedding=embeddings, metadatas=[doc.metadata for doc in splits])
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.4, 0.6]
    )
    compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )
    
    return compression_retriever

# ▼▼▼ [수정] 기존 동기 함수는 비동기 함수를 실행하는 래퍼(Wrapper) 역할만 수행 ▼▼▼
def get_retriever_from_source(source_type, source_input):
    """
    비동기 함수인 get_retriever_from_source_async를 실행하고 결과를 반환합니다.
    Streamlit과 같은 동기 환경과의 호환성을 위해 사용됩니다.
    """
    try:
        return asyncio.run(get_retriever_from_source_async(source_type, source_input))
    except Exception as e:
        print(f"Error running async retriever creation: {e}")
        return None

def get_conversational_rag_chain(retriever, system_prompt):
    """
    '역할 부여 및 사고 과정(Chain-of-Thought)' 프롬프트를 사용하여
    질의 재작성 기능이 포함된 RAG 체인을 생성합니다.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    query_rewrite_template = """You are an expert at rewriting user questions into effective search queries for a vector database.
Your goal is to transform a user's question into a query that is rich with keywords and semantic details, based on the context of an AI industry trend report.
The rewritten query must be in Korean.

Follow these steps:
1.  **Analyze Intent**: Understand the core information the user is seeking.
2.  **Identify Key Concepts**: Extract main topics and entities (e.g., companies, technologies, people, countries).
3.  **Expand and Specify**: Brainstorm related, more specific terms.
    - If the user asks about a country's AI (e.g., "AI from China"), expand to major tech companies from that country (e.g., "Alibaba", "Tencent", "Baidu") and specific technologies (e.g., "LLM", "Tongyi Qianwen").
    - If the user asks about a concept (e.g., "AI safety"), expand to related terms (e.g., "AI alignment", "red-teaming", "AI ethics").
4.  **Construct a Descriptive Query**: Combine the original and expanded concepts into a single, comprehensive search query.

Original question: {question}
Rewritten Search Query:"""
    query_rewrite_prompt = ChatPromptTemplate.from_template(query_rewrite_template)
    
    query_rewriter = query_rewrite_prompt | llm | StrOutputParser()

    def rewrite_and_retrieve(query: str):
        print(f"Original Query: {query}")
        rewritten_query = query_rewriter.invoke({"question": query})
        print(f"Rewritten Query: {rewritten_query}")
        return retriever.invoke(rewritten_query)

    smart_retriever = RunnableLambda(rewrite_and_retrieve)

    rag_prompt_template = f"""{system_prompt}

Answer the user's request based on the provided "Context".

**Context:**
{{context}}

**User's Request:**
{{input}}
"""
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
    
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    
    retrieval_chain = create_retrieval_chain(smart_retriever, document_chain)
    
    return retrieval_chain


def get_default_chain(system_prompt):
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{question}")]
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    return prompt | llm | StrOutputParser()
