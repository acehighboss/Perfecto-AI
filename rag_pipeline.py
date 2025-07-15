import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import re

class SmartTextSplitter:
    """개선된 문단/문장 기반 지능형 텍스트 분할기"""
    
    def __init__(self, max_chunk_size=1000, overlap_sentences=1):
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences
    
    def flexible_sentence_split(self, text):
        """유연한 문장 분할 - 마침표 없는 문장도 고려"""
        # 명확한 문장 끝 패턴
        sentence_endings = r'[.!?;]["\']?\s+'
        
        # 1차: 명확한 문장 끝으로 분할
        sentences = re.split(sentence_endings, text)
        
        # 2차: 마침표 없는 문장들을 줄바꿈으로 분할
        refined_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 긴 문장이고 줄바꿈이 있으면 줄바꿈 단위로 추가 분할
            if len(sentence) > 200 and '\n' in sentence:
                lines = sentence.split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        refined_sentences.append(line)
            else:
                refined_sentences.append(sentence)
        
        # 3차: 너무 짧은 문장들을 인접한 문장과 병합
        final_sentences = []
        current_sentence = ""
        
        for sentence in refined_sentences:
            if len(sentence) < 50 and current_sentence:
                # 짧은 문장은 이전 문장과 병합
                current_sentence += " " + sentence
            else:
                if current_sentence:
                    final_sentences.append(current_sentence)
                current_sentence = sentence
        
        if current_sentence:
            final_sentences.append(current_sentence)
        
        return final_sentences
    
    def split_by_paragraphs(self, text):
        """문단별 분할 (우선 방법)"""
        # 문단 분할 (빈 줄 기준)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_size = len(paragraph)
            
            # 토큰 제한 확인
            if current_size + paragraph_size > self.max_chunk_size and current_chunk:
                # 현재 청크 저장
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
                current_size = paragraph_size
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_size += paragraph_size
        
        # 마지막 청크 저장
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def split_by_sentences(self, text):
        """문장별 분할 (토큰 제한 초과 시)"""
        sentences = self.flexible_sentence_split(text)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence)
            
            # 단일 문장이 토큰 제한을 초과하는 경우
            if sentence_size > self.max_chunk_size:
                # 현재 청크 저장
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_size = 0
                
                # 긴 문장을 강제로 분할
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk + " " + word) > self.max_chunk_size:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word
                    else:
                        temp_chunk += " " + word if temp_chunk else word
                
                if temp_chunk:
                    chunks.append(temp_chunk.strip())
                continue
            
            # 토큰 제한 확인
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                # 오버랩 처리
                overlap_text = self._get_overlap_text(current_chunk)
                chunks.append(current_chunk.strip())
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_size = len(current_chunk)
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_size += sentence_size
        
        # 마지막 청크 저장
        if current_chunk:
            chunks.append(current_chunk.strip
