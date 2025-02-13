# rag_processor.py
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import ( 
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS 
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

import pandas as pd
import numpy as np
import os
import psutil
from typing import List, Dict, Union, Optional
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CPUOptimizedRAGProcessor:
    def __init__(self, google_api_key: str, cache_dir: str = "/tmp/rag_cache"):
        """Initialize the RAG processor optimized for CPU usage."""
        try:
            os.environ["GOOGLE_API_KEY"] = google_api_key
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Get available CPU memory
            available_memory = psutil.virtual_memory().available
            self.chunk_size = min(500, int(available_memory / (1024 * 1024 * 10)))  # Adjust chunk size based on available memory
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-001",
                google_api_key=google_api_key,
                temperature=0.3,
                convert_system_message_to_human=True,
                max_output_tokens=1024  # Reduced for CPU optimization
            )
            
            # CPU-specific embeddings configuration
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                cache_folder=str(self.cache_dir / "embeddings_cache"),
                model_kwargs={'device': 'cpu'}
            )
            
            # Optimize text splitting for CPU
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=20,  # Reduced overlap for memory efficiency
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            self.vectorstore = None
            logger.info(f"CPUOptimizedRAGProcessor initialized with chunk size: {self.chunk_size}")
            
        except Exception as e:
            logger.error(f"Error initializing CPUOptimizedRAGProcessor: {str(e)}")
            raise

    def _get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available CPU memory."""
        available_memory = psutil.virtual_memory().available
        return min(50, max(10, int(available_memory / (1024 * 1024 * 100))))

    def _process_batch(self, texts: List[str]) -> List[Document]:
        """Process texts in memory-efficient batches."""
        batch_size = self._get_optimal_batch_size()
        all_docs = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            docs = [Document(page_content=text) for text in batch]
            split_docs = self.text_splitter.split_documents(docs)
            all_docs.extend(split_docs)
            
            # Force garbage collection after each batch
            import gc
            gc.collect()
            
        return all_docs

    def load_document(self, file_path: str, text_columns: Optional[Union[str, List[str]]] = None) -> List[str]:
        """Load document with CPU memory optimization."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_extension = file_path.suffix.lower()
            logger.info(f"Loading document: {file_path}")
            
            texts = []
            
            if file_extension == '.pdf':
                # Process PDF in chunks
                loader = PyPDFLoader(str(file_path))
                documents = []
                for page in loader.load_and_split():
                    documents.append(page)
                    if len(documents) >= 10:  # Process 10 pages at a time
                        texts.extend([doc.page_content for doc in documents])
                        documents.clear()
                texts.extend([doc.page_content for doc in documents])  # Remaining pages
                
            elif file_extension == '.docx':
                loader = Docx2txtLoader(str(file_path))
                documents = loader.load()
                texts = [doc.page_content for doc in documents]
                
            elif file_extension in ['.csv', '.xlsx', '.xls']:
                if not text_columns:
                    raise ValueError("text_columns must be specified for CSV/Excel files")
                
                text_columns = [text_columns] if isinstance(text_columns, str) else text_columns
                
                # Read in chunks for large files
                if file_extension == '.csv':
                    chunk_size = 1000  # Adjust based on your memory constraints
                    for chunk in pd.read_csv(file_path, usecols=text_columns, chunksize=chunk_size):
                        texts.extend(self._process_dataframe_chunk(chunk, text_columns))
                else:
                    df = pd.read_excel(file_path, usecols=text_columns)
                    texts.extend(self._process_dataframe_chunk(df, text_columns))
            
            logger.info(f"Successfully loaded {len(texts)} text segments")
            return texts
            
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}")
            raise

    def _process_dataframe_chunk(self, df: pd.DataFrame, text_columns: List[str]) -> List[str]:
        """Process a dataframe chunk efficiently."""
        chunk_texts = []
        for _, row in df.iterrows():
            row_text = " ".join(
                f"{col}: {str(row[col])}" 
                for col in text_columns
                if pd.notna(row[col])
            ).strip()
            if row_text:
                chunk_texts.append(row_text)
        return chunk_texts

    def process_document(self, file_path: str, text_columns: Optional[Union[str, List[str]]] = None) -> None:
        """Process document with CPU optimization."""
        try:
            cache_key = f"{file_path}_{str(text_columns)}"
            cache_file = self.cache_dir / f"{hash(cache_key)}.faiss"
            
            if cache_file.exists():
                logger.info("Loading from cache...")
                self.vectorstore = FAISS.load_local(str(self.cache_dir), self.embeddings)
                return
            
            logger.info("Starting document processing")
            texts = self.load_document(file_path, text_columns)
            
            # Process in smaller batches
            logger.info("Processing documents in batches")
            documents = self._process_batch(texts)
            
            # Create vector store
            logger.info("Creating vector store")
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Save to cache
            self.vectorstore.save_local(str(self.cache_dir))
            logger.info("Vector store created and cached successfully")
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    def query(self, question: str) -> Dict:
        """Memory-efficient query processing."""
        try:
            if not self.vectorstore:
                raise ValueError("Vector store not initialized. Process document first.")
            
            logger.info(f"Processing question: {question}")
            
            retriever = self.vectorstore.as_retriever(
                search_kwargs={
                    "k": 3,
                    "score_threshold": 0.5
                }
            )
            
            template = """
            Answer the question based on the provided context. If the context doesn't contain 
            relevant information, say "I cannot answer based on the provided context."
            
            Context: {context}
            Question: {question}
            
            Answer:
            """
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            response = chain.invoke(question)
            
            return {
                "question": question,
                "answer": response
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            raise