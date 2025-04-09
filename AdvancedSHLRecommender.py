try:
    _import_('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pysqlite3-binary"], check=True)
    _import_('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import re
import chromadb
import pandas as pd
import chromadb
import requests
import PyPDF2
from tenacity import retry, stop_after_attempt
import io
from typing import List, Dict, Optional, Tuple
from groq import Groq
from chromadb.utils import embedding_functions
from chromadb.api.types import Where, WhereDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CSV_PATH = "shl_solutions_clean.csv"
PRIMARY_COLLECTION = "shl_solutions"
PDF_COLLECTION = "pdf_content"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TEMPORARY_COLLECTION = "temp_pdf_search"

class SHLAdvancedRecommender:
    def _init_(self):
        # Initialize components
        self.client = Groq(api_key=GROQ_API_KEY)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.chroma_client = chromadb.Client()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)  
        self.solutions_collection = self._initialize_collection(PRIMARY_COLLECTION)
        self.pdf_collection = self._initialize_pdf_collection()

    def _initialize_collection(self, collection_name: str):
        """Initialize primary solutions collection"""
        try:
            return self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except:
            return self._create_solutions_collection()
    
    def _initialize_pdf_collection(self):
        """Initialize PDF content collection (empty by default)"""
        try:
            return self.chroma_client.get_collection(
                name=PDF_COLLECTION,
                embedding_function=self.embedding_function
            )
        except:
            return self.chroma_client.create_collection(
                name=PDF_COLLECTION,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
    
    def _create_solutions_collection(self):
        """Create primary solutions collection from CSV"""
        df = pd.read_csv(CSV_PATH)
        documents = []
        metadatas = []
        ids = []
        
        for _, row in df.iterrows():
            doc_text = f"""
            Title: {row.get('title', '')}
            Description: {row.get('description', '')}
            Job Level: {row.get('job_level', '')}
            Test Type: {row.get('test_type', '')}
            Languages: {row.get('languages', '')}
            Completion Time: {row.get('completion_time', '')}
            """
            
            metadata = {
                'title': row.get('title', ''),
                'job_level': row.get('job_level', ''),
                'test_type': row.get('test_type', ''),
                'completion_time': row.get('completion_time', ''),
                'languages': row.get('languages', ''),
                'url': row.get('url', ''),
                'adaptive_support': row.get('adaptive_support', 'No'),
                'remote_support': row.get('remote_support', 'Yes'),
                'duration': row.get('duration', 60)
            }
            
            documents.append(doc_text)
            metadatas.append(metadata)
            ids.append(str(row.get('url', '')))
        
        collection = self.chroma_client.create_collection(
            name=PRIMARY_COLLECTION,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return collection
    
    def _process_pdf(self, url: str, preferred_language: str = "english") -> List[Tuple[str, Dict]]:
        """Download and process PDF into chunks with metadata"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with io.BytesIO(response.content) as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = "\n".join([page.extract_text() for page in reader.pages])
                
                # Clean text
                text = re.sub(r'\s+', ' ', text).strip()
                
                # Split into chunks
                chunks = self.text_splitter.split_text(text)
                
                # Prepare chunks with metadata
                return [
                    (chunk, {
                        'source_url': url,
                        'chunk_index': i,
                        'language': preferred_language,
                        'content_type': 'pdf_text'
                    })
                    for i, chunk in enumerate(chunks)
                ]
        except Exception as e:
            print(f"Error processing PDF {url}: {str(e)}")
            return []
    
    @retry(stop=stop_after_attempt(3))
    def hybrid_search(self, query: str, filters: Optional[Where] = None, where_doc: Optional[WhereDocument] = None) -> List[Dict]:
        """Hybrid search combining vector and keyword matching"""
        try:
            vector_results = self.solutions_collection.query(
                query_texts=[query],
                where=filters,
                where_document=where_doc,
                n_results=10  # Return up to 10 results as per requirements
            )
            
            # Simple keyword boost
            keyword_boosted = []
            for i, doc in enumerate(vector_results['documents'][0]):
                score = vector_results['distances'][0][i]
                metadata = vector_results['metadatas'][0][i]
                
                # Simple keyword matching boost
                keyword_matches = sum(
                    1 for word in query.lower().split() 
                    if word in doc.lower()
                )
                boosted_score = score * (1 + (keyword_matches * 0.1))
                
                keyword_boosted.append({
                    'document': doc,
                    'metadata': metadata,
                    'score': boosted_score,
                    'vector_score': score,
                    'keyword_matches': keyword_matches
                })
            
            # Re-rank by combined score
            keyword_boosted.sort(key=lambda x: x['score'], reverse=True)
            
            return keyword_boosted[:10]  # Return up to 10 results as per requirements
        except Exception as e:
            if "backfill" in str(e):
                # Reset ChromaDB connection
                self.chroma_client.reset()
                return self.hybrid_search(query)  # Retry once
            raise
    
    def recommend_assessments(self, query: str) -> Dict:
        """Generate assessment recommendations matching the exact API spec"""
        search_results = self.hybrid_search(query)
        
        recommended_assessments = []
        for result in search_results:
            metadata = result['metadata']
            recommended_assessments.append({
                "url": metadata.get('url', ''),
                "adaptive_support": metadata.get('adaptive_support', 'No'),
                "description": metadata.get('description', ''),
                "duration": metadata.get('duration', 60),
                "remote_support": metadata.get('remote_support', 'Yes'),
                "test_type": metadata.get('test_type', '').split('|') if metadata.get('test_type') else []
            })
        
        return {
            "recommended_assessments": recommended_assessments
        }