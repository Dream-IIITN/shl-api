import sys
import sqlite3
# try:
#     __import__('pysqlite3')
#     import sys
#     sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# except ImportError:
#     import subprocess
#     subprocess.run([sys.executable, "-m", "pip", "install", "pysqlite3-binary"], check=True)
#     __import__('pysqlite3')
#     import sys
#     sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
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
from config import GROQ_API_KEY
# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CSV_PATH = "shl_solutions_clean.csv"
PRIMARY_COLLECTION = "shl_solutions"
PDF_COLLECTION = "pdf_content"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TEMPORARY_COLLECTION = "temp_pdf_search"

class SHLAdvancedRecommender:
    def __init__(self):
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
                'download_url': row.get('download_url', ''),
                'download_language': row.get('download_language', '')
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
    
    def _search_pdf_content(self, query: str, solution_id: str, preferred_language: str = "english") -> List[Dict]:
        """Search within PDF content for a specific solution"""
        # Get solution metadata
        solution = self.solutions_collection.get(ids=[solution_id])
        if not solution or 'metadatas' not in solution or not solution['metadatas']:
            return []
        
        download_url = solution['metadatas'][0].get('download_url', '')
        if not download_url:
            return []
        
        # Create temporary collection for this search
        try:
            temp_collection = self.chroma_client.create_collection(
                name=TEMPORARY_COLLECTION,
                embedding_function=self.embedding_function
            )
        except:
            self.chroma_client.delete_collection(TEMPORARY_COLLECTION)
            temp_collection = self.chroma_client.create_collection(
                name=TEMPORARY_COLLECTION,
                embedding_function=self.embedding_function
            )
        
        # Process PDF and add to temporary collection
        pdf_chunks = self._process_pdf(download_url, preferred_language)
        if not pdf_chunks:
            return []
        
        temp_collection.add(
            documents=[chunk[0] for chunk in pdf_chunks],
            metadatas=[chunk[1] for chunk in pdf_chunks],
            ids=[f"{solution_id}_{i}" for i in range(len(pdf_chunks))]
        )
        
        # Perform search
        results = temp_collection.query(
            query_texts=[query],
            n_results=3
        )
        
        # Clean up
        self.chroma_client.delete_collection(TEMPORARY_COLLECTION)
        
        return [
            {
                'text': doc,
                'score': score,
                'metadata': metadata
            }
            for doc, score, metadata in zip(
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0]
            )
        ]

    @retry(stop=stop_after_attempt(3))
    def hybrid_search(self, query: str, filters: Optional[Where] = None, where_doc: Optional[WhereDocument] = None) -> List[Dict]:
        """Hybrid search combining vector and keyword matching"""
        try:
            vector_results = self.solutions_collection.query(
                query_texts=[query],
                where=filters,
                where_document=where_doc,
                n_results=5
            )
            
            # Simple keyword boost (could be enhanced with proper keyword search)
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
            
            return keyword_boosted[:3]  # Return top 3
        except Exception as e:
            if "backfill" in str(e):
                # Reset ChromaDB connection
                self.chroma_client.reset()
                return self.hybrid_search(query)  # Retry once
            raise
        
    
    def recommend_solution(self, user_query: str, user_language: str = "english") -> Dict:
        """Agentic recommendation flow with routing"""
        # Step 1: Initial hybrid search
        initial_results = self.hybrid_search(user_query)
        
        if not initial_results:
            return {"response": "No matching solutions found.", "sources": []}
        
        # Step 2: LLM decides whether to check PDF content
        router_prompt = f"""
        Analyze the user query and initial search results to determine if:
        1. The initial results are sufficient
        2. We should check PDF content for more details
        
        User Query: {user_query}
        
        Initial Results:
        {chr(10).join([res['document'][:200] + '...' for res in initial_results])}
        
        Respond ONLY with either "BASIC" or "PDF_DETAILS".
        """
        
        router_decision = self.client.chat.completions.create(
            messages=[{"role": "user", "content": router_prompt}],
            model="deepseek-r1-distill-llama-70b",
            temperature=0,
            max_tokens=10
        ).choices[0].message.content.strip()
        
        # Step 3: Handle routing decision
        if router_decision == "PDF_DETAILS":
            # Get top solution for PDF search
            top_solution = initial_results[0]
            pdf_results = self._search_pdf_content(
                query=user_query,
                solution_id=top_solution['metadata']['url'],
                preferred_language=user_language
            )
            
            # Generate enhanced response
            response = self._generate_enhanced_response(
                user_query=user_query,
                solution=top_solution,
                pdf_results=pdf_results
            )
            
            # Add user feedback mechanism
            response['feedback_prompt'] = "Was this information helpful? (yes/no)"
            return response
        else:
            # Generate basic response
            response = self._generate_basic_response(
                user_query=user_query,
                solutions=initial_results
            )
            response['feedback_prompt'] = "Was this recommendation helpful? (yes/no)"
            return response
    
    def _generate_enhanced_response(self, user_query: str, solution: Dict, pdf_results: List[Dict]) -> Dict:
        """Generate response with PDF details"""
        pdf_context = "\n\n".join(
            f"PDF Excerpt {i+1} (Relevance: {res['score']:.2f}):\n{res['text']}"
            for i, res in enumerate(pdf_results)
        )
        
        prompt = f"""
        You're an SHL solutions expert. Provide detailed recommendations based on:
        
        User Query: {user_query}
        
        Recommended Solution:
        Title: {solution['metadata']['title']}
        Description: {solution['document']}
        
        Relevant PDF Content:
        {pdf_context}
        
        Structure your response with:
        1. Solution Summary
        2. Key PDF Insights
        3. Implementation Recommendations
        4. Limitations/Caveats
        """
        
        llm_response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-r1-distill-llama-70b",
            temperature=0.3,
            max_tokens=1024
        ).choices[0].message.content
        
        return {
            "response": llm_response,
            "sources": [
                {
                    "type": "solution",
                    "title": solution['metadata']['title'],
                    "url": solution['metadata']['url']
                },
                *[
                    {
                        "type": "pdf_excerpt",
                        "source_url": res['metadata']['source_url'],
                        "relevance_score": res['score']
                    }
                    for res in pdf_results
                ]
            ]
        }
    def recommend_solution_json(self, user_query: str, user_language: str = "english"):
        result = self.recommend_solution(user_query, user_language)
        return {
            "solution": result.get("solution", ""),
            "reasoning": result.get("reasoning", ""),
            "sources": result.get("sources", []),
            "pdf_context": result.get("pdf_context", "")
        }
    def _generate_basic_response(self, user_query: str, solutions: List[Dict]) -> Dict:
        """Generate basic recommendation response"""
        solutions_str = "\n\n".join(
            f"Option {i+1}:\n"
            f"Title: {sol['metadata']['title']}\n"
            f"Description: {sol['document'][:200]}...\n"
            f"Job Level: {sol['metadata']['job_level']}\n"
            f"Test Type: {sol['metadata']['test_type']}\n"
            f"URL: {sol['metadata']['url']}"
            for i, sol in enumerate(solutions)
        )
        
        prompt = f"""
        Provide concise recommendations for these SHL solutions based on:
        
        User Query: {user_query}
        
        Available Solutions:
        {solutions_str}
        
        Structure your response with:
        1. Best Match Recommendation
        2. Alternative Options
        3. Key Selection Criteria
        """
        
        llm_response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-r1-distill-llama-70b",
            temperature=0.3,
            max_tokens=768
        ).choices[0].message.content
        
        return {
            "response": llm_response,
            "sources": [
                {
                    "type": "solution",
                    "title": sol['metadata']['title'],
                    "url": sol['metadata']['url']
                }
                for sol in solutions[:3]  # Top 3 solutions
            ]
        }
    
    def log_feedback(self, recommendation_id: str, feedback: str, notes: str = ""):
        """Log user feedback for continuous improvement"""
        # In a production system, this would write to a database
        print(f"Logged feedback for {recommendation_id}: {feedback} - {notes}")

# Example Usage
if __name__ == "__main__":
    recommender = SHLAdvancedRecommender()
    
    # Example query
    user_query = "I need a data entry test for entry-level candidates in the US that assesses typing speed and accuracy"
    
    # Get recommendation
    result = recommender.recommend_solution(user_query, user_language="english usa")
    print("Recommendation:")
    print(result["response"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source['type']}: {source.get('title', 'N/A')}")
    
    # Example feedback
    recommender.log_feedback(
        recommendation_id=result["sources"][0]["url"],
        feedback="yes",
        notes="Found exactly what I needed"
    )