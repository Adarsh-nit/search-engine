# ray_processor.py
import ray
from typing import List, Dict, Union, Optional
import time
import logging
import json
import psutil
from pathlib import Path

logger = logging.getLogger(__name__)

def get_optimal_cpu_count():
    """Determine optimal number of CPUs to use."""
    cpu_count = psutil.cpu_count(logical=False)  # Physical CPU cores only
    return max(1, cpu_count - 1)  # Leave one core free for system processes

@ray.remote(num_cpus=1)
class CPUOptimizedRAGActor:
    def __init__(
        self, 
        google_api_key: str, 
        file_path: str, 
        cache_dir: str,
        text_columns: Optional[Union[str, List[str]]] = None
    ):
        """Initialize RAG actor with CPU optimization."""
        try:
            from rag_processor import CPUOptimizedRAGProcessor
            
            self.processor = CPUOptimizedRAGProcessor(
                google_api_key=google_api_key,
                cache_dir=cache_dir
            )
            self.processor.process_document(file_path, text_columns)
            logger.info("RAG Actor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG Actor: {str(e)}")
            raise
    
    def process_question(self, question: str) -> Dict:
        """Process a single question with memory monitoring."""
        try:
            return self.processor.query(question)
        except Exception as e:
            logger.error(f"Error in actor processing question: {str(e)}")
            return {
                "question": question,
                "error": str(e),
                "answer": "Error processing question"
            }

def initialize_ray(dashboard_port: int = 8265) -> None:
    """Initialize Ray with CPU-specific settings."""
    if not ray.is_initialized():
        num_cpus = get_optimal_cpu_count()
        
        ray.init(
            num_cpus=num_cpus,
            dashboard_port=dashboard_port,
            include_dashboard=True,
            logging_level=logging.INFO,
            _system_config={
                "object_spilling_config": json.dumps({
                    "type": "filesystem",
                    "params": {
                        "directory_path": "/tmp/ray_spill"
                    }
                })
            }
        )
        logger.info(f"Ray initialized with {num_cpus} CPUs")

def process_questions_parallel(
    google_api_key: str,
    file_path: str,
    questions: List[str],
    text_columns: Optional[Union[str, List[str]]] = None,
    cache_dir: str = "/tmp/rag_cache",
    dashboard_port: int = 8265
) -> List[Dict]:
    """Process questions in parallel with CPU optimization."""
    try:
        # Initialize Ray with optimal CPU count
        initialize_ray(dashboard_port=dashboard_port)
        num_actors = get_optimal_cpu_count()
        logger.info(f"Starting parallel processing with {num_actors} actors")
        
        # Create cache directory
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize actors
        actors = [
            CPUOptimizedRAGActor.remote(
                google_api_key=google_api_key,
                file_path=file_path,
                cache_dir=cache_dir,
                text_columns=text_columns
            )
            for _ in range(num_actors)
        ]
        
        # Distribute questions among actors
        futures = []
        for i, question in enumerate(questions):
            actor = actors[i % num_actors]
            futures.append(actor.process_question.remote(question))
        
        # Process with memory-efficient batching
        logger.info(f"Processing {len(questions)} questions")
        results = []
        pending = futures
        
        while pending:
            done, pending = ray.wait(
                pending,
                num_returns=1,
                timeout=60.0
            )
            
            if done:
                result = ray.get(done[0])
                results.append(result)
                logger.info(f"Processed {len(results)}/{len(questions)} questions")
            
        logger.info("Parallel processing completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}")
        raise
    finally:
        ray.shutdown()