# main.py
from ray_processor import process_questions_parallel
import os
from dotenv import load_dotenv
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    # Configuration
    FILE_PATH = "D:\se_data\cities.csv"
    TEXT_COLUMNS = ["CityID","CityName","Zipcode","CountryID"]
    CACHE_DIR = Path("/tmp/rag_cache")
    
    # Example questions
    questions = [
        "What is the csv file about, explain it clearly",
        "How many different cities are in the entire list",
        "How many columns are in the csv file"
    ]
    
    try:
        start_time = time.time()
        
        results = process_questions_parallel(
            google_api_key=GOOGLE_API_KEY,
            file_path=FILE_PATH,
            questions=questions,
            text_columns=TEXT_COLUMNS,
            cache_dir=str(CACHE_DIR)
        )
        
        processing_time = time.time() - start_time
        
        # Print results
        print(f"\nProcessed {len(questions)} questions in {processing_time:.2f} seconds")
        print(f"Average time per question: {processing_time/len(questions):.2f} seconds")
        
        for result in results:
            print(f"\nQuestion: {result['question']}")
            print(f"Answer: {result['answer']}")
            if 'error' in result:
                print(f"Error: {result['error']}")
        
    except Exception as e:
        logger.error(f"Error running the application: {str(e)}")

if __name__ == "__main__":
    main()