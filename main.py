import sys
import argparse
from src.pipeline.pipeline import ContextualRAGPipeline
from config import config
import PyPDF2

def load_file_content(file_path):
    try:
        if file_path.lower().endswith('.pdf'):
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
                # Extract metadata
                metadata = {
                    "file_name": file_path.split("/")[-1],
                    "num_pages": len(pdf_reader.pages),
                    "author": pdf_reader.metadata.author if pdf_reader.metadata.author else "Unknown",
                    "creation_date": pdf_reader.metadata.creation_date.strftime('%Y-%m-%d') if pdf_reader.metadata.creation_date else "Unknown",
                    "modification_date": pdf_reader.metadata.modification_date.strftime('%Y-%m-%d') if pdf_reader.metadata.modification_date else "Unknown",
                    "producer": pdf_reader.metadata.producer if pdf_reader.metadata.producer else "Unknown",
                    "subject": pdf_reader.metadata.subject if pdf_reader.metadata.subject else "Unknown",
                    "title": pdf_reader.metadata.title if pdf_reader.metadata.title else "Unknown",
                }
                
                return content, metadata            
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                metadata = {
                    "file_name": file_path.split("/")[-1],
                    "file_type": "text"
                }
                return content, metadata
    except IOError as e:
        print(f"Error reading file: {e}")
        return None, None
    except PyPDF2.errors.PdfReadError as e:
        print(f"Error reading PDF file: {e}")
        return None, None

def list_knowledge_base(pipeline):
    all_docs = pipeline.vector_store.get_all_documents()
    if not all_docs:
        print("The knowledge base is empty.")
    else:
        print(f"Knowledge Base Contents: ")
        for i, (doc_id, text) in enumerate(zip(all_docs['ids'], all_docs['documents']), 1):
            print(f"{i}. ID: {doc_id}")
            print(f"   Content: {text[:100]}...")  # Display first 100 characters
            print()
        print(f"---- End of the Knowledge Base ----")

def main():
    parser = argparse.ArgumentParser(description="Contextual RAG Pipeline")
    parser.add_argument('--list_kb', action='store_true', help='List the contents of the knowledge base')
    parser.add_argument('--clear_kb', action='store_true', help='Clear the knowledge base')
    args = parser.parse_args()

     # Check if SERPAPI_API_KEY is set
    if not config.get('API', 'SERPAPI_API_KEY'):
        print("Error: SERPAPI_API_KEY is not set in the configuration.")
        sys.exit(1)


    pipeline = ContextualRAGPipeline()

    if args.list_kb:
        list_knowledge_base(pipeline)
        return
    
    if args.clear_kb:
        pipeline.vector_store.clear_database()
        print("Knowledge base has been cleared.")
        return

    while True:
        print("\nOptions:")
        print("1. Upload a file (PDF or utf-8 text only): ")
        print("2. Enter a query")
        print("3. List knowledge base")
        print("4. Exit")
        
        choice = input("Enter your choice (1/2/3/4): ").strip()
        
        if choice == '1':
            file_path = input("Enter the path to the file you want to upload: ").strip()
            file_content, metadata = load_file_content(file_path)
            if file_content and metadata:
                pipeline.add_document(file_content, metadata)
                print("File content added to local knowledge base.")
            else:
                print("Failed to load file content.")
        
        elif choice == '2':
            query = input("Enter your query: ").strip()
            if query:
                result = pipeline.process_query(query)
                print("\nGenerated Answer:")
                print(result['answer'])
                print("\nSources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {'Local Document' if source['is_local'] else 'Web Result'}")
                    print(f"   Text: {source['text'][:100]}...")
                    if not source['is_local']:
                        print(f"   URL: {source['url']}")
                    print(f"   Relevance Score: {source['combined_score']:.4f}")
                    print()
            else:
                print("Query cannot be empty.")
        
        elif choice == '3':
            list_knowledge_base(pipeline)
        
        elif choice == '4':
            print("Exiting the program. Goodbye!")
            sys.exit(0)
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()