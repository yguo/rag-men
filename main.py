import sys
import argparse
from src.pipeline.pipeline import ContextualRAGPipeline

def load_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except IOError as e:
        print(f"Error reading file: {e}")
        return None

def list_knowledge_base(pipeline):
    all_docs = pipeline.vector_store.get_all_documents()
    if not all_docs['ids']:
        print("The knowledge base is empty.")
    else:
        print("Knowledge Base Contents:")
        for i, (doc_id, text) in enumerate(zip(all_docs['ids'], all_docs['documents']), 1):
            print(f"{i}. ID: {doc_id}")
            print(f"   Content: {text[:100]}...")  # Display first 100 characters
            print()

def main():
    parser = argparse.ArgumentParser(description="Contextual RAG Pipeline")
    parser.add_argument('--list_kb', action='store_true', help='List the contents of the knowledge base')
    args = parser.parse_args()

    pipeline = ContextualRAGPipeline()

    if args.list_kb:
        list_knowledge_base(pipeline)
        return

    while True:
        print("\nOptions:")
        print("1. Upload a file")
        print("2. Enter a query")
        print("3. List knowledge base")
        print("4. Exit")
        
        choice = input("Enter your choice (1/2/3/4): ").strip()
        
        if choice == '1':
            file_path = input("Enter the path to the file you want to upload: ").strip()
            file_content = load_file_content(file_path)
            if file_content:
                pipeline.add_document(file_content)
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