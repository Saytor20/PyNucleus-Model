import os
from langchain_unstructured import UnstructuredLoader
from PyPDF2 import PdfReader

# Configuration - Updated to new directory structure
INPUT_DIR = 'source_documents'  # Where users put their PDF/DOCX files
OUTPUT_DIR = 'converted_to_txt'  # Where processed text files go

def process_documents():
    """
    Process all documents in the input directory and convert them to text files.
    Handles PDF, DOCX, TXT, and other file types.
    """
    # Check if the input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"📂 Creating directory: '{INPUT_DIR}'")
        os.makedirs(INPUT_DIR, exist_ok=True)
        print(f"ℹ Please place your files (PDF, DOCX, TXT, etc.) in the '{INPUT_DIR}' directory and run the script again.")
        return

    # Create the output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files_to_process = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]

    if not files_to_process:
        print(f"ℹ The '{INPUT_DIR}' directory is empty. Nothing to process.")
        return

    print(f"--- 📄 Starting processing for {len(files_to_process)} file(s) in '{INPUT_DIR}' ---")

    for filename in files_to_process:
        # Skip hidden files like .DS_Store
        if filename.startswith('.'):
            continue

        input_path = os.path.join(INPUT_DIR, filename)
        # Use the original filename without "processed_" prefix for cleaner names
        output_filename = f"{os.path.splitext(os.path.basename(filename))[0]}.txt"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        print(f" ▶ Processing: {filename}")

        try:
            # Handle PDF files differently
            if filename.lower().endswith('.pdf'):
                # Use PyPDF2 for PDF files
                reader = PdfReader(input_path)
                full_text = ""
                for page in reader.pages:
                    full_text += page.extract_text() + "\n\n"
            else:
                # Use UnstructuredLoader for other file types
                loader = UnstructuredLoader(input_path)
                documents = loader.load()
                full_text = "\n\n".join([doc.page_content for doc in documents])

            # Save the extracted text to a new .txt file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            print(f"   • Success! Saved to: {output_path}")

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")

    print(f"\n All files processed.")

def main():
    """Example usage of the document processor."""
    process_documents()

if __name__ == "__main__":
    main() 