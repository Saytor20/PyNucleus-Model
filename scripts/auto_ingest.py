import time
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import the ingestion function
def safe_ingest_single_file(file_path):
    try:
        from src.pynucleus.rag.collector import ingest_single_file
        result = ingest_single_file(str(file_path))
        logging.info(f"Ingested {file_path}: {result}")
    except Exception as e:
        logging.error(f"Failed to ingest {file_path}: {e}")

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        # Only ingest .pdf, .txt, .md files, ignore hidden/temp files
        if file_path.suffix.lower() in {'.pdf', '.txt', '.md'} and not file_path.name.startswith('.'):
            logging.info(f"Detected new file: {file_path}")
            # Wait briefly to ensure file is fully written
            time.sleep(2)
            safe_ingest_single_file(file_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    watch_dir = Path("data/01_raw/source_documents")
    if not watch_dir.exists():
        logging.error(f"Watch directory does not exist: {watch_dir}")
        exit(1)
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, str(watch_dir), recursive=False)
    observer.start()
    logging.info(f"Started auto-ingest watcher on {watch_dir}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join() 