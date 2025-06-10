"""Enhanced Performance Analyzer
------------------------------
Analyzes key metrics and saves the report to a timestamped text file
in the 'performance_analysis_records' directory.
"""

import os
import json
import statistics
from datetime import datetime


class PerformanceAnalyzer:
    def __init__(self, output_dir="performance_analysis_records"):
        """
        Initialize the performance analyzer and create the output directory.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_document_processing(self, processed_dir="converted_to_txt"):
        """
        Analyze document processing: file counts, sizes, and distribution.
        Returns a dictionary of metrics.
        """
        metrics = {
            "total_files": 0,
            "total_size_mb": 0.0,
            "min_size_mb": 0.0,
            "max_size_mb": 0.0,
            "avg_size_mb": 0.0,
            "stdev_size_mb": 0.0,
        }
        file_sizes = []

        if os.path.exists(processed_dir):
            for filename in os.listdir(processed_dir):
                if filename.endswith(".txt"):
                    file_path = os.path.join(processed_dir, filename)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    file_sizes.append(size_mb)

            if file_sizes:
                metrics["total_files"] = len(file_sizes)
                metrics["total_size_mb"] = sum(file_sizes)
                metrics["min_size_mb"] = min(file_sizes)
                metrics["max_size_mb"] = max(file_sizes)
                metrics["avg_size_mb"] = statistics.mean(file_sizes)
                if len(file_sizes) > 1:
                    metrics["stdev_size_mb"] = statistics.stdev(file_sizes)

        return metrics

    def analyze_chunking(
        self, chunked_data_path="converted_chunked_data/chunked_data_stats.json"
    ):
        """
        Analyze chunking: chunk counts and detailed size statistics.
        Returns a dictionary of metrics.
        """
        metrics = {
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
            "median_chunk_size": 0,
            "stdev_chunk_size": 0,
        }

        if os.path.exists(chunked_data_path):
            try:
                with open(chunked_data_path, "r") as f:
                    data = json.load(f)

                chunk_lengths = data.get("chunk_lengths", [])
                if chunk_lengths:
                    metrics["total_chunks"] = data.get(
                        "total_chunks", len(chunk_lengths)
                    )
                    metrics["avg_chunk_size"] = statistics.mean(chunk_lengths)
                    metrics["min_chunk_size"] = min(chunk_lengths)
                    metrics["max_chunk_size"] = max(chunk_lengths)
                    metrics["median_chunk_size"] = statistics.median(chunk_lengths)
                    if len(chunk_lengths) > 1:
                        metrics["stdev_chunk_size"] = statistics.stdev(chunk_lengths)

            except Exception as e:
                print(f"Warning: Could not read chunking stats: {e}")

        return metrics

    def analyze_vector_store(self, vec_dir="faiss_store"):
        """
        Analyze vector store: index size and number of documents.
        Returns a dictionary of metrics.
        """
        metrics = {"index_size_mb": 0.0, "doc_count": 0}

        if os.path.exists(vec_dir):
            # Calculate total size of the directory
            total_size = sum(
                os.path.getsize(os.path.join(vec_dir, f))
                for f in os.listdir(vec_dir)
                if os.path.isfile(os.path.join(vec_dir, f))
            )
            metrics["index_size_mb"] = total_size / (1024 * 1024)

            # Try to get document count from a FAISS index
            try:
                import faiss

                index_path = os.path.join(vec_dir, "pynucleus_mcp.faiss")
                if os.path.exists(index_path):
                    index = faiss.read_index(index_path)
                    metrics["doc_count"] = index.ntotal
                else:
                    print(f"Warning: FAISS index file not found at {index_path}")
            except ImportError:
                print(
                    "Warning: 'faiss' library not found. Cannot read document count from index."
                )
            except Exception as e:
                print(f"Warning: Could not read FAISS index: {e}")

        return metrics

    def analyze_search_performance(self, log_dir="vectordb_outputs"):
        """
        Analyze search performance: total queries and recall metrics.
        Returns a dictionary of metrics.
        """
        metrics = {"total_queries": 0, "successful_queries": 0, "recall_rate": 0.0}

        if os.path.exists(log_dir):
            log_files = [
                f for f in os.listdir(log_dir) if f.startswith("faiss_analysis_")
            ]
            if log_files:
                latest_log = max(
                    log_files, key=lambda f: os.path.getmtime(os.path.join(log_dir, f))
                )
                try:
                    with open(os.path.join(log_dir, latest_log), "r") as f:
                        content = f.read()
                        metrics["total_queries"] = content.count("Q:")
                        metrics["successful_queries"] = content.count("✓")
                        if metrics["total_queries"] > 0:
                            metrics["recall_rate"] = (
                                metrics["successful_queries"] / metrics["total_queries"]
                            )
                except Exception as e:
                    print(f"Warning: Could not read search performance log: {e}")

        return metrics

    def generate_report(self):
        """
        Generate, save, and print a comprehensive performance report.
        The report is saved as a timestamped .txt file.
        """
        # Collect metrics from all stages
        doc_metrics = self.analyze_document_processing()
        chunk_metrics = self.analyze_chunking()
        vec_metrics = self.analyze_vector_store()
        search_metrics = self.analyze_search_performance()

        # --- Build the report content as a single string ---
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_lines = [
            "=" * 60,
            f"          ENHANCED PERFORMANCE & STATISTICS REPORT",
            f"          Generated on: {timestamp}",
            "=" * 60,
            "\n📄 1. Document Processing",
            f"   - Total Files Processed: {doc_metrics['total_files']}",
            f"   - Total Size on Disk:    {doc_metrics['total_size_mb']:.2f} MB",
            "   --- File Size Stats (MB) ---",
            f"   - Average / Std Dev:     {doc_metrics['avg_size_mb']:.3f} / {doc_metrics['stdev_size_mb']:.3f}",
            f"   - Min / Max:             {doc_metrics['min_size_mb']:.3f} / {doc_metrics['max_size_mb']:.3f}",
            "\n🧩 2. Chunking",
            f"   - Total Chunks Created:  {chunk_metrics['total_chunks']:,}",
            "   --- Chunk Size Stats (characters) ---",
            f"   - Average / Std Dev:     {chunk_metrics['avg_chunk_size']:.1f} / {chunk_metrics['stdev_chunk_size']:.1f}",
            f"   - Min / Max:             {chunk_metrics['min_chunk_size']} / {chunk_metrics['max_chunk_size']}",
            f"   - Median:                {chunk_metrics['median_chunk_size']:.0f}",
            "\n💾 3. Vector Store",
            f"   - Total Index Size:      {vec_metrics['index_size_mb']:.2f} MB",
            f"   - Documents Indexed:     {vec_metrics['doc_count']:,}",
            "\n🔍 4. Search Performance",
            f"   - Total Queries Logged:  {search_metrics['total_queries']}",
            f"   - Successful Queries:    {search_metrics['successful_queries']}",
            f"   - Simple Recall Rate:    {search_metrics['recall_rate']:.2%}",
            "-" * 60,
        ]
        report_content = "\n".join(report_lines)

        # --- Save the report to a timestamped .txt file ---
        file_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"Report_{file_timestamp}.txt"
        file_path = os.path.join(self.output_dir, file_name)

        try:
            with open(file_path, "w") as f:
                f.write(report_content)
            print(f"\n✅ Report successfully saved to: {file_path}")
        except Exception as e:
            print(f"\n❌ Error: Could not save report. {e}")

        # --- Print the same report to the console ---
        print(report_content)


def main():
    """Example usage of the Enhanced Performance Analyzer."""
    # You may need to create dummy files/folders for this to run.
    # For example:
    # os.makedirs("processed_txt_files", exist_ok=True)
    # with open("processed_txt_files/doc1.txt", "w") as f: f.write("hello")
    analyzer = PerformanceAnalyzer()
    analyzer.generate_report()


if __name__ == "__main__":
    main()
