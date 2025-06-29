"""
Extract tables from a PDF, clean them with pandas, save as CSV.

Usage:
    from pynucleus.data.table_cleaner import extract_tables
    extract_tables('docs/specs.pdf', 'data/02_processed')
"""
import camelot
import pandas as pd
from pathlib import Path
from ..utils.logger import logger

def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up DataFrame by removing whitespace and empty rows/columns."""
    # First rename columns, then strip whitespace, then drop empty cols/rows
    df.columns = [c if c else f"col_{i}" for i, c in enumerate(df.columns)]
    df = df.map(lambda x: str(x).strip() if pd.notna(x) else x)
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    return df.reset_index(drop=True)

def extract_tables(pdf_path: str, out_dir: str, max_pages: int | None = None) -> list[str]:
    """Extract tables from PDF and save as cleaned CSV files."""
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages_arg = "all" if max_pages is None else f"1-{max_pages}"
    tables = camelot.read_pdf(str(pdf_path), pages=pages_arg)
    csv_files = []
    
    for idx, table in enumerate(tables):
        df = _tidy(table.df)
        csv_file = out_dir / f"{pdf_path.stem}_table{idx+1}.csv"
        df.to_csv(csv_file, index=False)
        csv_files.append(str(csv_file))
        logger.info(f"Saved cleaned table → {csv_file}")
    
    return csv_files

# Export TableCleaner class for compatibility
class TableCleaner:
    @staticmethod
    def extract_tables(pdf_path: str, out_dir: str, max_pages: int | None = None) -> list[str]:
        return extract_tables(pdf_path, out_dir, max_pages) 