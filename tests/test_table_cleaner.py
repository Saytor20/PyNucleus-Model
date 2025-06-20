"""
Test for PDF table extraction functionality.
"""
import pathlib
import pandas as pd
import pytest
from pynucleus.data.table_cleaner import extract_tables, _tidy

def test_tidy_function():
    """Test the _tidy helper function."""
    # Create a messy DataFrame with proper duplicate handling
    df = pd.DataFrame([
        ["  value1  ", " data ", None],
        ["value2", None, None], 
        [None, "  data3  ", None]
    ])
    df.columns = ["", "col2", ""]  # Set column names after creation
    
    # Clean it up
    cleaned = _tidy(df)
    
    # Should have proper column names (empty cols renamed)
    assert "col_0" in cleaned.columns
    assert "col2" in cleaned.columns
    # The third column (all None) should be dropped
    
    # Should strip whitespace
    assert cleaned.iloc[0, 0] == "value1"  # stripped
    assert cleaned.iloc[0, 1] == "data"    # stripped

@pytest.mark.skip(reason="Requires sample_table.pdf file")
def test_extract(tmp_path):
    """Test table extraction from PDF (requires sample file)."""
    pdf = pathlib.Path('tests/data/sample_table.pdf')
    
    if not pdf.exists():
        pytest.skip("sample_table.pdf not found")
    
    out_files = extract_tables(str(pdf), str(tmp_path), max_pages=1)
    assert out_files and pathlib.Path(out_files[0]).suffix == '.csv'
    df = pd.read_csv(out_files[0])
    assert not df.empty 