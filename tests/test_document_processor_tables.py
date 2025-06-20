"""
Test enhanced document processor with PDF table extraction.
"""
import tempfile
import pandas as pd
import pytest
from pathlib import Path
from pynucleus.rag.document_processor import DocumentProcessor

def test_table_type_detection():
    """Test table type detection functionality."""
    processor = DocumentProcessor()
    
    # Test financial table
    financial_df = pd.DataFrame({
        'Item': ['Product A', 'Product B'],
        'Cost': [100, 200],
        'Revenue': [150, 300],
        'Profit': [50, 100]
    })
    table_type = processor._detect_table_type(financial_df)
    assert table_type == "financial"
    
    # Test specifications table
    specs_df = pd.DataFrame({
        'Parameter': ['Temperature', 'Pressure'],
        'Specification': ['25°C', '1 atm'],
        'Requirement': ['±2°C', '±0.1 atm']
    })
    table_type = processor._detect_table_type(specs_df)
    assert table_type == "specifications"

def test_table_cleaning():
    """Test table cleaning functionality."""
    processor = DocumentProcessor()
    
    # Create messy DataFrame
    messy_df = pd.DataFrame([
        ["  Item A  ", " 100 ", None],
        ["Item B", None, "200"],
        [None, None, None],  # Empty row
        ["  Item C  ", "300", "400"]
    ])
    messy_df.columns = ["", "col2", ""]  # Some empty column names
    
    cleaned_df = processor._clean_table(messy_df)
    
    # Should have proper column names
    assert "col_0" in cleaned_df.columns
    assert "col2" in cleaned_df.columns
    
    # Should remove empty row and strip whitespace
    assert len(cleaned_df) == 3  # One empty row removed
    assert cleaned_df.iloc[0, 0] == "Item A"  # Whitespace stripped

def test_table_summary_creation():
    """Test table summary creation for RAG indexing."""
    processor = DocumentProcessor()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test CSV file
        test_csv = Path(temp_dir) / "test_financial_tables.csv"
        df = pd.DataFrame({
            'Product': ['A', 'B'],
            'Cost': [100, 200],
            'Revenue': [150, 300]
        })
        df.to_csv(test_csv, index=False)
        
        # Test summary creation
        summary = processor._create_table_summary([str(test_csv)])
        
        assert "test_financial_tables.csv" in summary
        assert "Product, Cost, Revenue" in summary
        assert "2" in summary  # Row count
        assert "Product: A, Cost: 100, Revenue: 150" in summary  # Sample data

@pytest.mark.skip(reason="Requires PDF file and camelot installation")
def test_pdf_table_extraction():
    """Test PDF table extraction (requires sample PDF)."""
    processor = DocumentProcessor()
    
    # This would require a sample PDF with tables
    # pdf_path = Path("tests/data/sample_with_tables.pdf")
    # result = processor._extract_pdf_tables(pdf_path)
    # assert result["tables_extracted"] >= 0
    pass

def test_document_processor_initialization():
    """Test document processor initialization with table extraction."""
    processor = DocumentProcessor(tables_output_dir="test_output")
    
    assert processor.tables_output_dir == Path("test_output")
    assert len(processor.table_keywords) > 0
    assert "table" in processor.table_keywords
    assert "data" in processor.table_keywords 