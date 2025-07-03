"""
PyNucleus Data Module

Data management and access utilities for the PyNucleus system.
"""

from .mock_data_manager import MockDataManager, get_mock_data_manager

# Make TableCleaner import optional to avoid camelot dependency
try:
    from .table_cleaner import TableCleaner
    TABLE_CLEANER_AVAILABLE = True
except ImportError:
    TableCleaner = None
    TABLE_CLEANER_AVAILABLE = False

__all__ = [
    'MockDataManager',
    'get_mock_data_manager',
    'TableCleaner',
    'TABLE_CLEANER_AVAILABLE'
] 