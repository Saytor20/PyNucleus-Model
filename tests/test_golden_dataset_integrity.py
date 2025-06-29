"""
Comprehensive dataset integrity tests for the golden dataset.

Tests ensure dataset quality, consistency, and completeness
for reliable evaluation and validation workflows.
"""

import pytest
import pandas as pd
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Set, Any
from datetime import datetime

# Test configuration
DATASET_PATH = "data/validation/golden_dataset.csv"
MIN_QUESTIONS = 50
MAX_QUESTIONS = 200
MIN_QUESTION_LENGTH = 10
MAX_QUESTION_LENGTH = 200
MIN_ANSWER_LENGTH = 15
MAX_ANSWER_LENGTH = 500

# Valid domains and difficulty levels
VALID_DOMAINS = {
    'separation', 'reaction_engineering', 'energy_efficiency', 'plant_design',
    'safety_management', 'thermodynamics', 'fluid_mechanics', 'heat_transfer',
    'equipment_standards', 'logistics', 'environmental', 'drying_processes',
    'economics', 'sustainability', 'mass_transfer', 'utilities', 'equipment_selection',
    'separation_process', 'maintenance', 'crystallization', 'information_retrieval',
    'adsorption', 'membrane_separation', 'fluidization', 'water_treatment',
    'filtration', 'heat_mass_transfer', 'distillation', 'material_selection'
}

VALID_DIFFICULTIES = {'easy', 'medium', 'hard'}

# Test fixtures
@pytest.fixture
def golden_dataset():
    """Load golden dataset for testing."""
    try:
        df = pd.read_csv(DATASET_PATH)
        return df
    except FileNotFoundError:
        pytest.fail(f"Golden dataset not found at {DATASET_PATH}")


class TestDatasetStructure:
    """Test dataset structure and format."""
    
    def test_dataset_exists(self):
        """Test that the golden dataset file exists."""
        assert Path(DATASET_PATH).exists(), f"Dataset file not found: {DATASET_PATH}"
    
    def test_dataset_loadable(self, golden_dataset):
        """Test that the dataset can be loaded successfully."""
        assert isinstance(golden_dataset, pd.DataFrame)
        assert not golden_dataset.empty, "Dataset is empty"
    
    def test_required_columns(self, golden_dataset):
        """Test that all required columns are present."""
        required_columns = {'question', 'expected_answer', 'expected_keywords', 'domain', 'difficulty'}
        actual_columns = set(golden_dataset.columns)
        
        missing_columns = required_columns - actual_columns
        assert not missing_columns, f"Missing required columns: {missing_columns}"
    
    def test_dataset_size(self, golden_dataset):
        """Test dataset size is within acceptable range."""
        dataset_size = len(golden_dataset)
        assert MIN_QUESTIONS <= dataset_size <= MAX_QUESTIONS, \
            f"Dataset size {dataset_size} not in range [{MIN_QUESTIONS}, {MAX_QUESTIONS}]"
    
    def test_no_empty_rows(self, golden_dataset):
        """Test that there are no completely empty rows."""
        empty_rows = golden_dataset.isnull().all(axis=1).sum()
        assert empty_rows == 0, f"Found {empty_rows} completely empty rows"


class TestDataQuality:
    """Test data quality and completeness."""
    
    def test_no_missing_values(self, golden_dataset):
        """Test that there are no missing values in required columns."""
        required_columns = ['question', 'expected_answer', 'expected_keywords', 'domain', 'difficulty']
        
        for column in required_columns:
            missing_count = golden_dataset[column].isnull().sum()
            assert missing_count == 0, f"Found {missing_count} missing values in column '{column}'"
    
    def test_no_duplicate_questions(self, golden_dataset):
        """Test that there are no duplicate questions."""
        duplicate_count = golden_dataset.duplicated(subset=['question']).sum()
        assert duplicate_count == 0, f"Found {duplicate_count} duplicate questions"
    
    def test_question_length(self, golden_dataset):
        """Test that all questions have appropriate length."""
        question_lengths = golden_dataset['question'].str.len()
        
        too_short = (question_lengths < MIN_QUESTION_LENGTH).sum()
        too_long = (question_lengths > MAX_QUESTION_LENGTH).sum()
        
        assert too_short == 0, f"Found {too_short} questions shorter than {MIN_QUESTION_LENGTH} characters"
        assert too_long == 0, f"Found {too_long} questions longer than {MAX_QUESTION_LENGTH} characters"
    
    def test_answer_length(self, golden_dataset):
        """Test that all expected answers have appropriate length."""
        answer_lengths = golden_dataset['expected_answer'].str.len()
        
        too_short = (answer_lengths < MIN_ANSWER_LENGTH).sum()
        too_long = (answer_lengths > MAX_ANSWER_LENGTH).sum()
        
        assert too_short == 0, f"Found {too_short} answers shorter than {MIN_ANSWER_LENGTH} characters"
        assert too_long == 0, f"Found {too_long} answers longer than {MAX_ANSWER_LENGTH} characters"
    
    def test_keywords_format(self, golden_dataset):
        """Test that expected keywords are properly formatted."""
        for idx, row in golden_dataset.iterrows():
            keywords = row['expected_keywords']
            
            # Check that keywords exist
            assert keywords and isinstance(keywords, str), \
                f"Row {idx}: Keywords missing or not string"
            
            # Check that keywords are comma-separated
            keyword_list = [k.strip() for k in keywords.split(',')]
            assert len(keyword_list) >= 2, \
                f"Row {idx}: Need at least 2 keywords, found {len(keyword_list)}"
            
            # Check that no keyword is empty
            empty_keywords = [k for k in keyword_list if not k]
            assert not empty_keywords, \
                f"Row {idx}: Found empty keywords"
    
    def test_valid_domains(self, golden_dataset):
        """Test that all domains are valid."""
        invalid_domains = golden_dataset[~golden_dataset['domain'].isin(VALID_DOMAINS)]['domain'].unique()
        assert len(invalid_domains) == 0, \
            f"Found invalid domains: {list(invalid_domains)}. Valid domains: {VALID_DOMAINS}"
    
    def test_valid_difficulties(self, golden_dataset):
        """Test that all difficulty levels are valid."""
        invalid_difficulties = golden_dataset[~golden_dataset['difficulty'].isin(VALID_DIFFICULTIES)]['difficulty'].unique()
        assert len(invalid_difficulties) == 0, \
            f"Found invalid difficulties: {list(invalid_difficulties)}. Valid difficulties: {VALID_DIFFICULTIES}"


class TestContentQuality:
    """Test content quality and coherence."""
    
    def test_question_format(self, golden_dataset):
        """Test that questions are properly formatted."""
        for idx, row in golden_dataset.iterrows():
            question = row['question']
            
            # Check that question ends with question mark or has question words
            is_question = (
                question.strip().endswith('?') or
                any(word in question.lower() for word in ['what', 'how', 'why', 'when', 'where', 'which', 'define', 'explain', 'describe', 'list', 'give'])
            )
            
            assert is_question, f"Row {idx}: Question doesn't appear to be a proper question: '{question}'"
    
    def test_answer_coherence(self, golden_dataset):
        """Test that answers are coherent and substantial."""
        for idx, row in golden_dataset.iterrows():
            answer = row['expected_answer']
            
            # Check for basic sentence structure
            word_count = len(answer.split())
            assert word_count >= 5, f"Row {idx}: Answer too short (only {word_count} words)"
            
            # Check that answer doesn't just repeat the question
            question_words = set(row['question'].lower().split())
            answer_words = set(answer.lower().split())
            
            overlap = len(question_words & answer_words)
            total_answer_words = len(answer_words)
            
            if total_answer_words > 0:
                overlap_ratio = overlap / total_answer_words
                assert overlap_ratio < 0.8, \
                    f"Row {idx}: Answer appears to mostly repeat question (overlap: {overlap_ratio:.2f})"
    
    def test_keyword_relevance(self, golden_dataset):
        """Test that keywords are relevant to questions and answers."""
        for idx, row in golden_dataset.iterrows():
            keywords = [k.strip().lower() for k in row['expected_keywords'].split(',')]
            question_text = row['question'].lower()
            answer_text = row['expected_answer'].lower()
            combined_text = f"{question_text} {answer_text}"
            
            # Check that at least some keywords appear in question or answer
            found_keywords = [k for k in keywords if k in combined_text]
            relevance_ratio = len(found_keywords) / len(keywords)
            
            assert relevance_ratio >= 0.3, \
                f"Row {idx}: Only {relevance_ratio:.1%} of keywords found in question/answer"


class TestDomainDistribution:
    """Test domain and difficulty distribution."""
    
    def test_domain_coverage(self, golden_dataset):
        """Test that dataset covers multiple domains adequately."""
        domain_counts = golden_dataset['domain'].value_counts()
        
        # Check minimum number of domains
        unique_domains = len(domain_counts)
        assert unique_domains >= 10, f"Too few domains represented: {unique_domains}"
        
        # Check that no single domain dominates
        max_domain_count = domain_counts.max()
        total_questions = len(golden_dataset)
        max_domain_ratio = max_domain_count / total_questions
        
        assert max_domain_ratio <= 0.3, \
            f"Single domain has too many questions: {max_domain_ratio:.1%}"
    
    def test_difficulty_distribution(self, golden_dataset):
        """Test that difficulty levels are well distributed."""
        difficulty_counts = golden_dataset['difficulty'].value_counts()
        
        # Check that all difficulty levels are present
        for difficulty in VALID_DIFFICULTIES:
            assert difficulty in difficulty_counts.index, \
                f"Missing difficulty level: {difficulty}"
            
            count = difficulty_counts[difficulty]
            assert count >= 5, f"Too few questions for difficulty '{difficulty}': {count}"
    
    def test_domain_difficulty_matrix(self, golden_dataset):
        """Test that domains have varied difficulty levels."""
        domain_difficulty = golden_dataset.groupby(['domain', 'difficulty']).size().unstack(fill_value=0)
        
        # Check that major domains have multiple difficulty levels
        major_domains = golden_dataset['domain'].value_counts().head(10).index
        
        for domain in major_domains:
            if domain in domain_difficulty.index:
                difficulties_present = (domain_difficulty.loc[domain] > 0).sum()
                assert difficulties_present >= 2, \
                    f"Domain '{domain}' should have at least 2 difficulty levels"


class TestDatasetIntegrity:
    """Test overall dataset integrity."""
    
    def test_unique_identifiers(self, golden_dataset):
        """Test that question-answer pairs can be uniquely identified."""
        # Create unique identifiers based on question and answer
        identifiers = []
        for _, row in golden_dataset.iterrows():
            content = f"{row['question']}_{row['expected_answer']}"
            identifier = hashlib.md5(content.encode()).hexdigest()
            identifiers.append(identifier)
        
        unique_identifiers = len(set(identifiers))
        total_questions = len(golden_dataset)
        
        assert unique_identifiers == total_questions, \
            f"Non-unique question-answer pairs found: {total_questions - unique_identifiers}"
    
    def test_encoding_consistency(self, golden_dataset):
        """Test that text encoding is consistent."""
        for idx, row in golden_dataset.iterrows():
            for column in ['question', 'expected_answer', 'expected_keywords']:
                text = row[column]
                
                # Try to encode/decode to check for encoding issues
                try:
                    text.encode('utf-8').decode('utf-8')
                except UnicodeEncodeError:
                    pytest.fail(f"Row {idx}, column '{column}': Encoding issue detected")
    
    def test_dataset_freshness(self):
        """Test that dataset file is reasonably fresh."""
        dataset_file = Path(DATASET_PATH)
        if dataset_file.exists():
            file_age_days = (datetime.now() - datetime.fromtimestamp(dataset_file.stat().st_mtime)).days
            
            # Warn if dataset is very old (but don't fail)
            if file_age_days > 90:
                print(f"Warning: Dataset file is {file_age_days} days old")


class TestValidationCompatibility:
    """Test compatibility with validation systems."""
    
    def test_evaluation_readiness(self, golden_dataset):
        """Test that dataset is ready for evaluation use."""
        # Check that all required fields for evaluation are present
        evaluation_fields = ['question', 'expected_answer', 'expected_keywords', 'domain', 'difficulty']
        
        for field in evaluation_fields:
            assert field in golden_dataset.columns, f"Missing evaluation field: {field}"
    
    def test_expert_validation_readiness(self, golden_dataset):
        """Test that dataset is ready for expert validation workflows."""
        # Check that questions can be categorized for expert assignment
        for idx, row in golden_dataset.iterrows():
            domain = row['domain']
            difficulty = row['difficulty']
            
            assert domain in VALID_DOMAINS, f"Row {idx}: Invalid domain for expert assignment"
            assert difficulty in VALID_DIFFICULTIES, f"Row {idx}: Invalid difficulty for expert assignment"


# Utility functions for test reporting

def generate_dataset_statistics(dataset_path: str = DATASET_PATH) -> Dict[str, Any]:
    """Generate comprehensive dataset statistics."""
    try:
        df = pd.read_csv(dataset_path)
        
        return {
            "total_questions": len(df),
            "domains": {
                "count": df['domain'].nunique(),
                "distribution": df['domain'].value_counts().to_dict()
            },
            "difficulties": {
                "count": df['difficulty'].nunique(),
                "distribution": df['difficulty'].value_counts().to_dict()
            },
            "question_lengths": {
                "mean": df['question'].str.len().mean(),
                "min": df['question'].str.len().min(),
                "max": df['question'].str.len().max()
            },
            "answer_lengths": {
                "mean": df['expected_answer'].str.len().mean(),
                "min": df['expected_answer'].str.len().min(),
                "max": df['expected_answer'].str.len().max()
            },
            "keyword_counts": {
                "mean": df['expected_keywords'].str.count(',').add(1).mean(),
                "min": df['expected_keywords'].str.count(',').add(1).min(),
                "max": df['expected_keywords'].str.count(',').add(1).max()
            }
        }
    except Exception as e:
        return {"error": str(e)}


def run_integrity_check(dataset_path: str = DATASET_PATH) -> Dict[str, Any]:
    """Run complete integrity check and return results."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_path": dataset_path,
        "statistics": generate_dataset_statistics(dataset_path),
        "tests_passed": 0,
        "tests_failed": 0,
        "issues": []
    }
    
    try:
        # Run pytest programmatically
        import subprocess
        result = subprocess.run(
            ['python', '-m', 'pytest', __file__, '-v'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        results["test_output"] = result.stdout
        results["test_errors"] = result.stderr
        results["return_code"] = result.returncode
        
    except Exception as e:
        results["error"] = str(e)
    
    return results


if __name__ == "__main__":
    # Run integrity check when script is executed directly
    results = run_integrity_check()
    print(json.dumps(results, indent=2)) 