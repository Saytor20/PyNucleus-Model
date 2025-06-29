# Golden Dataset Curation and Validation Guide

## Overview

This guide provides comprehensive documentation for the expert-driven golden dataset curation and validation system implemented in PyNucleus. The system ensures high-quality, expert-validated question-answer pairs for reliable evaluation and continuous improvement of the RAG system.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Dataset Structure](#dataset-structure)
3. [Expert Validation Workflow](#expert-validation-workflow)
4. [Validation Manager Usage](#validation-manager-usage)
5. [Dataset Integrity Testing](#dataset-integrity-testing)
6. [Quality Metrics and Monitoring](#quality-metrics-and-monitoring)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## System Architecture

### Components

The validation system consists of three main components:

1. **Golden Dataset (`data/validation/golden_dataset.csv`)**
   - Expanded to 100+ expert-curated question-answer pairs
   - Covers 25+ chemical engineering domains
   - Three difficulty levels: easy, medium, hard

2. **Validation Manager (`src/pynucleus/eval/validation_manager.py`)**
   - Expert registration and assignment system
   - Automated validation workflows
   - Quality metrics tracking
   - Feedback integration

3. **Integrity Tests (`tests/test_golden_dataset_integrity.py`)**
   - Comprehensive dataset validation
   - Automated quality checks
   - Distribution analysis

### Data Flow

```
Expert Registration → Task Assignment → Validation Submission → Quality Metrics → Dataset Updates
```

## Dataset Structure

### Required Columns

The golden dataset contains the following required columns:

- **question**: The question to be answered by the RAG system
- **expected_answer**: Expert-validated correct answer
- **expected_keywords**: Comma-separated keywords for evaluation
- **domain**: Technical domain (e.g., 'thermodynamics', 'separation')
- **difficulty**: Question difficulty ('easy', 'medium', 'hard')

### Domain Coverage

The dataset covers 25+ domains including:

- **Core Processes**: separation, reaction_engineering, thermodynamics
- **Equipment Design**: heat_transfer, equipment_standards, fluid_mechanics
- **Plant Design**: plant_design, safety_management, energy_efficiency
- **Specialized Areas**: membrane_separation, crystallization, adsorption
- **Economics & Sustainability**: economics, environmental, sustainability

### Quality Standards

#### Question Requirements
- Length: 10-200 characters
- Must be properly formatted questions
- Domain-relevant content
- Clear and unambiguous

#### Answer Requirements
- Length: 15-500 characters
- Technically accurate
- Comprehensive but concise
- Domain-specific terminology

#### Keyword Requirements
- Minimum 2 keywords per question
- Relevant to question/answer content
- Lowercase, comma-separated
- Technical terms preferred

## Expert Validation Workflow

### 1. Expert Registration

```python
from src.pynucleus.eval.validation_manager import ValidationManager, ExpertProfile, ExpertLevel

# Initialize validation system
manager = ValidationManager()

# Register expert
expert = ExpertProfile(
    expert_id="expert_001",
    name="Dr. Sarah Chen",
    email="s.chen@example.com",
    level=ExpertLevel.PRINCIPAL,
    domains=["separation", "distillation", "thermodynamics"],
    specializations=["distillation_design", "heat_integration"]
)

success = manager.register_expert(expert)
```

### 2. Task Assignment

```python
# Assign validation tasks automatically
assignments = manager.assign_validation_tasks(max_assignments_per_expert=10)

# assignments = {
#     "expert_001": ["q1_id", "q2_id", ...],
#     "expert_002": ["q3_id", "q4_id", ...]
# }
```

### 3. Validation Submission

```python
from src.pynucleus.eval.validation_manager import ValidationRecord, ValidationStatus
from datetime import datetime

# Submit validation
validation = ValidationRecord(
    question_id="q1_id",
    expert_id="expert_001",
    validation_date=datetime.now(),
    status=ValidationStatus.APPROVED,
    confidence_score=0.95,
    feedback="Technically accurate and well-formulated",
    suggested_improvements="Consider adding units to numerical values",
    domain_accuracy=0.9,
    technical_accuracy=0.95,
    clarity_score=0.9,
    version_hash="abc123"
)

manager.submit_validation(validation)
```

### 4. Status Tracking

```python
# Check validation status for specific question
status = manager.get_validation_status("q1_id")

# Check overall dataset status
overall_status = manager.get_validation_status()
```

## Validation Manager Usage

### Initialization

```python
from src.pynucleus.eval.validation_manager import initialize_validation_system

# Initialize with default settings
manager = initialize_validation_system()

# Custom initialization
manager = ValidationManager(
    dataset_path="custom/path/golden_dataset.csv",
    validation_dir="custom/validation/directory"
)
```

### Expert Management

#### Expert Levels and Capabilities

- **JUNIOR**: Can validate easy questions
- **SENIOR**: Can validate easy and medium questions  
- **PRINCIPAL**: Can validate all difficulty levels
- **DOMAIN_EXPERT**: Can validate all levels, domain-specific expertise

#### Domain Assignment Logic

Experts are automatically assigned to questions based on:
1. Domain expertise match
2. Difficulty level capability
3. Current workload balance
4. Historical validation performance

### Quality Metrics

```python
# Generate comprehensive validation report
report = manager.generate_validation_report(include_details=True)

# Key metrics included:
# - Validation coverage percentage
# - Expert consensus scores
# - Domain distribution analysis
# - Quality trends over time
```

### Periodic Validation

```python
# Schedule automatic re-validation every 30 days
manager.schedule_periodic_validation(interval_days=30)

# This identifies questions needing re-validation based on:
# - Age of last validation
# - Changes in consensus
# - Expert feedback indicating issues
```

### Feedback Integration

```python
# Integrate expert feedback into dataset
feedback = {
    "improved_answer": "Enhanced technical accuracy with specific units",
    "improved_keywords": "updated,keywords,with,better,terms",
    "corrected_domain": "thermodynamics",
    "corrected_difficulty": "medium"
}

manager.integrate_expert_feedback("q1_id", feedback)
```

## Dataset Integrity Testing

### Running Integrity Tests

```bash
# Run all integrity tests
python -m pytest tests/test_golden_dataset_integrity.py -v

# Run specific test categories
python -m pytest tests/test_golden_dataset_integrity.py::TestDatasetStructure -v
python -m pytest tests/test_golden_dataset_integrity.py::TestDataQuality -v
python -m pytest tests/test_golden_dataset_integrity.py::TestContentQuality -v
```

### Test Categories

#### 1. Dataset Structure Tests
- File existence and loadability
- Required columns presence
- Dataset size validation
- Empty row detection

#### 2. Data Quality Tests
- Missing value detection
- Duplicate question identification
- Length validation (questions, answers)
- Keyword format validation
- Domain and difficulty validation

#### 3. Content Quality Tests
- Question format validation
- Answer coherence analysis
- Keyword relevance checking
- Technical accuracy assessment

#### 4. Distribution Tests
- Domain coverage analysis
- Difficulty distribution validation
- Domain-difficulty matrix analysis

#### 5. Integrity Tests
- Unique identifier generation
- Encoding consistency
- Dataset freshness monitoring

### Automated Integrity Checking

```python
from tests.test_golden_dataset_integrity import run_integrity_check

# Run comprehensive integrity check
results = run_integrity_check()

# Results include:
# - Test pass/fail statistics
# - Dataset statistics
# - Issue identification
# - Recommendations
```

## Quality Metrics and Monitoring

### Key Performance Indicators

1. **Validation Coverage**: Percentage of questions validated by experts
2. **Consensus Score**: Agreement level between multiple expert validations
3. **Expert Confidence**: Average confidence scores from validations
4. **Domain Balance**: Distribution of questions across domains
5. **Quality Trends**: Improvement over time metrics

### Monitoring Dashboard

```python
# Generate monitoring report
validation_metrics = {
    "coverage": manager.quality_metrics.get('validation_coverage', 0.0),
    "consensus": manager.quality_metrics.get('validation_consensus', 0.0),
    "confidence": manager.quality_metrics.get('average_expert_confidence', 0.0),
    "pending": manager.quality_metrics.get('pending_validations', 0)
}
```

### Alert Thresholds

- **Coverage < 80%**: Insufficient validation coverage
- **Consensus < 70%**: Expert disagreement on validations
- **Confidence < 75%**: Low expert confidence in validations
- **Pending > 20%**: Too many unvalidated questions

## Best Practices

### For Dataset Curation

1. **Question Formulation**
   - Use clear, unambiguous language
   - Focus on practical engineering scenarios
   - Include varied complexity levels
   - Ensure domain relevance

2. **Answer Quality**
   - Provide technically accurate information
   - Include specific values and units where appropriate
   - Use domain-standard terminology
   - Balance completeness with conciseness

3. **Keyword Selection**
   - Choose technically relevant terms
   - Include key concepts and processes
   - Avoid overly generic terms
   - Maintain consistency with domain standards

### For Expert Validation

1. **Registration Best Practices**
   - Accurately specify domain expertise
   - Provide appropriate qualification level
   - List relevant specializations
   - Keep contact information current

2. **Validation Guidelines**
   - Assess technical accuracy thoroughly
   - Consider practical applicability
   - Provide constructive feedback
   - Suggest specific improvements

3. **Quality Maintenance**
   - Regular re-validation of older questions
   - Feedback integration for continuous improvement
   - Consensus building on disputed validations
   - Domain expert consultation for specialized topics

### For System Administration

1. **Monitoring**
   - Daily validation coverage checks
   - Weekly quality metric reviews
   - Monthly expert performance analysis
   - Quarterly dataset expansion planning

2. **Maintenance**
   - Regular integrity test execution
   - Expert feedback integration
   - Performance optimization
   - Documentation updates

## Troubleshooting

### Common Issues

#### 1. Low Validation Coverage

**Symptoms**: Coverage below 80%
**Causes**: 
- Insufficient expert participation
- Unbalanced expert domain assignments
- High expert workload

**Solutions**:
```python
# Increase assignment limits
assignments = manager.assign_validation_tasks(max_assignments_per_expert=15)

# Register additional experts
# Check expert workload distribution
expert_stats = manager.generate_expert_performance()
```

#### 2. Expert Consensus Issues

**Symptoms**: Consensus scores below 70%
**Causes**:
- Ambiguous questions
- Domain boundary issues
- Expert qualification mismatches

**Solutions**:
```python
# Review conflicting validations
consensus_issues = manager.get_validation_status()

# Refine question formulation
# Add domain expert reviewers
# Provide expert training materials
```

#### 3. Dataset Integrity Failures

**Symptoms**: Test failures in integrity checks
**Causes**:
- Data corruption
- Encoding issues
- Format inconsistencies

**Solutions**:
```bash
# Run detailed integrity check
python tests/test_golden_dataset_integrity.py

# Review specific test failures
# Apply data cleaning procedures
# Restore from validated backup
```

#### 4. Performance Issues

**Symptoms**: Slow validation processing
**Causes**:
- Large dataset size
- Complex validation logic
- I/O bottlenecks

**Solutions**:
```python
# Optimize validation algorithms
# Implement caching mechanisms
# Use parallel processing for batch operations
```

### Error Resolution

#### ValidationManager Errors

```python
# Handle registration errors
try:
    success = manager.register_expert(expert)
    if not success:
        # Check for duplicate expert IDs
        # Verify expert profile completeness
except Exception as e:
    logger.error(f"Registration failed: {e}")
```

#### Dataset Loading Errors

```python
# Handle dataset access issues
try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    # Check file path
    # Verify file permissions
    # Restore from backup
except pd.errors.EmptyDataError:
    # Check file content
    # Validate CSV format
```

### Support and Maintenance

#### Log Analysis

```python
# Review validation manager logs
from src.pynucleus.utils.logger import logger

# Check recent validation activities
# Identify error patterns
# Monitor performance metrics
```

#### Data Backup and Recovery

```bash
# Create dataset backup
cp data/validation/golden_dataset.csv data/validation/backups/golden_dataset_$(date +%Y%m%d).csv

# Restore from backup if needed
cp data/validation/backups/golden_dataset_YYYYMMDD.csv data/validation/golden_dataset.csv
```

#### System Health Checks

```python
# Run comprehensive system health check
health_report = {
    "dataset_integrity": run_integrity_check(),
    "validation_status": manager.get_validation_status(),
    "expert_performance": manager.generate_expert_performance(),
    "quality_metrics": manager.quality_metrics
}
```

## Conclusion

The expert validation system provides a robust framework for maintaining high-quality golden datasets through expert involvement, automated workflows, and comprehensive monitoring. Regular use of the validation processes, integrity testing, and quality monitoring ensures continuous improvement of the evaluation system and overall RAG performance.

For additional support or questions, refer to the system logs, run diagnostic scripts, or consult the technical documentation in the codebase. 