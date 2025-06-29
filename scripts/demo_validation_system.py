#!/usr/bin/env python3
"""
Demonstration of the Expert Validation System for Golden Dataset Curation.

This script shows how to:
1. Initialize the validation system
2. Register expert validators
3. Assign validation tasks
4. Check validation status
5. Run dataset integrity tests
6. Generate validation reports

Usage: python scripts/demo_validation_system.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pynucleus.eval.validation_manager import (
    ValidationManager,
    ExpertProfile,
    ExpertLevel,
    ValidationRecord,
    ValidationStatus,
    initialize_validation_system
)
from datetime import datetime
import json


def demo_expert_registration():
    """Demonstrate expert registration process."""
    print("=== EXPERT REGISTRATION DEMO ===")
    
    # Initialize validation manager
    manager = ValidationManager()
    
    # Create sample expert profiles
    experts = [
        ExpertProfile(
            expert_id="expert_001",
            name="Dr. Sarah Chen",
            email="s.chen@university.edu",
            level=ExpertLevel.PRINCIPAL,
            domains=["separation", "distillation", "thermodynamics"],
            specializations=["distillation_design", "heat_integration"]
        ),
        ExpertProfile(
            expert_id="expert_002", 
            name="Prof. Michael Rodriguez",
            email="m.rodriguez@institute.org",
            level=ExpertLevel.DOMAIN_EXPERT,
            domains=["reaction_engineering", "catalysis", "safety_management"],
            specializations=["reactor_design", "process_safety"]
        ),
        ExpertProfile(
            expert_id="expert_003",
            name="Dr. Lisa Wang",
            email="l.wang@company.com",
            level=ExpertLevel.SENIOR,
            domains=["heat_transfer", "equipment_design", "energy_efficiency"],
            specializations=["heat_exchanger_design", "process_optimization"]
        )
    ]
    
    # Register experts
    registered_count = 0
    for expert in experts:
        if manager.register_expert(expert):
            print(f"✓ Registered: {expert.name} ({expert.level.value})")
            registered_count += 1
        else:
            print(f"✗ Failed to register: {expert.name}")
    
    print(f"\nRegistered {registered_count} experts successfully")
    return manager


def demo_task_assignment(manager):
    """Demonstrate automatic task assignment."""
    print("\n=== TASK ASSIGNMENT DEMO ===")
    
    # Assign validation tasks
    assignments = manager.assign_validation_tasks(max_assignments_per_expert=5)
    
    if assignments:
        print("Task assignments:")
        for expert_id, task_list in assignments.items():
            print(f"  {expert_id}: {len(task_list)} tasks")
        
        total_tasks = sum(len(tasks) for tasks in assignments.values())
        print(f"\nTotal tasks assigned: {total_tasks}")
    else:
        print("No tasks to assign (all questions may already be validated)")
    
    return assignments


def demo_validation_submission(manager):
    """Demonstrate validation submission process."""
    print("\n=== VALIDATION SUBMISSION DEMO ===")
    
    # Create sample validation record
    validation = ValidationRecord(
        question_id="sample_question_001",
        expert_id="expert_001",
        validation_date=datetime.now(),
        status=ValidationStatus.APPROVED,
        confidence_score=0.95,
        feedback="Technically accurate and well-formulated question with clear expected answer.",
        suggested_improvements="Consider adding specific temperature units for better precision.",
        domain_accuracy=0.9,
        technical_accuracy=0.95,
        clarity_score=0.9,
        version_hash="demo_hash_001"
    )
    
    # Submit validation
    success = manager.submit_validation(validation)
    
    if success:
        print("✓ Sample validation submitted successfully")
        print(f"  Question ID: {validation.question_id}")
        print(f"  Expert: {validation.expert_id}")
        print(f"  Status: {validation.status.value}")
        print(f"  Confidence: {validation.confidence_score:.1%}")
    else:
        print("✗ Failed to submit validation")


def demo_status_checking(manager):
    """Demonstrate validation status checking."""
    print("\n=== STATUS CHECKING DEMO ===")
    
    # Check overall dataset status
    overall_status = manager.get_validation_status()
    
    print("Overall Dataset Status:")
    print(f"  Total questions: {overall_status.get('total_questions', 'N/A')}")
    print(f"  Validated questions: {overall_status.get('validated_questions', 'N/A')}")
    print(f"  Validation coverage: {overall_status.get('validation_coverage', 0):.1%}")
    print(f"  Pending validations: {overall_status.get('pending_validations', 'N/A')}")
    print(f"  Average confidence: {overall_status.get('average_confidence', 0):.1%}")


def demo_integrity_testing(manager):
    """Demonstrate dataset integrity testing."""
    print("\n=== INTEGRITY TESTING DEMO ===")
    
    # Run integrity check
    integrity_report = manager.validate_dataset_integrity()
    
    print("Dataset Integrity Report:")
    print(f"  Total questions: {integrity_report.get('total_questions', 'N/A')}")
    print(f"  Issues found: {integrity_report.get('issues_found', 'N/A')}")
    
    if integrity_report.get('issues'):
        print("  Issues:")
        for issue in integrity_report['issues']:
            print(f"    - {issue}")
    else:
        print("  ✓ No integrity issues found")
    
    # Display statistics
    stats = integrity_report.get('statistics', {})
    if stats:
        print(f"\nDataset Statistics:")
        print(f"  Average question length: {stats.get('average_question_length', 'N/A'):.1f} chars")
        print(f"  Domain distribution: {len(stats.get('domain_distribution', {}))} domains")
        print(f"  Difficulty distribution: {stats.get('difficulty_distribution', {})}")


def demo_quality_metrics():
    """Demonstrate quality metrics and reporting."""
    print("\n=== QUALITY METRICS DEMO ===")
    
    # Initialize fresh manager for reporting
    manager = initialize_validation_system()
    
    # Check if we can generate reports (may be empty without actual validations)
    try:
        # Get basic quality metrics
        quality_metrics = manager.quality_metrics
        
        print("Quality Metrics:")
        print(f"  Last updated: {quality_metrics.get('last_updated', 'Never')}")
        print(f"  Domain distribution available: {'domain_distribution' in quality_metrics}")
        print(f"  Difficulty distribution available: {'difficulty_distribution' in quality_metrics}")
        
        # Show domain distribution if available
        domain_dist = quality_metrics.get('domain_distribution', {})
        if domain_dist:
            print(f"\nTop 5 domains by question count:")
            sorted_domains = sorted(domain_dist.items(), key=lambda x: x[1], reverse=True)
            for domain, count in sorted_domains[:5]:
                print(f"    {domain}: {count} questions")
    
    except Exception as e:
        print(f"Quality metrics demo encountered an issue: {e}")


def main():
    """Run the complete validation system demonstration."""
    print("🔬 PYNUCLEUS EXPERT VALIDATION SYSTEM DEMO")
    print("=" * 50)
    
    try:
        # Step 1: Expert Registration
        manager = demo_expert_registration()
        
        # Step 2: Task Assignment  
        assignments = demo_task_assignment(manager)
        
        # Step 3: Validation Submission
        demo_validation_submission(manager)
        
        # Step 4: Status Checking
        demo_status_checking(manager)
        
        # Step 5: Integrity Testing
        demo_integrity_testing(manager)
        
        # Step 6: Quality Metrics
        demo_quality_metrics()
        
        print("\n" + "=" * 50)
        print("🎉 VALIDATION SYSTEM DEMO COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Register real expert validators")
        print("2. Assign validation tasks to experts")
        print("3. Collect expert validations")
        print("4. Monitor quality metrics")
        print("5. Integrate feedback for continuous improvement")
        
        print(f"\nFor detailed usage, see: docs/GOLDEN_DATASET_VALIDATION_GUIDE.md")
        
    except Exception as e:
        print(f"\n❌ Demo encountered an error: {e}")
        print("Check the logs and ensure all dependencies are properly installed.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 