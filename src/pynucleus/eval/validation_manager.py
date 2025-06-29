"""
Expert Validation Manager for Golden Dataset Curation and Quality Assurance.

This module provides comprehensive workflows for:
- Expert validation of question-answer pairs
- Automated review scheduling and tracking
- Feedback collection and integration
- Dataset quality metrics and monitoring
- Continuous improvement workflows
"""

import pandas as pd
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from ..utils.logger import logger


class ValidationStatus(Enum):
    """Status of validation for question-answer pairs."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    EXPERT_REVIEW = "expert_review"


class ExpertLevel(Enum):
    """Expert qualification levels."""
    JUNIOR = "junior"
    SENIOR = "senior"
    PRINCIPAL = "principal"
    DOMAIN_EXPERT = "domain_expert"


@dataclass
class ExpertProfile:
    """Expert profile for validation assignments."""
    expert_id: str
    name: str
    email: str
    level: ExpertLevel
    domains: List[str]
    specializations: List[str]
    validation_count: int = 0
    accuracy_score: float = 0.0
    last_activity: Optional[datetime] = None


@dataclass
class ValidationRecord:
    """Record of expert validation for a question-answer pair."""
    question_id: str
    expert_id: str
    validation_date: datetime
    status: ValidationStatus
    confidence_score: float  # 0.0 to 1.0
    feedback: str
    suggested_improvements: str
    domain_accuracy: float
    technical_accuracy: float
    clarity_score: float
    version_hash: str


class ValidationManager:
    """Manages expert validation workflows and dataset quality assurance."""
    
    def __init__(self, dataset_path: str = "data/validation/golden_dataset.csv",
                 validation_dir: str = "data/validation/expert_reviews"):
        self.dataset_path = Path(dataset_path)
        self.validation_dir = Path(validation_dir)
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage files
        self.experts_file = self.validation_dir / "experts.json"
        self.validations_file = self.validation_dir / "validations.json"
        self.quality_metrics_file = self.validation_dir / "quality_metrics.json"
        self.pending_reviews_file = self.validation_dir / "pending_reviews.json"
        
        # Load existing data
        self.experts = self._load_experts()
        self.validations = self._load_validations()
        self.quality_metrics = self._load_quality_metrics()
        
        logger.info("Validation Manager initialized")
    
    def register_expert(self, expert_profile: ExpertProfile) -> bool:
        """Register a new expert for validation tasks."""
        try:
            if expert_profile.expert_id in self.experts:
                logger.warning(f"Expert {expert_profile.expert_id} already registered")
                return False
            
            self.experts[expert_profile.expert_id] = asdict(expert_profile)
            self._save_experts()
            
            logger.info(f"Expert {expert_profile.name} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register expert: {e}")
            return False
    
    def assign_validation_tasks(self, max_assignments_per_expert: int = 10) -> Dict[str, List[str]]:
        """Assign validation tasks to experts based on their domains and workload."""
        try:
            # Load current dataset
            df = pd.read_csv(self.dataset_path)
            
            # Get questions needing validation
            pending_questions = self._get_pending_validations(df)
            
            # Assignment algorithm
            assignments = {}
            expert_workloads = {eid: 0 for eid in self.experts.keys()}
            
            for _, question_row in pending_questions.iterrows():
                question_id = self._generate_question_id(question_row)
                domain = question_row.get('domain', 'general')
                difficulty = question_row.get('difficulty', 'medium')
                
                # Find suitable experts
                suitable_experts = self._find_suitable_experts(domain, difficulty)
                
                # Assign to expert with lowest current workload
                if suitable_experts:
                    expert_id = min(suitable_experts, 
                                  key=lambda x: expert_workloads[x])
                    
                    if expert_workloads[expert_id] < max_assignments_per_expert:
                        if expert_id not in assignments:
                            assignments[expert_id] = []
                        assignments[expert_id].append(question_id)
                        expert_workloads[expert_id] += 1
            
            # Save assignment records
            self._save_pending_assignments(assignments)
            
            logger.info(f"Assigned {sum(len(tasks) for tasks in assignments.values())} validation tasks")
            return assignments
            
        except Exception as e:
            logger.error(f"Failed to assign validation tasks: {e}")
            return {}
    
    def submit_validation(self, validation_record: ValidationRecord) -> bool:
        """Submit expert validation for a question-answer pair."""
        try:
            # Validate input
            if not self._validate_submission(validation_record):
                return False
            
            # Store validation record
            record_id = f"{validation_record.question_id}_{validation_record.expert_id}_{validation_record.validation_date.isoformat()}"
            self.validations[record_id] = asdict(validation_record)
            
            # Update expert activity
            if validation_record.expert_id in self.experts:
                self.experts[validation_record.expert_id]['validation_count'] += 1
                self.experts[validation_record.expert_id]['last_activity'] = datetime.now().isoformat()
            
            # Save updates
            self._save_validations()
            self._save_experts()
            
            # Update quality metrics
            self._update_quality_metrics()
            
            logger.info(f"Validation submitted by {validation_record.expert_id} for question {validation_record.question_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit validation: {e}")
            return False
    
    def get_validation_status(self, question_id: Optional[str] = None) -> Dict[str, Any]:
        """Get validation status for specific question or overall dataset."""
        try:
            if question_id:
                # Status for specific question
                question_validations = [
                    v for v in self.validations.values()
                    if v['question_id'] == question_id
                ]
                
                if not question_validations:
                    return {"status": "not_validated", "validations": []}
                
                # Determine consensus
                statuses = [v['status'] for v in question_validations]
                consensus_status = self._determine_consensus_status(statuses)
                
                return {
                    "status": consensus_status,
                    "validations": question_validations,
                    "expert_count": len(question_validations),
                    "consensus_score": self._calculate_consensus_score(question_validations)
                }
            else:
                # Overall dataset status
                return {
                    "total_questions": self.quality_metrics.get('total_questions', 0),
                    "validated_questions": self.quality_metrics.get('validated_questions', 0),
                    "validation_coverage": self.quality_metrics.get('validation_coverage', 0.0),
                    "pending_validations": self.quality_metrics.get('pending_validations', 0),
                    "average_confidence": self.quality_metrics.get('average_expert_confidence', 0.0)
                }
                
        except Exception as e:
            logger.error(f"Failed to get validation status: {e}")
            return {"error": str(e)}
    
    def validate_dataset_integrity(self) -> Dict[str, Any]:
        """Validate dataset integrity and identify issues."""
        try:
            df = pd.read_csv(self.dataset_path)
            issues = []
            
            # Check for missing values
            missing_values = df.isnull().sum()
            for column, count in missing_values.items():
                if count > 0:
                    issues.append(f"Missing values in {column}: {count}")
            
            # Check for duplicate questions
            duplicates = df.duplicated(subset=['question']).sum()
            if duplicates > 0:
                issues.append(f"Duplicate questions found: {duplicates}")
            
            # Check domain consistency
            valid_domains = self._get_valid_domains()
            invalid_domains = df[~df['domain'].isin(valid_domains)]['domain'].unique()
            if len(invalid_domains) > 0:
                issues.append(f"Invalid domains found: {list(invalid_domains)}")
            
            # Check difficulty levels
            valid_difficulties = ['easy', 'medium', 'hard']
            invalid_difficulties = df[~df['difficulty'].isin(valid_difficulties)]['difficulty'].unique()
            if len(invalid_difficulties) > 0:
                issues.append(f"Invalid difficulty levels: {list(invalid_difficulties)}")
            
            # Check question length distribution
            question_lengths = df['question'].str.len()
            very_short = (question_lengths < 10).sum()
            very_long = (question_lengths > 200).sum()
            
            integrity_report = {
                "timestamp": datetime.now().isoformat(),
                "total_questions": len(df),
                "issues_found": len(issues),
                "issues": issues,
                "statistics": {
                    "very_short_questions": very_short,
                    "very_long_questions": very_long,
                    "average_question_length": question_lengths.mean(),
                    "domain_distribution": df['domain'].value_counts().to_dict(),
                    "difficulty_distribution": df['difficulty'].value_counts().to_dict()
                }
            }
            
            logger.info(f"Dataset integrity validated. Found {len(issues)} issues.")
            return integrity_report
            
        except Exception as e:
            logger.error(f"Failed to validate dataset integrity: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    
    def _load_experts(self) -> Dict[str, Any]:
        """Load expert profiles from storage."""
        if self.experts_file.exists():
            with open(self.experts_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_experts(self):
        """Save expert profiles to storage."""
        with open(self.experts_file, 'w') as f:
            json.dump(self.experts, f, indent=2, default=str)
    
    def _load_validations(self) -> Dict[str, Any]:
        """Load validation records from storage."""
        if self.validations_file.exists():
            with open(self.validations_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_validations(self):
        """Save validation records to storage."""
        with open(self.validations_file, 'w') as f:
            json.dump(self.validations, f, indent=2, default=str)
    
    def _load_quality_metrics(self) -> Dict[str, Any]:
        """Load quality metrics from storage."""
        if self.quality_metrics_file.exists():
            with open(self.quality_metrics_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _generate_question_id(self, question_row: pd.Series) -> str:
        """Generate unique ID for a question."""
        content = f"{question_row['question']}_{question_row.get('expected_answer', '')}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_pending_validations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get questions that need validation."""
        validated_question_ids = set(v['question_id'] for v in self.validations.values())
        
        pending_questions = []
        for _, row in df.iterrows():
            question_id = self._generate_question_id(row)
            if question_id not in validated_question_ids:
                pending_questions.append(row)
        
        return pd.DataFrame(pending_questions)
    
    def _find_suitable_experts(self, domain: str, difficulty: str) -> List[str]:
        """Find experts suitable for validating a question."""
        suitable_experts = []
        
        for expert_id, expert_data in self.experts.items():
            domains = expert_data.get('domains', [])
            level = expert_data.get('level', 'junior')
            
            # Check domain match
            if domain in domains or 'general' in domains:
                # Check level appropriateness for difficulty
                if self._expert_suitable_for_difficulty(level, difficulty):
                    suitable_experts.append(expert_id)
        
        return suitable_experts
    
    def _expert_suitable_for_difficulty(self, expert_level: str, difficulty: str) -> bool:
        """Check if expert level is suitable for question difficulty."""
        level_hierarchy = {
            'junior': ['easy'],
            'senior': ['easy', 'medium'],
            'principal': ['easy', 'medium', 'hard'],
            'domain_expert': ['easy', 'medium', 'hard']
        }
        
        return difficulty in level_hierarchy.get(expert_level, [])
    
    def _determine_consensus_status(self, statuses: List[str]) -> str:
        """Determine consensus status from multiple validations."""
        if not statuses:
            return "not_validated"
        
        # Count status occurrences
        status_counts = {}
        for status in statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Find most common status
        max_count = max(status_counts.values())
        consensus_statuses = [s for s, c in status_counts.items() if c == max_count]
        
        if len(consensus_statuses) == 1:
            return consensus_statuses[0]
        else:
            return "conflicting_validations"
    
    def _calculate_consensus_score(self, validations: List[Dict]) -> float:
        """Calculate consensus score for validations."""
        if len(validations) <= 1:
            return 1.0 if validations else 0.0
        
        confidences = [v['confidence_score'] for v in validations]
        statuses = [v['status'] for v in validations]
        
        # Consensus based on status agreement and confidence alignment
        status_agreement = len(set(statuses)) == 1
        confidence_variance = sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences)
        
        if status_agreement:
            return max(0.5, 1.0 - confidence_variance)
        else:
            return min(0.5, 1.0 - confidence_variance)
    
    def _get_valid_domains(self) -> List[str]:
        """Get list of valid domains."""
        return [
            'separation', 'reaction_engineering', 'energy_efficiency', 'plant_design',
            'safety_management', 'thermodynamics', 'fluid_mechanics', 'heat_transfer',
            'equipment_standards', 'logistics', 'environmental', 'drying_processes',
            'economics', 'sustainability', 'mass_transfer', 'utilities', 'equipment_selection',
            'separation_process', 'maintenance', 'crystallization', 'information_retrieval',
            'adsorption', 'membrane_separation', 'fluidization', 'water_treatment',
            'filtration', 'heat_mass_transfer', 'distillation', 'material_selection'
        ]
    
    def _save_pending_assignments(self, assignments: Dict[str, List[str]]):
        """Save pending assignment records."""
        assignment_record = {
            "timestamp": datetime.now().isoformat(),
            "assignments": assignments
        }
        
        with open(self.pending_reviews_file, 'w') as f:
            json.dump(assignment_record, f, indent=2)
    
    def _validate_submission(self, validation_record: ValidationRecord) -> bool:
        """Validate submission data."""
        if not validation_record.question_id:
            logger.error("Missing question_id in validation submission")
            return False
        
        if not validation_record.expert_id:
            logger.error("Missing expert_id in validation submission")
            return False
        
        if validation_record.expert_id not in self.experts:
            logger.error(f"Unknown expert_id: {validation_record.expert_id}")
            return False
        
        if not (0.0 <= validation_record.confidence_score <= 1.0):
            logger.error("Confidence score must be between 0.0 and 1.0")
            return False
        
        return True
    
    def _update_quality_metrics(self):
        """Update quality metrics based on current validations."""
        try:
            df = pd.read_csv(self.dataset_path)
            
            # Calculate metrics
            total_questions = len(df)
            validated_questions = len(set(v['question_id'] for v in self.validations.values()))
            validation_coverage = validated_questions / total_questions if total_questions > 0 else 0
            
            confidences = [v['confidence_score'] for v in self.validations.values()]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            pending_validations = total_questions - validated_questions
            
            self.quality_metrics = {
                "total_questions": total_questions,
                "validated_questions": validated_questions,
                "validation_coverage": validation_coverage,
                "average_expert_confidence": avg_confidence,
                "domain_distribution": df['domain'].value_counts().to_dict(),
                "difficulty_distribution": df['difficulty'].value_counts().to_dict(),
                "pending_validations": pending_validations,
                "last_updated": datetime.now().isoformat()
            }
            
            # Save metrics
            with open(self.quality_metrics_file, 'w') as f:
                json.dump(self.quality_metrics, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to update quality metrics: {e}")


def initialize_validation_system() -> ValidationManager:
    """Initialize validation system with sample experts."""
    manager = ValidationManager()
    logger.info("Validation system initialized")
    return manager
