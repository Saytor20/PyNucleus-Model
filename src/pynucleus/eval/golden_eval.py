import pandas as pd
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from ..rag.engine import ask
from ..utils.logger import logger
from ..metrics import Metrics

CSV_PATH = "data/validation/golden_dataset.csv"


@dataclass
class EvaluationResult:
    """Detailed evaluation result for a single question."""
    question: str
    domain: str
    difficulty: str
    expected_keywords: List[str]
    generated_answer: str
    sources_used: List[str]
    keyword_matches: List[str]
    keyword_score: float
    response_time: float
    answer_length: int
    confidence_score: float
    success: bool
    error_message: str = ""


class EnhancedEvaluator:
    """Enhanced evaluation system with comprehensive metrics and analysis."""
    
    def __init__(self):
        self.results: List[EvaluationResult] = []
        self.start_time = None
        self.end_time = None
    
    def run_comprehensive_eval(self, threshold: float = 0.7, sample_size: Optional[int] = None, 
                             save_results: bool = True) -> Dict[str, Any]:
        """Run comprehensive evaluation with detailed analysis."""
        logger.info("🔍 Starting comprehensive evaluation...")
        self.start_time = datetime.now()
        
        # Load dataset
        try:
            df = pd.read_csv(CSV_PATH)
        except FileNotFoundError:
            logger.error(f"Golden dataset not found at {CSV_PATH}")
            return {"error": "Dataset not found", "success": False}
        
        # Random sampling if specified
        original_size = len(df)
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Randomly sampled {sample_size} questions from {original_size} total questions")
        
        # Process each question
        self.results = []
        for idx, row in df.iterrows():
            result = self._evaluate_single_question(row)
            self.results.append(result)
            
            # Record in global metrics
            Metrics.record_query(
                question=result.question,
                response_time=result.response_time,
                sources_count=len(result.sources_used),
                answer_length=result.answer_length,
                confidence_score=result.confidence_score,
                domain=result.domain,
                success=result.success
            )
        
        self.end_time = datetime.now()
        
        # Generate comprehensive analysis
        analysis = self._generate_comprehensive_analysis()
        
        # Save results if requested
        if save_results:
            self._save_detailed_results(analysis)
        
        # Determine overall success
        overall_success = analysis["overall_metrics"]["success_rate"] >= threshold
        
        logger.info(f"Evaluation completed: {analysis['overall_metrics']['success_rate']:.1%} success rate")
        
        return {
            "success": overall_success,
            "threshold_met": overall_success,
            "analysis": analysis,
            "sample_info": {
                "total_questions": len(self.results),
                "original_dataset_size": original_size,
                "sample_size": sample_size
            }
        }
    
    def _evaluate_single_question(self, row: pd.Series) -> EvaluationResult:
        """Evaluate a single question with detailed metrics."""
        question = row["question"]
        expected_keywords = [kw.strip().lower() for kw in row["expected_keywords"].split(",")]
        domain = row.get("domain", "general")
        difficulty = row.get("difficulty", "medium")
        
        # Time the request
        start_time = time.time()
        try:
            response = ask(question)
            response_time = time.time() - start_time
            
            answer = response.get("answer", "")
            sources = response.get("sources", [])
            
            # Calculate metrics
            keyword_matches = [kw for kw in expected_keywords if kw in answer.lower()]
            keyword_score = len(keyword_matches) / len(expected_keywords) if expected_keywords else 0
            
            # Calculate confidence score based on multiple factors
            confidence_score = self._calculate_confidence_score(
                keyword_score, len(sources), len(answer), response_time
            )
            
            success = keyword_score >= 0.5  # At least 50% keywords matched
            
            return EvaluationResult(
                question=question,
                domain=domain,
                difficulty=difficulty,
                expected_keywords=expected_keywords,
                generated_answer=answer,
                sources_used=sources,
                keyword_matches=keyword_matches,
                keyword_score=keyword_score,
                response_time=response_time,
                answer_length=len(answer),
                confidence_score=confidence_score,
                success=success
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Failed to evaluate question '{question}': {e}")
            
            return EvaluationResult(
                question=question,
                domain=domain,
                difficulty=difficulty,
                expected_keywords=expected_keywords,
                generated_answer="",
                sources_used=[],
                keyword_matches=[],
                keyword_score=0.0,
                response_time=response_time,
                answer_length=0,
                confidence_score=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _calculate_confidence_score(self, keyword_score: float, sources_count: int, 
                                  answer_length: int, response_time: float) -> float:
        """Calculate confidence score based on multiple factors."""
        # Keyword match weight (50%)
        keyword_weight = keyword_score * 0.5
        
        # Source quality weight (25%)
        source_weight = min(sources_count / 3.0, 1.0) * 0.25
        
        # Answer completeness weight (15%)
        completeness_weight = min(answer_length / 200.0, 1.0) * 0.15
        
        # Response time weight (10%) - faster is slightly better
        time_weight = max(0, 1.0 - (response_time / 10.0)) * 0.1
        
        return round(keyword_weight + source_weight + completeness_weight + time_weight, 3)
    
    def _generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of evaluation results."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        # Overall metrics
        total_questions = len(self.results)
        successful_questions = sum(1 for r in self.results if r.success)
        success_rate = successful_questions / total_questions
        
        # Performance metrics
        avg_response_time = sum(r.response_time for r in self.results) / total_questions
        avg_keyword_score = sum(r.keyword_score for r in self.results) / total_questions
        avg_confidence = sum(r.confidence_score for r in self.results) / total_questions
        avg_answer_length = sum(r.answer_length for r in self.results) / total_questions
        avg_sources_used = sum(len(r.sources_used) for r in self.results) / total_questions
        
        # Domain analysis
        domain_stats = self._analyze_by_category("domain")
        difficulty_stats = self._analyze_by_category("difficulty")
        
        # Quality distribution
        quality_distribution = self._analyze_quality_distribution()
        
        # Performance trends
        performance_trends = self._analyze_performance_trends()
        
        # Error analysis
        error_analysis = self._analyze_errors()
        
        return {
            "evaluation_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "total_duration_seconds": total_duration,
                "questions_per_second": round(total_questions / total_duration, 2)
            },
            "overall_metrics": {
                "total_questions": total_questions,
                "successful_questions": successful_questions,
                "failed_questions": total_questions - successful_questions,
                "success_rate": round(success_rate, 3),
                "avg_response_time": round(avg_response_time, 3),
                "avg_keyword_score": round(avg_keyword_score, 3),
                "avg_confidence_score": round(avg_confidence, 3),
                "avg_answer_length": round(avg_answer_length, 1),
                "avg_sources_used": round(avg_sources_used, 1)
            },
            "domain_analysis": domain_stats,
            "difficulty_analysis": difficulty_stats,
            "quality_distribution": quality_distribution,
            "performance_trends": performance_trends,
            "error_analysis": error_analysis,
            "recommendations": self._generate_recommendations()
        }
    
    def _analyze_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Analyze results by category (domain or difficulty)."""
        category_stats = {}
        
        for result in self.results:
            cat_value = getattr(result, category, "unknown")
            
            if cat_value not in category_stats:
                category_stats[cat_value] = {
                    "questions": 0,
                    "successful": 0,
                    "total_response_time": 0,
                    "total_keyword_score": 0,
                    "total_confidence": 0
                }
            
            stats = category_stats[cat_value]
            stats["questions"] += 1
            if result.success:
                stats["successful"] += 1
            stats["total_response_time"] += result.response_time
            stats["total_keyword_score"] += result.keyword_score
            stats["total_confidence"] += result.confidence_score
        
        # Calculate averages
        for cat_value, stats in category_stats.items():
            stats["success_rate"] = round(stats["successful"] / stats["questions"], 3)
            stats["avg_response_time"] = round(stats["total_response_time"] / stats["questions"], 3)
            stats["avg_keyword_score"] = round(stats["total_keyword_score"] / stats["questions"], 3)
            stats["avg_confidence"] = round(stats["total_confidence"] / stats["questions"], 3)
            
            # Remove totals for cleaner output
            del stats["total_response_time"]
            del stats["total_keyword_score"]
            del stats["total_confidence"]
        
        return category_stats
    
    def _analyze_quality_distribution(self) -> Dict[str, Any]:
        """Analyze quality score distribution."""
        confidence_scores = [r.confidence_score for r in self.results]
        keyword_scores = [r.keyword_score for r in self.results]
        
        return {
            "confidence_scores": {
                "excellent": sum(1 for s in confidence_scores if s >= 0.8),
                "good": sum(1 for s in confidence_scores if 0.6 <= s < 0.8),
                "fair": sum(1 for s in confidence_scores if 0.4 <= s < 0.6),
                "poor": sum(1 for s in confidence_scores if s < 0.4)
            },
            "keyword_accuracy": {
                "perfect": sum(1 for s in keyword_scores if s == 1.0),
                "high": sum(1 for s in keyword_scores if 0.8 <= s < 1.0),
                "medium": sum(1 for s in keyword_scores if 0.5 <= s < 0.8),
                "low": sum(1 for s in keyword_scores if s < 0.5)
            }
        }
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(self.results) < 5:
            return {"message": "Insufficient data for trend analysis"}
        
        # Split into batches to see if performance changes over time
        batch_size = max(1, len(self.results) // 5)
        batches = [self.results[i:i + batch_size] for i in range(0, len(self.results), batch_size)]
        
        batch_stats = []
        for i, batch in enumerate(batches):
            if not batch:
                continue
                
            avg_time = sum(r.response_time for r in batch) / len(batch)
            success_rate = sum(1 for r in batch if r.success) / len(batch)
            
            batch_stats.append({
                "batch": i + 1,
                "questions": len(batch),
                "avg_response_time": round(avg_time, 3),
                "success_rate": round(success_rate, 3)
            })
        
        # Determine trend
        if len(batch_stats) >= 2:
            first_half_time = sum(b["avg_response_time"] for b in batch_stats[:len(batch_stats)//2])
            second_half_time = sum(b["avg_response_time"] for b in batch_stats[len(batch_stats)//2:])
            
            trend = "improving" if second_half_time < first_half_time else "stable"
        else:
            trend = "stable"
        
        return {
            "batch_performance": batch_stats,
            "performance_trend": trend
        }
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns."""
        errors = [r for r in self.results if not r.success]
        
        if not errors:
            return {"message": "No errors to analyze"}
        
        error_types = {}
        low_keyword_scores = []
        
        for error in errors:
            if error.error_message:
                error_types[error.error_message] = error_types.get(error.error_message, 0) + 1
            else:
                low_keyword_scores.append(error.keyword_score)
        
        return {
            "total_errors": len(errors),
            "error_rate": round(len(errors) / len(self.results), 3),
            "error_types": error_types,
            "low_quality_answers": len(low_keyword_scores),
            "avg_failed_keyword_score": round(sum(low_keyword_scores) / len(low_keyword_scores), 3) if low_keyword_scores else 0
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Performance recommendations
        avg_response_time = sum(r.response_time for r in self.results) / len(self.results)
        if avg_response_time > 2.0:
            recommendations.append("⚡ Consider optimizing retrieval speed - average response time is above 2 seconds")
        
        # Quality recommendations
        avg_confidence = sum(r.confidence_score for r in self.results) / len(self.results)
        if avg_confidence < 0.7:
            recommendations.append("📚 Consider expanding document collection - confidence scores are below optimal")
        
        # Source utilization
        avg_sources = sum(len(r.sources_used) for r in self.results) / len(self.results)
        if avg_sources < 2:
            recommendations.append("🔍 Increase retrieval k parameter - using fewer than 2 sources on average")
        
        # Domain-specific issues
        domain_stats = self._analyze_by_category("domain")
        weak_domains = [domain for domain, stats in domain_stats.items() if stats["success_rate"] < 0.6]
        if weak_domains:
            recommendations.append(f"📖 Focus on improving {', '.join(weak_domains)} domain knowledge")
        
        # Error patterns
        error_analysis = self._analyze_errors()
        if error_analysis.get("error_rate", 0) > 0.1:
            recommendations.append("🛠️ Address system errors - error rate exceeds 10%")
        
        return recommendations
    
    def _save_detailed_results(self, analysis: Dict[str, Any]):
        """Save detailed evaluation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"data/validation/results/enhanced_evaluation_{timestamp}.json"
        
        # Ensure directory exists
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare detailed data
        detailed_results = {
            "metadata": {
                "timestamp": timestamp,
                "evaluation_type": "enhanced_comprehensive",
                "tool_version": "enhanced_evaluator_v1.0"
            },
            "analysis": analysis,
            "detailed_results": [
                {
                    "question": r.question,
                    "domain": r.domain,
                    "difficulty": r.difficulty,
                    "expected_keywords": r.expected_keywords,
                    "keyword_matches": r.keyword_matches,
                    "keyword_score": r.keyword_score,
                    "response_time": r.response_time,
                    "answer_length": r.answer_length,
                    "sources_count": len(r.sources_used),
                    "confidence_score": r.confidence_score,
                    "success": r.success,
                    "error_message": r.error_message
                }
                for r in self.results
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        logger.info(f"📊 Detailed evaluation results saved to: {results_file}")


def run_eval(threshold=0.7, sample_size=None):
    """Legacy function for backward compatibility."""
    evaluator = EnhancedEvaluator()
    result = evaluator.run_comprehensive_eval(threshold, sample_size, save_results=False)
    return result.get("success", False)


def run_enhanced_eval(threshold=0.7, sample_size=None, save_results=True):
    """Run enhanced evaluation with comprehensive analysis."""
    evaluator = EnhancedEvaluator()
    return evaluator.run_comprehensive_eval(threshold, sample_size, save_results) 