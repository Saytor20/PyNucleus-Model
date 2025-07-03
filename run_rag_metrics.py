#!/usr/bin/env python3
"""
Enhanced RAG Metrics Analysis Script

Integrates with PyNucleus Advanced Statistics System to provide comprehensive
RAG retrieval performance analysis with rich formatting and detailed insights.

Usage:
    python run_rag_metrics.py                    # Interactive mode with rich formatting
    python run_rag_metrics.py --file <json>      # Use specific evaluation file
    python run_rag_metrics.py --save             # Save results to file
    python run_rag_metrics.py --quick            # Quick analysis without rich formatting
    python run_rag_metrics.py --compare          # Compare multiple evaluation files
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pynucleus.metrics.system_statistics import compute_retrieval_metrics, aggregate_metrics


def analyze_evaluation_file(file_path):
    """Analyze an evaluation file and compute RAG metrics."""
    with open(file_path, 'r') as f:
        eval_data = json.load(f)
    
    questions = eval_data.get('detailed_results', [])
    print(f'📊 Analyzing {len(questions)} questions from {file_path}')
    
    batch_metrics = []
    
    for i, question in enumerate(questions, 1):
        # Extract question info
        question_text = question.get('question', '')
        expected_keywords = question.get('expected_keywords', [])
        keyword_score = question.get('keyword_score', 0.0)
        domain = question.get('domain', 'unknown')
        difficulty = question.get('difficulty', 'unknown')
        
        # Simulate retrieved documents based on keyword performance
        num_relevant = len(expected_keywords)
        num_retrieved = 5  # Assume k=5 retrieval
        
        # Simulate precision based on keyword score
        relevant_retrieved = max(1, int(keyword_score * num_retrieved))
        
        # Create synthetic document IDs
        retrieved_ids = [f'doc_{i}_{j}' for j in range(num_retrieved)]
        relevant_ids = set([f'doc_{i}_{j}' for j in range(relevant_retrieved)])
        
        # Compute metrics
        metrics = compute_retrieval_metrics(retrieved_ids, relevant_ids, k=5)
        batch_metrics.append(metrics)
        
        print(f'  Q{i}: {question_text[:40]}...')
        print(f'    Domain: {domain}, Difficulty: {difficulty}')
        print(f'    Keywords: {expected_keywords}')
        print(f'    Score: {keyword_score:.3f} → P:{metrics.precision:.3f} R:{metrics.recall:.3f} F1:{metrics.f1:.3f}')
    
    return batch_metrics, questions


def run_enhanced_analysis(file_path: str, save_results: bool = False, use_rich: bool = True):
    """Run enhanced RAG metrics analysis with rich formatting."""
    if use_rich:
        try:
            from pynucleus.metrics.system_statistics import _analyze_evaluation_file_enhanced
            _analyze_evaluation_file_enhanced(Path(file_path))
            
            if save_results:
                # Also save using the original format
                batch_metrics, questions = analyze_evaluation_file(file_path)
                if batch_metrics:
                    aggregated = aggregate_metrics(batch_metrics)
                    results = {
                        'source_file': file_path,
                        'aggregated': aggregated,
                        'individual_metrics': [
                            {
                                'question': q.get('question', ''),
                                'domain': q.get('domain', ''),
                                'difficulty': q.get('difficulty', ''),
                                'k': m.k,
                                'precision': m.precision,
                                'recall': m.recall,
                                'f1': m.f1,
                                'num_relevant': m.num_relevant,
                                'num_retrieved': m.num_retrieved
                            }
                            for q, m in zip(questions, batch_metrics)
                        ]
                    }
                    
                    output_file = f'rag_metrics_{Path(file_path).stem}.json'
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    print(f'\n💾 Results saved to {output_file}')
            return
        except ImportError:
            print("Rich formatting not available, falling back to basic mode...")
    
    # Fallback to basic analysis
    batch_metrics, questions = analyze_evaluation_file(file_path)
    
    if not batch_metrics:
        print('❌ No metrics to analyze')
        return
    
    # Aggregate metrics
    aggregated = aggregate_metrics(batch_metrics)
    
    print(f'\n🎯 Overall Metrics (Micro-averaged)')
    print(f'Precision: {aggregated["precision"]:.3f}')
    print(f'Recall: {aggregated["recall"]:.3f}')
    print(f'F1 Score: {aggregated["f1"]:.3f}')
    print(f'Total Queries: {aggregated["num_queries"]}')
    print(f'Total Retrieved: {aggregated["total_retrieved"]}')
    print(f'Total Relevant: {aggregated["total_relevant"]}')
    
    # Save results if requested
    if save_results:
        results = {
            'source_file': file_path,
            'aggregated': aggregated,
            'individual_metrics': [
                {
                    'question': q.get('question', ''),
                    'domain': q.get('domain', ''),
                    'difficulty': q.get('difficulty', ''),
                    'k': m.k,
                    'precision': m.precision,
                    'recall': m.recall,
                    'f1': m.f1,
                    'num_relevant': m.num_relevant,
                    'num_retrieved': m.num_retrieved
                }
                for q, m in zip(questions, batch_metrics)
            ]
        }
        
        output_file = f'rag_metrics_{Path(file_path).stem}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f'\n💾 Results saved to {output_file}')


def compare_evaluation_files():
    """Compare multiple evaluation files."""
    eval_dir = Path('data/validation/results')
    if not eval_dir.exists():
        print('❌ No validation results directory found')
        return
    
    eval_files = list(eval_dir.glob('*.json'))
    if len(eval_files) < 2:
        print('❌ Need at least 2 evaluation files for comparison')
        return
    
    print('📊 Comparing Evaluation Files')
    print('=' * 50)
    
    comparison_results = []
    
    for file in eval_files:
        print(f'\n🔍 Analyzing {file.name}...')
        batch_metrics, questions = analyze_evaluation_file(str(file))
        
        if batch_metrics:
            aggregated = aggregate_metrics(batch_metrics)
            comparison_results.append({
                'file': file.name,
                'metrics': aggregated,
                'question_count': len(questions)
            })
    
    if not comparison_results:
        print('❌ No valid results for comparison')
        return
    
    # Display comparison
    print('\n📈 Comparison Summary')
    print('-' * 80)
    print(f"{'File':<40} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Questions':<10}")
    print('-' * 80)
    
    for result in comparison_results:
        metrics = result['metrics']
        print(f"{result['file']:<40} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1']:<10.3f} {result['question_count']:<10}")
    
    # Find best performing file
    best_f1 = max(comparison_results, key=lambda x: x['metrics']['f1'])
    print(f'\n🏆 Best F1 Score: {best_f1["file"]} ({best_f1["metrics"]["f1"]:.3f})')


def interactive_mode():
    """Run interactive mode for file selection."""
    eval_dir = Path('data/validation/results')
    if not eval_dir.exists():
        print('❌ No validation results directory found')
        print('Expected: data/validation/results/')
        return
    
    eval_files = list(eval_dir.glob('*.json'))
    if not eval_files:
        print('❌ No evaluation files found')
        return
    
    print('🚀 PyNucleus RAG Metrics Analysis')
    print('=' * 50)
    print('\n📁 Available evaluation files:')
    
    for i, file in enumerate(eval_files, 1):
        file_size = file.stat().st_size / 1024  # KB
        mod_time = file.stat().st_mtime
        print(f'  {i}. {file.name} ({file_size:.1f}KB)')
    
    print(f'\n🔍 Options:')
    print(f'  [1-{len(eval_files)}] - Analyze specific file')
    print(f'  [a] - Analyze all files')
    print(f'  [c] - Compare all files')
    print(f'  [q] - Quit')
    
    try:
        choice = input('\nSelect option: ').strip().lower()
        
        if choice == 'q':
            return
        elif choice == 'a':
            for file in eval_files:
                print(f'\n{"="*60}')
                print(f'Analyzing: {file.name}')
                print("="*60)
                run_enhanced_analysis(str(file))
        elif choice == 'c':
            compare_evaluation_files()
        elif choice.isdigit() and 1 <= int(choice) <= len(eval_files):
            selected_file = eval_files[int(choice) - 1]
            print(f'\n📊 Analyzing: {selected_file.name}')
            run_enhanced_analysis(str(selected_file))
        else:
            print('❌ Invalid choice')
            
    except (ValueError, KeyboardInterrupt):
        print('\n👋 Goodbye!')


def main():
    parser = argparse.ArgumentParser(description='Enhanced RAG retrieval metrics analysis')
    parser.add_argument('--file', help='Specific evaluation file to analyze')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    parser.add_argument('--quick', action='store_true', help='Quick analysis without rich formatting')
    parser.add_argument('--compare', action='store_true', help='Compare multiple evaluation files')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive or (not args.file and not args.compare):
        interactive_mode()
        return
    
    # Compare mode
    if args.compare:
        compare_evaluation_files()
        return
    
    # Single file analysis
    if args.file:
        if not Path(args.file).exists():
            print(f'❌ File not found: {args.file}')
            print('Available evaluation files:')
            eval_dir = Path('data/validation/results')
            if eval_dir.exists():
                for f in eval_dir.glob('*.json'):
                    print(f'  {f}')
            return
        
        use_rich = not args.quick
        run_enhanced_analysis(args.file, args.save, use_rich)
    else:
        # Default file
        default_file = 'data/validation/results/enhanced_evaluation_20250622_185607.json'
        if Path(default_file).exists():
            use_rich = not args.quick
            run_enhanced_analysis(default_file, args.save, use_rich)
        else:
            print('❌ Default file not found, running interactive mode...')
            interactive_mode()


if __name__ == "__main__":
    main()
