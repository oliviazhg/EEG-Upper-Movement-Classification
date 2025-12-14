"""
Cross-Method Comparison Script
SYDE 522 Final Project

Compares CNN results with other classification methods (CSP+LDA, SVM, RF)
for the final project paper.

Usage:
    python compare_methods.py --methods cnn csp svm rf
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from pathlib import Path
from typing import Dict, List
import argparse


def load_cnn_results(results_dir: str = 'results/cnn') -> Dict:
    """Load and aggregate CNN results"""
    results_path = Path(results_dir) / 'aggregated_results.json'
    
    if not results_path.exists():
        raise FileNotFoundError(f"CNN results not found at {results_path}")
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Get best architecture
    best = max(data, key=lambda x: x['test_acc_mean'])
    
    return {
        'method': f"CNN-{best['architecture']}",
        'test_acc_mean': best['test_acc_mean'],
        'test_acc_ci_low': best['test_acc_ci_low'],
        'test_acc_ci_high': best['test_acc_ci_high'],
        'test_f1_mean': best['test_f1_mean'],
        'n_parameters': best['n_parameters'],
        'all_architectures': data
    }


def load_csp_results(results_dir: str = 'results/csp') -> Dict:
    """Load and aggregate CSP+LDA results"""
    results_path = Path(results_dir) / 'aggregated_results.json'
    
    if not results_path.exists():
        print(f"Warning: CSP results not found at {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Get best frequency band
    best = max(data, key=lambda x: x['test_acc_mean'])
    
    return {
        'method': f"CSP+LDA-{best['freq_band']}",
        'test_acc_mean': best['test_acc_mean'],
        'test_acc_ci_low': best['test_acc_ci_low'],
        'test_acc_ci_high': best['test_acc_ci_high'],
        'test_f1_mean': best['test_f1_mean'],
        'n_parameters': best.get('n_parameters', 'N/A'),
        'all_configs': data
    }


def load_svm_results(results_dir: str = 'results/svm') -> Dict:
    """Load SVM results (if available)"""
    results_path = Path(results_dir) / 'best_results.json'
    
    if not results_path.exists():
        print(f"Warning: SVM results not found at {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        best = json.load(f)
    
    return {
        'method': f"SVM (C={best.get('C', 'auto')}, γ={best.get('gamma', 'auto')})",
        'test_acc_mean': best['test_acc_mean'],
        'test_acc_ci_low': best['test_acc_ci_low'],
        'test_acc_ci_high': best['test_acc_ci_high'],
        'test_f1_mean': best['test_f1_mean'],
        'n_parameters': 'N/A',
    }


def load_rf_results(results_dir: str = 'results/random_forest') -> Dict:
    """Load Random Forest results (if available)"""
    results_path = Path(results_dir) / 'best_results.json'
    
    if not results_path.exists():
        print(f"Warning: Random Forest results not found at {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        best = json.load(f)
    
    return {
        'method': f"Random Forest (n={best.get('n_estimators', 100)})",
        'test_acc_mean': best['test_acc_mean'],
        'test_acc_ci_low': best['test_acc_ci_low'],
        'test_acc_ci_high': best['test_acc_ci_high'],
        'test_f1_mean': best['test_f1_mean'],
        'n_parameters': 'N/A',
    }


def plot_method_comparison(results: List[Dict], 
                          output_dir: str = 'figures'):
    """
    Create comparison plots across all methods
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    methods = [r['method'] for r in results]
    test_accs = [r['test_acc_mean'] * 100 for r in results]
    
    err_low = [(r['test_acc_mean'] - r['test_acc_ci_low']) * 100 for r in results]
    err_high = [(r['test_acc_ci_high'] - r['test_acc_mean']) * 100 for r in results]
    errors = np.array([err_low, err_high])
    
    # FIGURE 1: Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'][:len(methods)]
    
    bars = ax.bar(x, test_accs, yerr=errors, capsize=10,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, test_accs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Styling
    ax.set_xlabel('Classification Method', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Method Comparison: 11-Class Upper-Limb Movement Classification\n'
                'EEG-Based Real Movement Decoding',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11, rotation=15, ha='right')
    ax.set_ylim([0, 100])
    ax.axhline(y=100/11, color='red', linestyle='--', linewidth=1, 
               label='Chance level (9.09%)', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'method_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'method_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {output_path / 'method_comparison.png'}")
    plt.close()
    
    # FIGURE 2: Accuracy vs F1 scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    
    test_f1s = [r['test_f1_mean'] * 100 for r in results]
    
    scatter = ax.scatter(test_accs, test_f1s, c=colors, s=200, 
                        alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add labels
    for i, method in enumerate(methods):
        ax.annotate(method, 
                   (test_accs[i], test_f1s[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    # Diagonal line (acc = F1)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0, label='Acc = F1')
    
    ax.set_xlabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test F1 Score (%)', fontsize=13, fontweight='bold')
    ax.set_title('Accuracy vs F1 Score Comparison',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(output_path / 'accuracy_vs_f1.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'accuracy_vs_f1.pdf', bbox_inches='tight')
    print(f"Saved: {output_path / 'accuracy_vs_f1.png'}")
    plt.close()


def print_comparison_table(results: List[Dict]):
    """
    Print formatted comparison table for paper
    """
    print(f"\n{'='*80}")
    print(f"{'METHOD COMPARISON TABLE':^80}")
    print(f"{'='*80}\n")
    
    # Header
    print(f"{'Method':<25} {'Test Acc (%)':<25} {'Test F1 (%)':<15} {'Params':<15}")
    print(f"{'-'*80}")
    
    # Rows
    for r in results:
        acc_str = f"{r['test_acc_mean']*100:.2f} [{r['test_acc_ci_low']*100:.2f}, {r['test_acc_ci_high']*100:.2f}]"
        f1_str = f"{r['test_f1_mean']*100:.2f}"
        params_str = f"{r['n_parameters']:,}" if isinstance(r['n_parameters'], int) else r['n_parameters']
        
        print(f"{r['method']:<25} {acc_str:<25} {f1_str:<15} {params_str:<15}")
    
    print(f"{'-'*80}\n")
    
    # Statistical comparison
    print(f"{'Statistical Comparison (Paired t-tests)':^80}")
    print(f"{'-'*80}")
    
    # Find best method
    best_idx = np.argmax([r['test_acc_mean'] for r in results])
    best_method = results[best_idx]['method']
    
    print(f"Best method: {best_method} ({results[best_idx]['test_acc_mean']*100:.2f}%)")
    
    # Check if CIs overlap
    print(f"\nConfidence Interval Analysis:")
    for i, r in enumerate(results):
        if i == best_idx:
            continue
        
        best_ci_low = results[best_idx]['test_acc_ci_low']
        r_ci_high = r['test_acc_ci_high']
        
        if best_ci_low > r_ci_high:
            print(f"  {best_method} > {r['method']} (non-overlapping CIs, p < 0.05)")
        else:
            print(f"  {best_method} ≈ {r['method']} (overlapping CIs, not significant)")
    
    print(f"\n{'='*80}\n")


def main():
    """Main comparison pipeline"""
    parser = argparse.ArgumentParser(description='Compare classification methods')
    parser.add_argument('--methods', nargs='+', default=['cnn', 'csp'],
                       help='Methods to compare (cnn, csp, svm, rf)')
    parser.add_argument('--output_dir', default='figures',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"{'CROSS-METHOD COMPARISON':^80}")
    print(f"{'='*80}\n")
    
    # Load results
    results = []
    
    for method in args.methods:
        print(f"Loading {method.upper()} results...")
        try:
            if method.lower() == 'cnn':
                res = load_cnn_results('results/cnn')
            elif method.lower() == 'csp':
                res = load_csp_results('results/csp')
            elif method.lower() == 'svm':
                res = load_svm_results('results/svm')
            elif method.lower() == 'rf':
                res = load_rf_results('results/random_forest')
            else:
                print(f"  Unknown method: {method}")
                continue
            
            if res is not None:
                results.append(res)
                print(f"  ✓ Loaded {res['method']}")
            
        except Exception as e:
            print(f"  ✗ Failed to load {method}: {e}")
    
    if len(results) < 2:
        print("\nError: Need at least 2 methods to compare")
        print("Run experiments for more methods first")
        return
    
    # Sort by accuracy
    results = sorted(results, key=lambda x: x['test_acc_mean'], reverse=True)
    
    # Print table
    print_comparison_table(results)
    
    # Generate plots
    print("Generating comparison plots...")
    plot_method_comparison(results, output_dir=args.output_dir)
    
    print(f"\n{'='*80}")
    print(f"Comparison complete! Check {args.output_dir}/ for figures")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
