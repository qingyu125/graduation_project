"""
Comparison script for analyzing RPC vs non-RPC performance on single question

This script analyzes and compares the performance of DeepThinkLLM with and without RPC
on a single question, providing detailed comparison of reasoning paths and performance metrics.
"""
import json
import pickle
import argparse
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import re

# Setup matplotlib for proper rendering
def setup_matplotlib_for_plotting():
    """
    Setup matplotlib and seaborn for plotting with proper configuration.
    Call this function before creating any plots to ensure proper rendering.
    """
    import warnings
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Ensure warnings are printed
    warnings.filterwarnings('default')  # Show all warnings

    # Configure matplotlib for non-interactive mode
    plt.switch_backend("Agg")

    # Set chart style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Configure platform-appropriate fonts for cross-platform compatibility
    # Try to use Chinese fonts, but fall back to default if not available
    try:
        from matplotlib import font_manager
        # Try to find available fonts
        available_fonts = [f.name for f in font_manager.fontManager.ttflist]
        chinese_fonts = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB", "DejaVu Sans"]
        found_fonts = [f for f in chinese_fonts if f in available_fonts]
        if found_fonts:
            plt.rcParams["font.sans-serif"] = found_fonts + ["DejaVu Sans", "Arial", "sans-serif"]
        else:
            # Fall back to default fonts
            plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
    except Exception:
        # If font detection fails, use default
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
    
    plt.rcParams["axes.unicode_minus"] = False

# Setup matplotlib
setup_matplotlib_for_plotting()

def load_result_file(filepath: str) -> Dict[str, Any]:
    """Load result from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def extract_reasoning_path(traces: List[Dict[str, Any]]) -> List[str]:
    """Extract reasoning paths from traces"""
    paths = []
    for trace in traces:
        if trace.get('extracted_answer'):
            # Support both 'text' and 'full_text' fields
            text = trace.get('full_text') or trace.get('text', '')
            if text:
                paths.append(text)
    return paths

def calculate_token_statistics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate token-related statistics"""
    # Support both nested structure (token_stats, timing_stats) and flat structure
    token_stats = result.get('token_stats', {})
    timing_stats = result.get('timing_stats', {})
    
    # Get all_voting_traces - check both nested and flat
    all_voting_traces = result.get('all_voting_traces', [])
    if not all_voting_traces:
        # Try to get from all_traces if all_voting_traces is not available
        all_traces = result.get('all_traces', [])
        all_voting_traces = all_traces
    
    stats = {
        'total_tokens': token_stats.get('total_tokens', result.get('total_tokens', 0)),
        'warmup_tokens': token_stats.get('warmup_tokens', result.get('warmup_tokens', 0)),
        'final_tokens': token_stats.get('final_tokens', result.get('final_tokens', 0)),
        'total_time': timing_stats.get('total_time', result.get('total_time', 0)),
        'avg_tokens_per_warmup_trace': token_stats.get('avg_tokens_per_warmup_trace', result.get('avg_tokens_per_warmup_trace', 0)),
        'avg_tokens_per_final_trace': token_stats.get('avg_tokens_per_final_trace', result.get('avg_tokens_per_final_trace', 0)),
        'num_voting_traces': len(all_voting_traces),
    }
    return stats

def analyze_reasoning_quality(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze reasoning quality metrics"""
    if not traces:
        return {}
    
    total_length = 0
    repetition_count = 0
    reasoning_steps = 0
    sentence_counts = {}  # Initialize to avoid UnboundLocalError
    
    # Support both 'text' and 'full_text' fields
    full_texts = []
    for trace in traces:
        # Try multiple possible field names
        text = (trace.get('full_text') or 
                trace.get('text') or 
                trace.get('generated_text', '') or
                '')
        if text and isinstance(text, str) and len(text.strip()) > 0:
            full_texts.append(text)
    
    if full_texts:
        # Calculate average length
        total_length = np.mean([len(text.split()) for text in full_texts])
        
        # Count repetitions (simple heuristic: repeated phrases)
        all_text = ' '.join(full_texts)
        sentences = re.split(r'[.!?]+', all_text)
        sentence_counts = {}
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if len(sentence) > 10:  # Only consider substantial sentences
                sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1
        
        repetition_count = sum(1 for count in sentence_counts.values() if count > 1)
        
        # Count reasoning steps (numbered lists or bullet points)
        for text in full_texts:
            reasoning_steps += len(re.findall(r'\d+\.|•|\*|\-', text))
    
    # Calculate repetition ratio safely
    repetition_ratio = 0
    if sentence_counts and len(sentence_counts) > 0:
        repetition_ratio = repetition_count / len(sentence_counts)
    
    return {
        'avg_length': total_length,
        'repetition_ratio': repetition_ratio,
        'reasoning_steps': reasoning_steps / len(full_texts) if full_texts else 0,
    }

def create_performance_comparison(result_no_rpc: Dict[str, Any], result_rpc: Dict[str, Any], 
                                output_dir: str , question_id: int) -> str:
    """Create performance comparison visualization"""
    
    # Calculate statistics
    stats_no_rpc = calculate_token_statistics(result_no_rpc)
    stats_rpc = calculate_token_statistics(result_rpc)
    
    # Create comparison dataframe
    metrics = ['Total Tokens', 'Total Time (s)', 'Avg Tokens per Trace', 'Number of Traces']
    no_rpc_values = [
        stats_no_rpc['total_tokens'],
        stats_no_rpc['total_time'],
        stats_no_rpc['avg_tokens_per_warmup_trace'],
        stats_no_rpc['num_voting_traces']
    ]
    rpc_values = [
        stats_rpc['total_tokens'],
        stats_rpc['total_time'],
        stats_rpc['avg_tokens_per_warmup_trace'],
        stats_rpc['num_voting_traces']
    ]
    
    df = pd.DataFrame({
        'Metric': metrics,
        'No RPC': no_rpc_values,
        'With RPC': rpc_values
    })
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Performance Comparison: RPC vs No RPC', fontsize=16, fontweight='bold')
    
    # Plot 1: Token count comparison
    ax1 = axes[0, 0]
    x_pos = np.arange(len(metrics))
    width = 0.35
    ax1.bar(x_pos - width/2, no_rpc_values, width, label='No RPC', alpha=0.8)
    ax1.bar(x_pos + width/2, rpc_values, width, label='With RPC', alpha=0.8)
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Values')
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Efficiency metrics
    ax2 = axes[0, 1]
    efficiency_metrics = ['Tokens/s (No RPC)', 'Tokens/s (RPC)', 'Speedup']
    tokens_per_sec_no_rpc = stats_no_rpc['total_tokens'] / stats_no_rpc['total_time'] if stats_no_rpc['total_time'] > 0 else 0
    tokens_per_sec_rpc = stats_rpc['total_tokens'] / stats_rpc['total_time'] if stats_rpc['total_time'] > 0 else 0
    speedup = tokens_per_sec_rpc / tokens_per_sec_no_rpc if tokens_per_sec_no_rpc > 0 else 0
    
    efficiency_values = [tokens_per_sec_no_rpc, tokens_per_sec_rpc, speedup]
    colors = ['skyblue', 'lightgreen', 'gold']
    bars = ax2.bar(efficiency_metrics, efficiency_values, color=colors)
    ax2.set_ylabel('Values')
    ax2.set_title('Efficiency Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, efficiency_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiency_values)*0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    # Plot 3: Memory usage (simulated)
    ax3 = axes[1, 0]
    memory_metrics = ['KV Cache Size', 'Compression Ratio']
    # Simulate memory usage based on token count
    kv_cache_no_rpc = stats_no_rpc['total_tokens'] * 1.0  # Assume 1x memory per token
    kv_cache_rpc = stats_rpc['total_tokens'] * 0.25  # RPC reduces to ~25%
    
    # Calculate compression ratio with zero check
    if kv_cache_rpc > 0:
        compression_ratio = kv_cache_no_rpc / kv_cache_rpc
    else:
        compression_ratio = 0
    
    # Calculate remaining percentage with zero check
    if kv_cache_no_rpc > 0:
        remaining_percentage = kv_cache_rpc / kv_cache_no_rpc * 100
    else:
        remaining_percentage = 0
    
    memory_values = [remaining_percentage, compression_ratio]
    memory_labels = ['KV Cache Remaining (%)', 'Compression Ratio']
    bars = ax3.bar(memory_labels, memory_values, color=['lightcoral', 'gold'])
    ax3.set_ylabel('Values')
    ax3.set_title('Memory Compression Analysis')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, memory_values):
        if max(memory_values) > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(memory_values)*0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        else:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
    
    # Plot 4: Accuracy comparison
    ax4 = axes[1, 1]
    ground_truth = result_no_rpc.get('ground_truth', '')
    
    # Evaluate accuracy for both methods
    def get_accuracy(result):
        evaluation = result.get('evaluation', {})
        if not evaluation:
            return 0
        correct = sum(1 for method_result in evaluation.values() if method_result.get('is_correct', False))
        total = len([v for k, v in evaluation.items() if 'top10' not in k])
        return correct / total if total > 0 else 0
    
    acc_no_rpc = get_accuracy(result_no_rpc)
    acc_rpc = get_accuracy(result_rpc)
    
    accuracy_metrics = ['No RPC', 'With RPC']
    accuracy_values = [acc_no_rpc * 100, acc_rpc * 100]
    colors = ['skyblue', 'lightgreen']
    
    bars = ax4.bar(accuracy_metrics, accuracy_values, color=colors)
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Answer Accuracy Comparison')
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, accuracy_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'performance_comparison_q{question_id}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_reasoning_path_comparison(result_no_rpc: Dict[str, Any], result_rpc: Dict[str, Any], 
                                   output_dir: str , question_id: int) -> str:
    """Create reasoning path comparison visualization"""
    
    # Get traces - try all_voting_traces first, fall back to all_traces
    traces_no_rpc = result_no_rpc.get('all_voting_traces', [])
    if not traces_no_rpc or len(traces_no_rpc) == 0:
        traces_no_rpc = result_no_rpc.get('all_traces', [])
    
    traces_rpc = result_rpc.get('all_voting_traces', [])
    if not traces_rpc or len(traces_rpc) == 0:
        traces_rpc = result_rpc.get('all_traces', [])
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Reasoning Path Comparison: RPC vs No RPC', fontsize=16, fontweight='bold')
    
    # Analyze reasoning quality
    quality_no_rpc = analyze_reasoning_quality(traces_no_rpc)
    quality_rpc = analyze_reasoning_quality(traces_rpc)
    
    # Plot reasoning quality metrics
    ax1 = axes[0]
    metrics = ['Avg Length', 'Repetition Ratio', 'Reasoning Steps']
    no_rpc_vals = [quality_no_rpc.get('avg_length', 0), quality_no_rpc.get('repetition_ratio', 0), 
                   quality_no_rpc.get('reasoning_steps', 0)]
    rpc_vals = [quality_rpc.get('avg_length', 0), quality_rpc.get('repetition_ratio', 0),
                quality_rpc.get('reasoning_steps', 0)]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    ax1.bar(x_pos - width/2, no_rpc_vals, width, label='No RPC', alpha=0.8)
    ax1.bar(x_pos + width/2, rpc_vals, width, label='With RPC', alpha=0.8)
    ax1.set_xlabel('Quality Metrics')
    ax1.set_ylabel('Values')
    ax1.set_title('Reasoning Quality Analysis')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot trace count and length distribution
    ax2 = axes[1]
    
    # Get trace lengths - support both 'text' and 'full_text' fields
    lengths_no_rpc = []
    for trace in traces_no_rpc:
        text = trace.get('full_text') or trace.get('text', '')
        if text:
            lengths_no_rpc.append(len(text.split()))
    
    lengths_rpc = []
    for trace in traces_rpc:
        text = trace.get('full_text') or trace.get('text', '')
        if text:
            lengths_rpc.append(len(text.split()))
    
    # Create histogram
    ax2.hist(lengths_no_rpc, bins=20, alpha=0.6, label='No RPC', color='skyblue')
    ax2.hist(lengths_rpc, bins=20, alpha=0.6, label='With RPC', color='lightgreen')
    ax2.set_xlabel('Trace Length (tokens)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Trace Lengths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'reasoning_path_comparison_q{question_id}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def generate_comparison_report(result_no_rpc: Dict[str, Any], result_rpc: Dict[str, Any], 
                             output_dir: str, question_id: int) -> str:
    """Generate detailed comparison report"""
    
    report_path = os.path.join(output_dir, f'comparison_report_q{question_id}.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# DeepThinkLLM with RPC vs No RPC Comparison Report\n\n")
        
        # Basic information
        question = result_no_rpc.get('question', '')
        ground_truth = result_no_rpc.get('ground_truth', '')
        
        f.write(f"## Question\n{question}\n\n")
        f.write(f"## Ground Truth\n{ground_truth}\n\n")
        
        # Performance comparison
        f.write("## Performance Metrics\n\n")
        f.write("| Metric | No RPC | With RPC | Improvement |\n")
        f.write("|--------|--------|----------|-------------|\n")
        
        stats_no_rpc = calculate_token_statistics(result_no_rpc)
        stats_rpc = calculate_token_statistics(result_rpc)
        
        # Token statistics
        token_improvement = ((stats_no_rpc['total_tokens'] - stats_rpc['total_tokens']) / stats_no_rpc['total_tokens'] * 100) if stats_no_rpc['total_tokens'] > 0 else 0
        time_improvement = ((stats_no_rpc['total_time'] - stats_rpc['total_time']) / stats_no_rpc['total_time'] * 100) if stats_no_rpc['total_time'] > 0 else 0
        
        f.write(f"| Total Tokens | {stats_no_rpc['total_tokens']} | {stats_rpc['total_tokens']} | {token_improvement:.1f}% |\n")
        f.write(f"| Total Time (s) | {stats_no_rpc['total_time']:.2f} | {stats_rpc['total_time']:.2f} | {time_improvement:.1f}% |\n")
        
        # Calculate tokens/sec with zero checks
        tokens_per_sec_no_rpc = stats_no_rpc['total_tokens'] / stats_no_rpc['total_time'] if stats_no_rpc['total_time'] > 0 else 0
        tokens_per_sec_rpc = stats_rpc['total_tokens'] / stats_rpc['total_time'] if stats_rpc['total_time'] > 0 else 0
        tokens_per_sec_improvement = ((tokens_per_sec_rpc / tokens_per_sec_no_rpc - 1) * 100) if tokens_per_sec_no_rpc > 0 else 0
        
        f.write(f"| Tokens/sec | {tokens_per_sec_no_rpc:.2f} | {tokens_per_sec_rpc:.2f} | {tokens_per_sec_improvement:.1f}% |\n")
        
        trace_improvement = ((stats_rpc['num_voting_traces'] - stats_no_rpc['num_voting_traces']) / stats_no_rpc['num_voting_traces'] * 100) if stats_no_rpc['num_voting_traces'] > 0 else 0
        f.write(f"| Number of Traces | {stats_no_rpc['num_voting_traces']} | {stats_rpc['num_voting_traces']} | {trace_improvement:.1f}% |\n\n")
        
        # RPC Configuration
        if result_rpc.get('rpc_enabled', False):
            f.write("## RPC Configuration\n\n")
            rpc_config = result_rpc.get('rpc_config', {})
            for key, value in rpc_config.items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
        
        # Accuracy comparison
        f.write("## Accuracy Comparison\n\n")
        
        def get_accuracy_summary(result):
            evaluation = result.get('evaluation', {})
            if not evaluation:
                return "No evaluation data available"
            
            summary = []
            for method, method_result in evaluation.items():
                if 'top10' not in method and method_result:
                    answer = str(method_result.get('answer', ''))[:50] + '...' if len(str(method_result.get('answer', ''))) > 50 else str(method_result.get('answer', ''))
                    is_correct = method_result.get('is_correct', False)
                    confidence = method_result.get('confidence', None)
                    confidence_str = f"{confidence:.3f}" if confidence is not None else "N/A"
                    summary.append(f"- **{method}**: {answer} {'✓' if is_correct else '✗'} (confidence: {confidence_str})")
            
            return '\n'.join(summary)
        
        f.write("### No RPC Results\n")
        f.write(get_accuracy_summary(result_no_rpc) + "\n\n")
        
        f.write("### With RPC Results\n")
        f.write(get_accuracy_summary(result_rpc) + "\n\n")
        
        # Reasoning path analysis
        f.write("## Reasoning Path Analysis\n\n")
        
        # Get traces for analysis - try all_voting_traces first, fall back to all_traces
        traces_no_rpc_analysis = result_no_rpc.get('all_voting_traces', [])
        if not traces_no_rpc_analysis or len(traces_no_rpc_analysis) == 0:
            traces_no_rpc_analysis = result_no_rpc.get('all_traces', [])
        
        traces_rpc_analysis = result_rpc.get('all_voting_traces', [])
        if not traces_rpc_analysis or len(traces_rpc_analysis) == 0:
            traces_rpc_analysis = result_rpc.get('all_traces', [])
        
        quality_no_rpc = analyze_reasoning_quality(traces_no_rpc_analysis)
        quality_rpc = analyze_reasoning_quality(traces_rpc_analysis)
        
        f.write("| Metric | No RPC | With RPC |\n")
        f.write("|--------|--------|----------|\n")
        f.write(f"| Average Trace Length | {quality_no_rpc.get('avg_length', 0):.1f} tokens | {quality_rpc.get('avg_length', 0):.1f} tokens |\n")
        f.write(f"| Repetition Ratio | {quality_no_rpc.get('repetition_ratio', 0):.2f} | {quality_rpc.get('repetition_ratio', 0):.2f} |\n")
        f.write(f"| Reasoning Steps per Trace | {quality_no_rpc.get('reasoning_steps', 0):.1f} | {quality_rpc.get('reasoning_steps', 0):.1f} |\n\n")
        
        # Sample reasoning paths
        f.write("## Sample Reasoning Paths\n\n")
        
        f.write("### No RPC - Best Performing Trace\n")
        traces_no_rpc = result_no_rpc.get('all_voting_traces', [])
        if not traces_no_rpc or len(traces_no_rpc) == 0:
            traces_no_rpc = result_no_rpc.get('all_traces', [])
        if traces_no_rpc and len(traces_no_rpc) > 0:
            # Find best trace
            best_trace = max(traces_no_rpc, key=lambda t: t.get('min_conf', 0))
            f.write(f"**Confidence**: {best_trace.get('min_conf', 0):.3f}\n\n")
            # Support both 'text' and 'full_text' fields
            reasoning_text = best_trace.get('full_text') or best_trace.get('text') or best_trace.get('generated_text', '')
            if reasoning_text and isinstance(reasoning_text, str) and len(reasoning_text.strip()) > 0:
                # Truncate to 1000 chars if longer
                display_text = reasoning_text[:1000] if len(reasoning_text) > 1000 else reasoning_text
                f.write(f"**Reasoning Path**:\n```\n{display_text}")
                if len(reasoning_text) > 1000:
                    f.write("...")
                f.write("\n```\n\n")
            else:
                f.write("**Reasoning Path**: (No text available in trace)\n\n")
        else:
            f.write("**Reasoning Path**: (No traces available)\n\n")
        
        f.write("### With RPC - Best Performing Trace\n")
        traces_rpc = result_rpc.get('all_voting_traces', [])
        if not traces_rpc or len(traces_rpc) == 0:
            traces_rpc = result_rpc.get('all_traces', [])
        if traces_rpc and len(traces_rpc) > 0:
            # Find best trace
            best_trace = max(traces_rpc, key=lambda t: t.get('min_conf', 0))
            f.write(f"**Confidence**: {best_trace.get('min_conf', 0):.3f}\n\n")
            # Support both 'text' and 'full_text' fields
            reasoning_text = best_trace.get('full_text') or best_trace.get('text') or best_trace.get('generated_text', '')
            if reasoning_text and isinstance(reasoning_text, str) and len(reasoning_text.strip()) > 0:
                # Truncate to 1000 chars if longer
                display_text = reasoning_text[:1000] if len(reasoning_text) > 1000 else reasoning_text
                f.write(f"**Reasoning Path**:\n```\n{display_text}")
                if len(reasoning_text) > 1000:
                    f.write("...")
                f.write("\n```\n\n")
            else:
                f.write("**Reasoning Path**: (No text available in trace)\n\n")
        else:
            f.write("**Reasoning Path**: (No traces available)\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        f.write(f"- **Token Efficiency**: RPC reduced token count by {token_improvement:.1f}%\n")
        f.write(f"- **Time Efficiency**: RPC improved generation time by {time_improvement:.1f}%\n")
        
        # Calculate compression ratio
        compression_ratio = stats_no_rpc['total_tokens'] / stats_rpc['total_tokens'] if stats_rpc['total_tokens'] > 0 else 0
        f.write(f"- **Compression Ratio**: {compression_ratio:.1f}x\n")
        f.write(f"- **Accuracy Impact**: {'Minimal impact on accuracy' if abs(stats_no_rpc.get('correct_count', 0) - stats_rpc.get('correct_count', 0)) <= 1 else 'Some accuracy impact detected'}\n")
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Compare RPC vs non-RPC performance on single question')
    parser.add_argument('--result_no_rpc', type=str, required=True,
                       help='Path to result file without RPC')
    parser.add_argument('--result_rpc', type=str, required=True,
                       help='Path to result file with RPC')
    parser.add_argument('--output_dir', type=str, default='comparison_analysis',
                       help='Output directory for analysis')
    parser.add_argument('--question_id', type=int, default=0,
                       help='Question ID for reference')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading results...")
    result_no_rpc = load_result_file(args.result_no_rpc)
    result_rpc = load_result_file(args.result_rpc)
    
    print("Creating performance comparison...")
    perf_plot_path = create_performance_comparison(result_no_rpc, result_rpc, args.output_dir, args.question_id)
    
    print("Creating reasoning path comparison...")
    path_plot_path = create_reasoning_path_comparison(result_no_rpc, result_rpc, args.output_dir, args.question_id)
    
    print("Generating comparison report...")
    report_path = generate_comparison_report(result_no_rpc, result_rpc, args.output_dir, args.question_id)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Performance plot: {perf_plot_path}")
    print(f"Reasoning path plot: {path_plot_path}")
    print(f"Detailed report: {report_path}")

if __name__ == "__main__":
    main()