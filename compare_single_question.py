#!/usr/bin/env python3
"""
Simplified RPC vs Non-RPC Single Question Comparison

This script provides a streamlined interface for comparing RPC vs non-RPC performance
on a single question with your specific parameters.
"""
import argparse
import os
import sys

def run_comparison():
    parser = argparse.ArgumentParser(description='Compare RPC vs non-RPC on single question')
    
    # Required parameters
    parser.add_argument('--model', type=str, required=True,
                       help='Model path (e.g., ./models/DeepSeek-R1-0528-Qwen3-8B)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset file (e.g., data/aime_2025.jsonl)')
    parser.add_argument('--qid', type=int, required=True,
                       help='Question ID to process (0-based index)')
    
    # Optional output directories
    parser.add_argument('--no_rpc_dir', type=str, default='no_rpc_results',
                       help='Output directory for no-RPC results')
    parser.add_argument('--rpc_dir', type=str, default='rpc_results',
                       help='Output directory for RPC results')
    parser.add_argument('--analysis_dir', type=str, default='comparison_results',
                       help='Output directory for comparison analysis')
    
    # RPC parameters
    parser.add_argument('--rpc_P', type=int, default=1024,
                       help='RPC compression interval (default: 1024)')
    parser.add_argument('--rpc_R', type=int, default=32,
                       help='RPC selector window size (default: 32)')
    parser.add_argument('--rpc_c', type=int, default=4,
                       help='RPC compression ratio (default: 4)')
    
    # Other parameters
    parser.add_argument('--warmup_traces', type=int, default=2,
                       help='Number of warmup traces (default: 2)')
    parser.add_argument('--total_budget', type=int, default=4,
                       help='Total trace budget (default: 4)')
    parser.add_argument('--confidence_percentile', type=int, default=90,
                       help='Confidence percentile (default: 90)')
    parser.add_argument('--max_tokens', type=int, default=64000,
                       help='Max tokens per generation (default: 64000)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"DeepThinkLLM RPC Integration - Single Question Comparison")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Question ID: {args.qid}")
    print(f"RPC Config: P={args.rpc_P}, R={args.rpc_R}, c={args.rpc_c}")
    print(f"{'='*80}\n")
    
    # Check if model and dataset exist
    if not os.path.exists(args.model):
        print(f"ERROR: Model path does not exist: {args.model}")
        return 1
    
    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset file does not exist: {args.dataset}")
        return 1
    
    # Create output directories
    os.makedirs(args.no_rpc_dir, exist_ok=True)
    os.makedirs(args.rpc_dir, exist_ok=True)
    os.makedirs(args.analysis_dir, exist_ok=True)
    
    # Construct and run the complete comparison pipeline
    cmd = [
        sys.executable, "run_rpc_comparison.py",
        "--model", args.model,
        "--dataset", args.dataset,
        "--qid", str(args.qid),
        "--no_rpc_dir", args.no_rpc_dir,
        "--rpc_dir", args.rpc_dir,
        "--analysis_dir", args.analysis_dir,
        "--rpc_P", str(args.rpc_P),
        "--rpc_R", str(args.rpc_R),
        "--rpc_c", str(args.rpc_c),
        "--warmup_traces", str(args.warmup_traces),
        "--total_budget", str(args.total_budget),
        "--confidence_percentile", str(args.confidence_percentile),
        "--max_tokens", str(args.max_tokens),
    ]
    
    print("Starting comparison pipeline...")
    print(f"Command: {' '.join(cmd)}\n")
    
    import subprocess
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n{'='*80}")
        print(f"SUCCESS: Comparison completed!")
        print(f"Results saved in:")
        print(f"  - No RPC: {args.no_rpc_dir}/")
        print(f"  - With RPC: {args.rpc_dir}/")
        print(f"  - Analysis: {args.analysis_dir}/")
        print(f"{'='*80}")
    else:
        print(f"\nERROR: Comparison failed with return code {result.returncode}")
    
    return result.returncode

if __name__ == "__main__":
    exit(run_comparison())