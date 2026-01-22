#!/usr/bin/env python3
"""
Complete RPC Integration and Comparison Pipeline

This script runs the complete pipeline for integrating RPC into deepconf and comparing
performance with and without RPC on a single question.

Usage:
    python run_rpc_comparison.py --model MODEL_PATH --dataset DATASET --qid QUESTION_ID

Example:
    python run_rpc_comparison.py \
        --model ./models/DeepSeek-R1-0528-Qwen3-8B \
        --dataset data/aime_2025.jsonl \
        --qid 0
"""
import argparse
import subprocess
import os
import sys
import json
from datetime import datetime
import pickle

def run_command(cmd, description):
    """Run a command and return the result"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return None
    
    print(f"SUCCESS: {description} completed")
    if result.stdout:
        print(f"Output: {result.stdout}")
    
    return result

def find_latest_result(directory, prefix, qid, rid):
    """Find the most recent result file matching the criteria"""
    import glob
    pattern = f"{directory}/{prefix}*qid{qid}*rid{rid}*.pkl"
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Return the most recent file
    return max(files, key=os.path.getmtime)

def main():
    parser = argparse.ArgumentParser(description='Run complete RPC vs non-RPC comparison')
    
    # Model and data arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Model path or name')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset file path')
    parser.add_argument('--qid', type=int, required=True,
                       help='Question ID to process (0-based index)')
    
    # Output directories
    parser.add_argument('--no_rpc_dir', type=str, default='no_rpc_outputs',
                       help='Output directory for no-RPC results')
    parser.add_argument('--rpc_dir', type=str, default='rpc_outputs',
                       help='Output directory for RPC results')
    parser.add_argument('--analysis_dir', type=str, default='comparison_analysis',
                       help='Output directory for analysis results')
    
    # Run parameters (same for both experiments)
    parser.add_argument('--rid_base', type=str, default='comparison_run',
                       help='Base run ID for identification')
    parser.add_argument('--warmup_traces', type=int, default=2,
                       help='Number of warmup traces')
    parser.add_argument('--total_budget', type=int, default=4,
                       help='Total trace budget')
    parser.add_argument('--confidence_percentile', type=int, default=90,
                       help='Confidence percentile for threshold')
    parser.add_argument('--window_size', type=int, default=2048,
                       help='Sliding window size for confidence computation')
    parser.add_argument('--max_tokens', type=int, default=64000,
                       help='Maximum tokens per generation')
    parser.add_argument('--model_type', type=str, default="deepseek", choices=["deepseek", "gpt"],
                       help='Model type for prompt formatting')
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top-p sampling parameter')
    parser.add_argument('--top_k', type=int, default=0,
                       help='Top-k sampling parameter')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Tensor parallel size for model')
    
    # RPC parameters
    parser.add_argument('--rpc_P', type=int, default=1024,
                       help='RPC compression interval')
    parser.add_argument('--rpc_R', type=int, default=32,
                       help='RPC selector window size')
    parser.add_argument('--rpc_c', type=int, default=4,
                       help='RPC compression ratio')
    parser.add_argument('--rpc_selectors', type=str, default='recent', choices=['recent', 'prompt', 'new'],
                       help='RPC selector type')
    parser.add_argument('--rpc_aggregation', type=str, default='all', choices=['all', 'group', 'none'],
                       help='RPC aggregation method')
    parser.add_argument('--rpc_kernel_size', type=int, default=7,
                       help='RPC pooling kernel size')
    parser.add_argument('--rpc_pooling', type=str, default='avgpool', choices=['avgpool', 'maxpool'],
                       help='RPC pooling method')
    
    # Analysis options
    parser.add_argument('--no_analysis', action='store_true',
                       help='Skip analysis and comparison step')
    parser.add_argument('--verbose', action='store_true',
                       help='Show verbose output')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"DeepThinkLLM RPC Integration and Comparison Pipeline")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Question ID: {args.qid}")
    print(f"RPC Config: P={args.rpc_P}, R={args.rpc_R}, c={args.rpc_c}")
    print(f"{'='*80}")
    
    # Create output directories
    os.makedirs(args.no_rpc_dir, exist_ok=True)
    os.makedirs(args.rpc_dir, exist_ok=True)
    
    # Run experiment without RPC
    print(f"\n{'#'*60}")
    print(f"STEP 1: Running experiment WITHOUT RPC")
    print(f"{'#'*60}")
    
    no_rpc_cmd = [
        sys.executable, "example_online.py",
        "--model", args.model,
        "--dataset", args.dataset,
        "--qid", str(args.qid),
        "--rid", f"{args.rid_base}_no_rpc",
        "--warmup_traces", str(args.warmup_traces),
        "--total_budget", str(args.total_budget),
        "--confidence_percentile", str(args.confidence_percentile),
        "--window_size", str(args.window_size),
        "--max_tokens", str(args.max_tokens),
        "--model_type", args.model_type,
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--top_k", str(args.top_k),
        "--output_dir", args.no_rpc_dir,
        "--tensor_parallel_size", str(args.tensor_parallel_size),
    ]
    
    no_rpc_result = run_command(no_rpc_cmd, "Running experiment without RPC")
    if no_rpc_result is None:
        print("ERROR: Failed to run experiment without RPC")
        return 1
    
    # Find the result file
    no_rpc_result_file = find_latest_result(args.no_rpc_dir, "deepthink_online_qid", args.qid, f"{args.rid_base}_no_rpc")
    if not no_rpc_result_file:
        print("ERROR: Could not find no-RPC result file")
        return 1
    
    print(f"No-RPC result saved to: {no_rpc_result_file}")
    
    # Run experiment with RPC
    print(f"\n{'#'*60}")
    print(f"STEP 2: Running experiment WITH RPC")
    print(f"{'#'*60}")
    
    rpc_cmd = [
        sys.executable, "example_online_rpc.py",
        "--model", args.model,
        "--dataset", args.dataset,
        "--qid", str(args.qid),
        "--rid", f"{args.rid_base}_with_rpc",
        "--warmup_traces", str(args.warmup_traces),
        "--total_budget", str(args.total_budget),
        "--confidence_percentile", str(args.confidence_percentile),
        "--window_size", str(args.window_size),
        "--max_tokens", str(args.max_tokens),
        "--model_type", args.model_type,
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--top_k", str(args.top_k),
        "--output_dir", args.rpc_dir,
        "--tensor_parallel_size", str(args.tensor_parallel_size),
        "--enable_rpc",  # Flag parameter, no value needed
        "--rpc_P", str(args.rpc_P),
        "--rpc_R", str(args.rpc_R),
        "--rpc_c", str(args.rpc_c),
        "--rpc_selectors", args.rpc_selectors,
        "--rpc_aggregation", args.rpc_aggregation,
        "--rpc_kernel_size", str(args.rpc_kernel_size),
        "--rpc_pooling", args.rpc_pooling,
    ]
    
    rpc_result = run_command(rpc_cmd, "Running experiment with RPC")
    if rpc_result is None:
        print("ERROR: Failed to run experiment with RPC")
        return 1
    
    # Find the result file
    rpc_result_file = find_latest_result(args.rpc_dir, "deepthink_online_rpc_qid", args.qid, f"{args.rid_base}_with_rpc")
    if not rpc_result_file:
        print("ERROR: Could not find RPC result file")
        return 1
    
    print(f"RPC result saved to: {rpc_result_file}")
    
    # Run comparison analysis
    if not args.no_analysis:
        print(f"\n{'#'*60}")
        print(f"STEP 3: Running comparison analysis")
        print(f"{'#'*60}")
        
        analysis_cmd = [
            sys.executable, "example_compare_rpc.py",
            "--result_no_rpc", no_rpc_result_file,
            "--result_rpc", rpc_result_file,
            "--output_dir", args.analysis_dir,
            "--question_id", str(args.qid),
        ]
        
        analysis_result = run_command(analysis_cmd, "Running comparison analysis")
        if analysis_result is None:
            print("ERROR: Failed to run comparison analysis")
            return 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"COMPLETE PIPELINE FINISHED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"Results summary:")
    print(f"  No RPC results: {no_rpc_result_file}")
    print(f"  RPC results: {rpc_result_file}")
    if not args.no_analysis:
        print(f"  Analysis directory: {args.analysis_dir}")
        analysis_files = [
            os.path.join(args.analysis_dir, 'performance_comparison.png'),
            os.path.join(args.analysis_dir, 'reasoning_path_comparison.png'),
            os.path.join(args.analysis_dir, 'comparison_report.md')
        ]
        print(f"  Analysis outputs:")
        for file_path in analysis_files:
            if os.path.exists(file_path):
                print(f"    - {file_path}")
    
    # Load and display quick comparison
    try:
        with open(no_rpc_result_file, 'rb') as f:
            no_rpc_data = pickle.load(f)
        with open(rpc_result_file, 'rb') as f:
            rpc_data = pickle.load(f)
        
        print(f"\nQuick Performance Comparison:")
        print(f"  No RPC: {no_rpc_data.get('total_tokens', 0)} tokens, {no_rpc_data.get('total_time', 0):.2f}s")
        print(f"  With RPC: {rpc_data.get('total_tokens', 0)} tokens, {rpc_data.get('total_time', 0):.2f}s")
        
        if rpc_data.get('total_tokens', 0) > 0 and no_rpc_data.get('total_tokens', 0) > 0:
            token_reduction = (no_rpc_data['total_tokens'] - rpc_data['total_tokens']) / no_rpc_data['total_tokens'] * 100
            print(f"  Token reduction: {token_reduction:.1f}%")
        
        if rpc_data.get('total_time', 0) > 0 and no_rpc_data.get('total_time', 0) > 0:
            speedup = no_rpc_data['total_time'] / rpc_data['total_time']
            print(f"  Speedup: {speedup:.2f}x")
        
    except Exception as e:
        print(f"Warning: Could not load result files for quick comparison: {e}")
    
    print(f"\n{'='*80}")
    print(f"Pipeline completed! Check the output directories for detailed results.")
    print(f"{'='*80}")
    
    return 0

if __name__ == "__main__":
    exit(main())