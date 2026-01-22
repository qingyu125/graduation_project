#!/usr/bin/env python3
"""
简单测试脚本：验证RPC功能是否正常工作

这个脚本将运行一个非常简单的测试，使用第一个问题来验证：
1. 非RPC模式是否正常工作
2. RPC模式是否可以启动
3. 两者之间的基本差异

使用方法：
python simple_test.py --model ./models/DeepSeek-R1-0528-Qwen3-8B

注意：您需要确保模型文件存在！
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="简单RPC功能测试")
    parser.add_argument("--model", required=True, help="模型文件路径")
    parser.add_argument("--dataset", default="data/aime_2025.jsonl", help="数据集文件路径")
    parser.add_argument("--qid", type=int, default=0, help="要测试的问题ID")
    parser.add_argument("--output_dir", default="test_outputs", help="输出目录")
    
    args = parser.parse_args()
    
    # 验证文件存在
    if not os.path.exists(args.model):
        print(f"❌ 错误：模型文件不存在: {args.model}")
        print("请确保模型文件路径正确")
        return 1
        
    if not os.path.exists(args.dataset):
        print(f"❌ 错误：数据集文件不存在: {args.dataset}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取问题
    try:
        with open(args.dataset, 'r') as f:
            lines = f.readlines()
        if args.qid >= len(lines):
            print(f"❌ 错误：qid {args.qid} 超出范围，数据集只有 {len(lines)} 个问题")
            return 1
            
        question_data = json.loads(lines[args.qid])
        question = question_data['question']
        answer = question_data['answer']
        
        print(f"📝 测试问题 {args.qid}:")
        print(f"问题: {question[:100]}{'...' if len(question) > 100 else ''}")
        print(f"答案: {answer}")
        print()
        
    except Exception as e:
        print(f"❌ 读取数据集失败: {e}")
        return 1
    
    # 构建运行命令
    base_args = [
        "python", "examples/example_online.py",
        "--model", args.model,
        "--dataset", args.dataset,
        "--qid", str(args.qid),
        "--rid", "test_baseline",
        "--warmup_traces", "0",
        "--total_budget", "1",
        "--confidence_percentile", "90",
        "--window_size", "2048",
        "--max_tokens", "64000",
        "--model_type", "deepseek",
        "--output_dir", args.output_dir
    ]
    
    # 运行非RPC版本
    print("🚀 运行非RPC版本...")
    print("命令:", " ".join(base_args))
    
    start_time = time.time()
    result = os.system(" ".join(base_args))
    baseline_time = time.time() - start_time
    
    if result == 0:
        print(f"✅ 非RPC版本运行成功，耗时: {baseline_time:.2f}秒")
    else:
        print(f"❌ 非RPC版本运行失败，返回码: {result}")
        return 1
    
    # 运行RPC版本
    rpc_args = [
        "python", "examples/example_online_rpc.py",
        "--model", args.model,
        "--dataset", args.dataset,
        "--qid", str(args.qid),
        "--rid", "test_rpc",
        "--warmup_traces", "0",
        "--total_budget", "1",
        "--confidence_percentile", "90",
        "--window_size", "2048",
        "--max_tokens", "64000",
        "--model_type", "deepseek",
        "--output_dir", args.output_dir,
        "--enable_rpc",
        "--rpc_P", "1024",
        "--rpc_R", "32",
        "--rpc_c", "4"
    ]
    
    print()
    print("🚀 运行RPC版本...")
    print("命令:", " ".join(rpc_args))
    
    start_time = time.time()
    result = os.system(" ".join(rpc_args))
    rpc_time = time.time() - start_time
    
    if result == 0:
        print(f"✅ RPC版本运行成功，耗时: {rpc_time:.2f}秒")
    else:
        print(f"❌ RPC版本运行失败，返回码: {result}")
        return 1
    
    # 显示结果对比
    print("\n" + "="*60)
    print("🎉 测试完成！结果对比:")
    print("="*60)
    print(f"非RPC版本耗时: {baseline_time:.2f}秒")
    print(f"RPC版本耗时:   {rpc_time:.2f}秒")
    print(f"时间差异:      {rpc_time - baseline_time:.2f}秒")
    
    if rpc_time < baseline_time:
        print("🚀 RPC版本更快!")
    elif rpc_time > baseline_time:
        print("📈 非RPC版本更快")
    else:
        print("⏱️  两个版本耗时相同")
    
    print(f"\n📁 输出文件保存在: {args.output_dir}")
    print("您可以使用以下命令查看详细对比:")
    print(f"  python compare_single_question.py --output_dir {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())