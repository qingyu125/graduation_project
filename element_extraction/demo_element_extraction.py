#!/usr/bin/env python3
"""
要素抽取演示脚本 - 面向高效推理的要素抽取与应用算法设计与实现

本脚本演示如何使用要素抽取模块从DeepConf和RPC推理结果中抽取要素。

使用示例:
    python demo_element_extraction.py --result_file ../online_outputs/result.pkl
    python demo_element_extraction.py --result_no_rpc no_rpc.pkl --result_rpc rpc.pkl
    python demo_element_extraction.py --batch --result_dir ../online_outputs/

Copyright (c) MiniMax Agent
"""

import argparse
import os
import sys
import json
import pickle
from datetime import datetime
from typing import Dict, Any, List, Optional
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from element_extractor import (
    ElementExtractionPipeline,
    LocalConfidenceExtractor,
    CompressionPathExtractor,
    load_and_extract
)
from element_analyzer import (
    ElementAnalyzer,
    EfficiencyQualityOptimizer,
    visualize_comparison
)


def setup_matplotlib():
    """配置matplotlib以支持中文显示"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False


def demo_single_result(result_path: str,
                       question_text: str = "",
                       ground_truth: Optional[str] = None,
                       output_dir: str = "element_extraction_results"):
    """
    演示：从单个结果文件中抽取要素
    
    Args:
        result_path: 结果文件路径
        question_text: 问题文本（可选）
        ground_truth: 正确答案（可选）
        output_dir: 输出目录
    """
    print(f"\n{'='*60}")
    print(f"要素抽取演示 - 单个结果")
    print(f"{'='*60}")
    
    # 检查文件是否存在
    if not os.path.exists(result_path):
        print(f"错误: 文件不存在 - {result_path}")
        return
    
    print(f"加载结果文件: {result_path}")
    
    # 加载结果
    if result_path.endswith('.pkl'):
        with open(result_path, 'rb') as f:
            result = pickle.load(f)
    elif result_path.endswith('.json'):
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
    else:
        print(f"错误: 不支持的文件格式 - {result_path}")
        return
    
    # 显示基本信息
    print(f"\n结果基本信息:")
    print(f"  问题ID: {result.get('question_id', 'N/A')}")
    print(f"  RPC启用: {result.get('rpc_enabled', False)}")
    print(f"  Trace数量: {len(result.get('all_traces', []))}")
    print(f"  总Token数: {result.get('total_tokens', 0)}")
    
    # 创建抽取管道
    pipeline = ElementExtractionPipeline(
        confidence_window_size=50,
        critical_percentile=0.1,
        rpc_P=1024,
        rpc_R=32,
        rpc_c=4
    )
    
    # 执行要素抽取
    print(f"\n执行要素抽取...")
    extraction_result = pipeline.extract_from_result(result, question_text, ground_truth)
    
    # 显示结果摘要
    print(f"\n要素抽取结果摘要:")
    print(f"  局部置信度要素数量: {len(extraction_result.local_confidence_elements)}")
    print(f"  路径压缩要素数量: {len(extraction_result.compression_path_elements)}")
    print(f"  效率评分: {extraction_result.overall_efficiency_score}")
    print(f"  质量评分: {extraction_result.reasoning_quality_score}")
    
    # 显示置信度统计
    conf_stats = extraction_result.confidence_statistics
    if conf_stats:
        print(f"\n置信度统计:")
        print(f"  平均置信度: {conf_stats.get('global_mean', 0):.4f}")
        print(f"  置信度标准差: {conf_stats.get('global_std', 0):.4f}")
        print(f"  置信度范围: [{conf_stats.get('global_min', 0):.4f}, {conf_stats.get('global_max', 0):.4f}]")
        print(f"  关键点数量: {conf_stats.get('critical_points_count', 0)}")
    
    # 显示压缩统计
    comp_stats = extraction_result.compression_statistics
    if comp_stats:
        print(f"\n压缩统计:")
        print(f"  压缩次数: {comp_stats.get('total_compressions', 0)}")
        print(f"  平均压缩比: {comp_stats.get('avg_compression_ratio', 0):.2f}x")
        print(f"  压缩Token数: {comp_stats.get('total_tokens_compressed', 0)}")
    
    # 保存结果
    print(f"\n保存结果...")
    json_path, pkl_path = pipeline.save_results(output_dir)
    
    # 生成可视化
    print(f"\n生成可视化...")
    pipeline.generate_visualization(output_dir)
    
    return extraction_result


def demo_compare_results(result_no_rpc: str,
                         result_rpc: str,
                         question_text: str = "",
                         ground_truth: Optional[str] = None,
                         output_dir: str = "element_comparison_results"):
    """
    演示：对比RPC和非RPC结果的要素抽取
    
    Args:
        result_no_rpc: 无RPC结果文件路径
        result_rpc: 有RPC结果文件路径
        question_text: 问题文本（可选）
        ground_truth: 正确答案（可选）
        output_dir: 输出目录
    """
    print(f"\n{'='*60}")
    print(f"要素抽取演示 - RPC vs 非RPC对比")
    print(f"{'='*60}")
    
    # 加载两个结果
    print(f"加载无RPC结果: {result_no_rpc}")
    with open(result_no_rpc, 'rb') as f:
        result_no_rpc_data = pickle.load(f)
    
    print(f"加载RPC结果: {result_rpc}")
    with open(result_rpc, 'rb') as f:
        result_rpc_data = pickle.load(f)
    
    # 创建抽取管道
    pipeline = ElementExtractionPipeline(
        confidence_window_size=50,
        critical_percentile=0.1,
        rpc_P=1024,
        rpc_R=32,
        rpc_c=4
    )
    
    # 抽取两组结果的要素
    print(f"\n抽取无RPC结果的要素...")
    extraction_no_rpc = pipeline.extract_from_result(result_no_rpc_data, question_text, ground_truth)
    
    print(f"抽取RPC结果的要素...")
    extraction_rpc = pipeline.extract_from_result(result_rpc_data, question_text, ground_truth)
    
    # 对比分析
    print(f"\n{'='*60}")
    print(f"对比分析结果")
    print(f"{'='*60}")
    
    print(f"\n1. 效率评分对比:")
    print(f"   无RPC: {extraction_no_rpc.overall_efficiency_score:.4f}")
    print(f"   有RPC: {extraction_rpc.overall_efficiency_score:.4f}")
    
    if extraction_rpc.compression_statistics.get('avg_compression_ratio', 0) > 1:
        improvement = (extraction_rpc.overall_efficiency_score - extraction_no_rpc.overall_efficiency_score) / max(extraction_no_rpc.overall_efficiency_score, 0.001) * 100
        print(f"   效率提升: {improvement:.1f}%")
    
    print(f"\n2. 质量评分对比:")
    print(f"   无RPC: {extraction_no_rpc.reasoning_quality_score:.4f}")
    print(f"   有RPC: {extraction_rpc.reasoning_quality_score:.4f}")
    
    quality_diff = extraction_rpc.reasoning_quality_score - extraction_no_rpc.reasoning_quality_score
    print(f"   质量变化: {'+' if quality_diff > 0 else ''}{quality_diff:.4f}")
    
    print(f"\n3. 置信度对比:")
    conf_no_rpc = extraction_no_rpc.confidence_statistics
    conf_rpc = extraction_rpc.confidence_statistics
    
    if conf_no_rpc and conf_rpc:
        print(f"   无RPC平均置信度: {conf_no_rpc.get('global_mean', 0):.4f}")
        print(f"   有RPC平均置信度: {conf_rpc.get('global_mean', 0):.4f}")
    
    print(f"\n4. 压缩效果 (仅RPC):")
    comp_stats = extraction_rpc.compression_statistics
    if comp_stats and comp_stats.get('total_compressions', 0) > 0:
        print(f"   总压缩次数: {comp_stats.get('total_compressions', 0)}")
        print(f"   平均压缩比: {comp_stats.get('avg_compression_ratio', 0):.2f}x")
        print(f"   节省KV缓存: {comp_stats.get('total_kv_saved_mb', 0):.2f} MB")
    
    # 创建分析器并生成报告
    print(f"\n生成分析报告...")
    analyzer = ElementAnalyzer()
    
    results = [extraction_no_rpc, extraction_rpc]
    reports = analyzer.generate_comprehensive_report(results)
    
    # 保存报告
    os.makedirs(output_dir, exist_ok=True)
    analyzer.save_reports(reports, output_dir)
    
    # 生成对比可视化
    print(f"\n生成对比可视化...")
    visualize_comparison([extraction_no_rpc], [extraction_rpc], output_dir)
    
    # 保存对比结果
    comparison_result = {
        'timestamp': datetime.now().isoformat(),
        'no_rpc': {
            'efficiency_score': extraction_no_rpc.overall_efficiency_score,
            'quality_score': extraction_no_rpc.reasoning_quality_score,
            'confidence_mean': extraction_no_rpc.confidence_statistics.get('global_mean', 0),
            'token_count': len(extraction_no_rpc.local_confidence_elements)
        },
        'rpc': {
            'efficiency_score': extraction_rpc.overall_efficiency_score,
            'quality_score': extraction_rpc.reasoning_quality_score,
            'confidence_mean': extraction_rpc.confidence_statistics.get('global_mean', 0),
            'token_count': len(extraction_rpc.local_confidence_elements),
            'compression_ratio': extraction_rpc.compression_statistics.get('avg_compression_ratio', 0),
            'total_compressions': extraction_rpc.compression_statistics.get('total_compressions', 0)
        },
        'analysis_reports': {name: report.to_dict() for name, report in reports.items()}
    }
    
    comparison_path = os.path.join(output_dir, f"comparison_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n对比结果已保存到: {comparison_path}")
    
    return comparison_result


def demo_batch_results(result_dir: str,
                       output_dir: str = "element_batch_results"):
    """
    演示：批量处理结果目录中的所有结果文件
    
    Args:
        result_dir: 结果目录路径
        output_dir: 输出目录
    """
    print(f"\n{'='*60}")
    print(f"要素抽取演示 - 批量处理")
    print(f"{'='*60}")
    
    # 检查目录
    if not os.path.isdir(result_dir):
        print(f"错误: 目录不存在 - {result_dir}")
        return
    
    # 查找所有pkl文件
    pkl_files = [f for f in os.listdir(result_dir) if f.endswith('.pkl')]
    
    if not pkl_files:
        print(f"错误: 目录中没有.pkl文件 - {result_dir}")
        return
    
    print(f"找到 {len(pkl_files)} 个结果文件")
    
    # 创建抽取管道
    pipeline = ElementExtractionPipeline(
        confidence_window_size=50,
        critical_percentile=0.1,
        rpc_P=1024,
        rpc_R=32,
        rpc_c=4
    )
    
    all_results = []
    
    # 处理每个文件
    for i, pkl_file in enumerate(pkl_files):
        file_path = os.path.join(result_dir, pkl_file)
        
        print(f"\n处理文件 {i+1}/{len(pkl_files)}: {pkl_file}")
        
        try:
            with open(file_path, 'rb') as f:
                result = pickle.load(f)
            
            extraction = pipeline.extract_from_result(result)
            all_results.append(extraction)
            
            print(f"  完成: 效率={extraction.overall_efficiency_score:.4f}, 质量={extraction.reasoning_quality_score:.4f}")
            
        except Exception as e:
            print(f"  错误: {str(e)}")
    
    # 生成综合报告
    print(f"\n{'='*60}")
    print(f"批量处理完成 - 共 {len(all_results)} 个结果")
    print(f"{'='*60}")
    
    if all_results:
        # 分析所有结果
        analyzer = ElementAnalyzer()
        reports = analyzer.generate_comprehensive_report(all_results)
        
        # 保存报告
        os.makedirs(output_dir, exist_ok=True)
        analyzer.save_reports(reports, output_dir)
        
        # 打印摘要
        print(f"\n综合统计:")
        print(f"  平均效率评分: {np.mean([r.overall_efficiency_score for r in all_results]):.4f}")
        print(f"  平均质量评分: {np.mean([r.reasoning_quality_score for r in all_results]):.4f}")
        print(f"  RPC启用比例: {sum(1 for r in all_results if r.rpc_enabled) / len(all_results) * 100:.1f}%")
    
    return all_results


def demo_analyze_extraction(extraction_result):
    """
    演示：深度分析抽取结果
    
    Args:
        extraction_result: 要素抽取结果
    """
    print(f"\n{'='*60}")
    print(f"要素深度分析")
    print(f"{'='*60}")
    
    # 创建分析器
    analyzer = ElementAnalyzer()
    
    # 生成报告
    reports = analyzer.generate_comprehensive_report([extraction_result])
    
    # 打印报告
    for name, report in reports.items():
        print(f"\n{name.upper()} 分析报告:")
        print(f"  报告ID: {report.report_id}")
        
        print(f"\n  摘要:")
        for key, value in report.summary.items():
            print(f"    {key}: {value}")
        
        if report.detailed_findings:
            print(f"\n  详细发现:")
            for finding in report.detailed_findings:
                print(f"    - [{finding['severity']}] {finding['title']}")
                print(f"      {finding['description']}")
        
        if report.recommendations:
            print(f"\n  推荐:")
            for rec in report.recommendations:
                print(f"    - {rec}")
    
    return reports


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='要素抽取演示脚本 - 面向高效推理的要素抽取与应用算法设计与实现',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 单个结果分析
    python demo_element_extraction.py --result_file result.pkl
    
    # RPC vs 非RPC对比
    python demo_element_extraction.py --compare --result_no_rpc no_rpc.pkl --result_rpc rpc.pkl
    
    # 批量处理
    python demo_element_extraction.py --batch --result_dir ./results/
        """
    )
    
    # 输入参数
    parser.add_argument('--result_file', type=str,
                       help='单个结果文件路径 (.pkl 或 .json)')
    parser.add_argument('--result_no_rpc', type=str,
                       help='无RPC结果文件路径')
    parser.add_argument('--result_rpc', type=str,
                       help='有RPC结果文件路径')
    parser.add_argument('--result_dir', type=str,
                       help='结果目录路径（批量处理）')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='element_extraction_results',
                       help='输出目录')
    parser.add_argument('--question_text', type=str, default='',
                       help='问题文本')
    parser.add_argument('--ground_truth', type=str, default=None,
                       help='正确答案')
    
    # 分析选项
    parser.add_argument('--analyze', action='store_true',
                       help='对抽取结果进行深度分析')
    parser.add_argument('--no_visualize', action='store_true',
                       help='不生成可视化')
    
    args = parser.parse_args()
    
    # 配置matplotlib
    setup_matplotlib()
    
    import numpy as np
    
    # 根据参数执行不同的演示
    # 使用getattr安全获取参数
    result_file = getattr(args, 'result_file', None)
    result_no_rpc = getattr(args, 'result_no_rpc', None)
    result_rpc = getattr(args, 'result_rpc', None)
    result_dir = getattr(args, 'result_dir', None)
    do_analyze = getattr(args, 'analyze', False)
    
    # 如果同时提供了result_no_rpc和result_rpc，则进行对比分析
    if result_no_rpc and result_rpc:
        # 对比分析
        demo_compare_results(
            args.result_no_rpc,
            args.result_rpc,
            args.question_text,
            args.ground_truth,
            args.output_dir
        )
        
    elif result_dir:
        # 批量处理
        demo_batch_results(args.result_dir, args.output_dir)
        
    elif result_file:
        # 单个结果分析
        result = demo_single_result(
            args.result_file,
            args.question_text,
            args.ground_truth,
            args.output_dir
        )
        
        # 深度分析
        if do_analyze and result:
            demo_analyze_extraction(result)
        
    else:
        parser.print_help()
        print(f"\n错误: 请指定 --result_file, --compare 或 --result_dir 参数")


if __name__ == "__main__":
    main()
