#!/usr/bin/env python3
"""
验证处理后的数据是否正确
"""

import json
import sys
from pathlib import Path

def validate_training_data():
    """验证训练数据"""
    print("=== 验证训练数据 ===")
    
    # 检查训练数据文件
    data_files = {
        "完整训练数据": "data/processed/docred_training_data.json",
        "训练集": "data/processed/train.json", 
        "验证集": "data/processed/val.json",
        "测试集": "data/processed/test.json",
        "GUI数据": "data/processed/gui_data.json",
        "数据统计": "data/processed/dataset_stats.json"
    }
    
    validation_results = {}
    
    for name, path in data_files.items():
        file_path = Path(path)
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    size = len(data)
                elif isinstance(data, dict):
                    size = len(data)
                    if 'train_data' in data:
                        size = f"{len(data['train_data'])}/{len(data)}"
                else:
                    size = "未知"
                
                print(f"✅ {name}: {path} (大小: {size})")
                validation_results[name] = True
                
            except Exception as e:
                print(f"❌ {name}: {path} (错误: {e})")
                validation_results[name] = False
        else:
            print(f"❌ {name}: {path} (文件不存在)")
            validation_results[name] = False
    
    return validation_results

def check_training_data_content():
    """检查训练数据内容"""
    print("\n=== 检查训练数据内容 ===")
    
    try:
        with open("data/processed/docred_training_data.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"总文档数: {data.get('total_documents', 'N/A')}")
        print(f"训练样本数: {len(data.get('train_data', []))}")
        print(f"验证样本数: {len(data.get('val_data', []))}")
        print(f"测试样本数: {len(data.get('test_data', []))}")
        print(f"关系类型数: {len(data.get('relation_types', []))}")
        print(f"实体类型: {data.get('entity_types', [])}")
        
        # 检查数据格式
        if data.get('train_data'):
            sample = data['train_data'][0]
            required_fields = ['doc_id', 'title', 'original_text', 'entities', 'relations']
            for field in required_fields:
                if field in sample:
                    print(f"✅ 训练数据包含字段: {field}")
                else:
                    print(f"❌ 训练数据缺少字段: {field}")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查训练数据失败: {e}")
        return False

def check_gui_data():
    """检查GUI数据"""
    print("\n=== 检查GUI数据 ===")
    
    try:
        with open("data/processed/gui_data.json", 'r', encoding='utf-8') as f:
            gui_data = json.load(f)
        
        print(f"GUI文档数: {len(gui_data)}")
        
        if gui_data:
            sample = gui_data[0]
            print(f"示例文档: {sample.get('title', 'N/A')}")
            print(f"实体数: {len(sample.get('entities', []))}")
            print(f"关系数: {len(sample.get('relations', []))}")
            
            # 检查实体格式
            if sample.get('entities'):
                entity = sample['entities'][0]
                print(f"实体格式: {entity.get('name', 'N/A')} ({entity.get('type', 'N/A')})")
            
            # 检查关系格式
            if sample.get('relations'):
                relation = sample['relations'][0]
                print(f"关系格式: {relation.get('head_entity_name', 'N/A')} -> {relation.get('relation_name', 'N/A')} -> {relation.get('tail_entity_name', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查GUI数据失败: {e}")
        return False

def main():
    print("DocRED数据验证工具")
    print("=" * 50)
    
    # 验证文件存在性
    validation_results = validate_training_data()
    
    # 检查内容
    if validation_results.get("完整训练数据"):
        check_training_data_content()
    
    if validation_results.get("GUI数据"):
        check_gui_data()
    
    # 生成验证报告
    print("\n=== 验证总结 ===")
    total_checks = len(validation_results)
    passed_checks = sum(validation_results.values())
    
    if passed_checks == total_checks:
        print(f"✅ 所有检查通过 ({passed_checks}/{total_checks})")
        print("数据准备完成，可以开始训练！")
        return True
    else:
        print(f"❌ 部分检查失败 ({passed_checks}/{total_checks})")
        print("请检查数据文件并重新运行数据处理脚本")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)