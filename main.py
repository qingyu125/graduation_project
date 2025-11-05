import argparse
import yaml
from utils.data_processor import DataProcessor
# from train.lora_trainer import LoRATrainer
# from evaluate.evaluator import Evaluator
# from models.base.code_llama_wrapper import CodeLlamaWrapper

def main(args):
    # 加载配置
    data_config = yaml.safe_load(open("./config/data_config.yaml", encoding="utf-8"))
    model_config = yaml.safe_load(open("./config/model_config.yaml", encoding="utf-8"))
    
    if args.step == "preprocess":
        # 数据预处理：生成伪样本+划分数据集
        processor = DataProcessor()
        processor.generate_general_pseudo_samples()
        train, val, test = processor.process_and_split()
        print(f"数据预处理完成：训练集{len(train)}条，验证集{len(val)}条，测试集{len(test)}条")
    
    elif args.step == "train":
        # 模型训练
        trainer = LoRATrainer()
        trainer.train(
            train_data_path=data_config["paths"]["processed_train"],
            val_data_path=data_config["paths"]["processed_val"],
            output_dir=data_config["paths"]["output_dir"]
        )
        print("模型训练完成，权重已保存至", data_config["paths"]["output_dir"])
    
    elif args.step == "evaluate":
        # 模型评估
        model_wrapper = CodeLlamaWrapper(
            lora_path=f"{data_config['paths']['output_dir']}/lora_weights"
        )
        evaluator = Evaluator(model_wrapper)
        metrics = evaluator.evaluate_dataset(data_config["paths"]["processed_test"])
        print("评估结果：")
        print(f"实体F1: {metrics['avg_entity_f1']:.4f}")
        print(f"关系F1: {metrics['avg_relation_f1']:.4f}")
        print(f"推理准确率: {metrics['avg_reasoning_acc']:.4f}")
    
    elif args.step == "app":
        # 启动应用
        import subprocess
        subprocess.run(["streamlit", "run", "./applications/legal_relation_extraction.py"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, required=True, 
                      choices=["preprocess", "train", "evaluate", "app"],
                      help="执行步骤：预处理/训练/评估/应用")
    args = parser.parse_args()
    main(args)