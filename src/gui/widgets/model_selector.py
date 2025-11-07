# -*- coding: utf-8 -*-
"""
模型选择器控件
支持三个不同模型的切换和配置
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
                             QLabel, QGroupBox, QPushButton, QSpinBox, 
                             QDoubleSpinBox, QCheckBox, QSlider, QFormLayout,
                             QScrollArea, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QIcon
import json
import os
import sys
import logging
from typing import Dict, Any

# # 添加路径以导入模型推理模块
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
# from src.models.inference import get_model_manager
# 获取当前文件（model_selector.py）的目录
current_file_dir = os.path.dirname(os.path.abspath(__file__))  # .../src/widgets/

# 计算 src 目录的父目录（即项目中包含 src 的那个目录，如 docred_paper/）
src_parent_dir = os.path.abspath(os.path.join(current_file_dir, '../../..'))  
# 解析：../../.. 表示从 widgets/ 向上退3级：widgets → src → docred_paper → 得到 docred_paper/

# 将 src 的父目录添加到搜索路径
sys.path.append(src_parent_dir)

# 此时可直接从 src 导入（因为 src 已在 Python 搜索路径的目录下）
from src.models.inference import get_model_manager

logger = logging.getLogger(__name__)


class ModelSelector(QWidget):
    """模型选择器，支持三个不同模型"""
    
    # 信号定义
    model_changed = pyqtSignal(str)  # 模型变化信号
    config_changed = pyqtSignal(dict)  # 配置变化信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_model = "model1"
        self.model_configs = {}
        self.setup_ui()
        self.load_default_configs()
        
    def setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("模型选择与配置")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # 滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # 主内容部件
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # 模型选择组
        model_group = QGroupBox("模型选择")
        model_layout = QFormLayout(model_group)
        
        # 模型下拉框
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "CodeLlama-7B + LoRA (推荐)",
            "BERT + 微调",
            "传统规则方法"
        ])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addRow("模型类型:", self.model_combo)
        
        # 模型状态
        self.model_status = QLabel("未加载")
        self.model_status.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        model_layout.addRow("状态:", self.model_status)
        
        content_layout.addWidget(model_group)
        
        # CodeLlama配置组
        self.codelama_group = QGroupBox("CodeLlama配置")
        self.setup_codelama_config()
        content_layout.addWidget(self.codelama_group)
        
        # BERT配置组
        self.bert_group = QGroupBox("BERT配置")
        self.setup_bert_config()
        content_layout.addWidget(self.bert_group)
        
        # 规则方法配置组
        self.rule_group = QGroupBox("规则方法配置")
        self.setup_rule_config()
        content_layout.addWidget(self.rule_group)
        
        # 通用配置组
        self.general_group = QGroupBox("通用配置")
        self.setup_general_config()
        content_layout.addWidget(self.general_group)
        
        # 按钮组
        button_layout = QHBoxLayout()
        
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_model)
        self.test_model_btn = QPushButton("测试模型")
        self.test_model_btn.clicked.connect(self.test_model)
        self.reset_config_btn = QPushButton("重置配置")
        self.reset_config_btn.clicked.connect(self.reset_config)
        
        button_layout.addWidget(self.load_model_btn)
        button_layout.addWidget(self.test_model_btn)
        button_layout.addWidget(self.reset_config_btn)
        button_layout.addStretch()
        
        content_layout.addLayout(button_layout)
        content_layout.addStretch()
        
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        
        # 初始化显示
        self.update_visibility()
        
    def setup_codelama_config(self):
        """设置CodeLlama配置"""
        layout = QFormLayout(self.codelama_group)
        
        # LoRA配置
        self.lora_rank = QSpinBox()
        self.lora_rank.setRange(1, 64)
        self.lora_rank.setValue(8)
        self.lora_rank.valueChanged.connect(self.on_config_changed)
        layout.addRow("LoRA秩:", self.lora_rank)
        
        self.lora_alpha = QSpinBox()
        self.lora_alpha.setRange(1, 128)
        self.lora_alpha.setValue(32)
        self.lora_alpha.valueChanged.connect(self.on_config_changed)
        layout.addRow("LoRA α:", self.lora_alpha)
        
        # 学习率
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.0001, 0.1)
        self.learning_rate.setValue(0.0002)
        self.learning_rate.setDecimals(6)
        self.learning_rate.valueChanged.connect(self.on_config_changed)
        layout.addRow("学习率:", self.learning_rate)
        
        # 批大小
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 32)
        self.batch_size.setValue(4)
        self.batch_size.valueChanged.connect(self.on_config_changed)
        layout.addRow("批大小:", self.batch_size)
        
        # 4位量化
        self.use_4bit = QCheckBox("启用4位量化")
        self.use_4bit.setChecked(True)
        self.use_4bit.toggled.connect(self.on_config_changed)
        layout.addRow("量化:", self.use_4bit)
        
    def setup_bert_config(self):
        """设置BERT配置"""
        layout = QFormLayout(self.bert_group)
        
        # BERT模型大小
        self.bert_size = QComboBox()
        self.bert_size.addItems(["bert-base-chinese", "bert-large-chinese"])
        self.bert_size.currentTextChanged.connect(self.on_config_changed)
        layout.addRow("BERT模型:", self.bert_size)
        
        # 隐藏层大小
        self.hidden_size = QSpinBox()
        self.hidden_size.setRange(128, 1024)
        self.hidden_size.setValue(768)
        self.hidden_size.valueChanged.connect(self.on_config_changed)
        layout.addRow("隐藏层大小:", self.hidden_size)
        
        # 训练轮数
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 50)
        self.epochs.setValue(10)
        self.epochs.valueChanged.connect(self.on_config_changed)
        layout.addRow("训练轮数:", self.epochs)
        
        # Dropout
        self.dropout = QDoubleSpinBox()
        self.dropout.setRange(0.0, 0.9)
        self.dropout.setValue(0.1)
        self.dropout.setSingleStep(0.1)
        self.dropout.valueChanged.connect(self.on_config_changed)
        layout.addRow("Dropout:", self.dropout)
        
    def setup_rule_config(self):
        """设置规则方法配置"""
        layout = QFormLayout(self.rule_group)
        
        # 规则强度
        self.rule_strength = QSlider(Qt.Orientation.Horizontal)
        self.rule_strength.setRange(1, 10)
        self.rule_strength.setValue(5)
        self.rule_strength.valueChanged.connect(self.on_config_changed)
        layout.addRow("规则强度:", self.rule_strength)
        
        # 置信度阈值
        self.confidence_threshold = QDoubleSpinBox()
        self.confidence_threshold.setRange(0.0, 1.0)
        self.confidence_threshold.setValue(0.7)
        self.confidence_threshold.setSingleStep(0.1)
        self.confidence_threshold.valueChanged.connect(self.on_config_changed)
        layout.addRow("置信度阈值:", self.confidence_threshold)
        
        # 启用预定义规则
        self.use_predefined = QCheckBox("启用预定义规则")
        self.use_predefined.setChecked(True)
        self.use_predefined.toggled.connect(self.on_config_changed)
        layout.addRow("预定义规则:", self.use_predefined)
        
        # 启用模式匹配
        self.use_pattern = QCheckBox("启用模式匹配")
        self.use_pattern.setChecked(True)
        self.use_pattern.toggled.connect(self.on_config_changed)
        layout.addRow("模式匹配:", self.use_pattern)
        
    def setup_general_config(self):
        """设置通用配置"""
        layout = QFormLayout(self.general_group)
        
        # 最大序列长度
        self.max_length = QSpinBox()
        self.max_length.setRange(64, 2048)
        self.max_length.setValue(512)
        self.max_length.valueChanged.connect(self.on_config_changed)
        layout.addRow("最大序列长度:", self.max_length)
        
        # GPU使用
        self.use_gpu = QCheckBox("使用GPU加速")
        self.use_gpu.setChecked(True)
        self.use_gpu.toggled.connect(self.on_config_changed)
        layout.addRow("GPU加速:", self.use_gpu)
        
        # 线程数
        self.num_threads = QSpinBox()
        self.num_threads.setRange(1, 16)
        self.num_threads.setValue(4)
        self.num_threads.valueChanged.connect(self.on_config_changed)
        layout.addRow("并发线程:", self.num_threads)
        
    def load_default_configs(self):
        """加载默认配置"""
        self.model_configs = {
            "model1": {  # CodeLlama + LoRA
                "type": "codelama_lora",
                "lora_rank": 8,
                "lora_alpha": 32,
                "learning_rate": 0.0002,
                "batch_size": 4,
                "use_4bit": True,
                "max_length": 512,
                "use_gpu": True,
                "num_threads": 4
            },
            "model2": {  # BERT微调
                "type": "bert_finetune",
                "bert_size": "bert-base-chinese",
                "hidden_size": 768,
                "epochs": 10,
                "dropout": 0.1,
                "learning_rate": 0.0001,
                "batch_size": 16,
                "max_length": 512,
                "use_gpu": True,
                "num_threads": 4
            },
            "model3": {  # 规则方法
                "type": "rule_based",
                "rule_strength": 5,
                "confidence_threshold": 0.7,
                "use_predefined": True,
                "use_pattern": True,
                "max_length": 512,
                "use_gpu": False,
                "num_threads": 2
            }
        }
        
        # 设置默认模型
        self.set_model("model1")
        
    def on_model_changed(self, model_name):
        """模型选择变化处理"""
        model_map = {
            "CodeLlama-7B + LoRA (推荐)": "model1",
            "BERT + 微调": "model2", 
            "传统规则方法": "model3"
        }
        
        model_id = model_map.get(model_name, "model1")
        self.set_model(model_id)
        
    def set_model(self, model_id):
        """设置当前模型"""
        self.current_model = model_id
        self.model_changed.emit(model_id)
        self.update_visibility()
        self.load_config_to_ui()
        
    def update_visibility(self):
        """更新界面可见性"""
        # 隐藏所有配置组
        self.codelama_group.setVisible(False)
        self.bert_group.setVisible(False)
        self.rule_group.setVisible(False)
        
        # 根据模型类型显示相应配置
        if self.current_model == "model1":
            self.codelama_group.setVisible(True)
        elif self.current_model == "model2":
            self.bert_group.setVisible(True)
        elif self.current_model == "model3":
            self.rule_group.setVisible(True)
            
    def load_config_to_ui(self):
        """从配置加载到UI"""
        config = self.model_configs.get(self.current_model, {})
        
        if self.current_model == "model1":
            self.lora_rank.setValue(config.get("lora_rank", 8))
            self.lora_alpha.setValue(config.get("lora_alpha", 32))
            self.learning_rate.setValue(config.get("learning_rate", 0.0002))
            self.batch_size.setValue(config.get("batch_size", 4))
            self.use_4bit.setChecked(config.get("use_4bit", True))
            
        elif self.current_model == "model2":
            self.bert_size.setCurrentText(config.get("bert_size", "bert-base-chinese"))
            self.hidden_size.setValue(config.get("hidden_size", 768))
            self.epochs.setValue(config.get("epochs", 10))
            self.dropout.setValue(config.get("dropout", 0.1))
            
        elif self.current_model == "model3":
            self.rule_strength.setValue(config.get("rule_strength", 5))
            self.confidence_threshold.setValue(config.get("confidence_threshold", 0.7))
            self.use_predefined.setChecked(config.get("use_predefined", True))
            self.use_pattern.setChecked(config.get("use_pattern", True))
            
        # 通用配置
        self.max_length.setValue(config.get("max_length", 512))
        self.use_gpu.setChecked(config.get("use_gpu", True))
        self.num_threads.setValue(config.get("num_threads", 4))
        
    def on_config_changed(self):
        """配置变化处理"""
        self.collect_config_from_ui()
        self.config_changed.emit(self.get_current_config())
        
    def collect_config_from_ui(self):
        """从UI收集配置"""
        if self.current_model not in self.model_configs:
            self.model_configs[self.current_model] = {}
            
        config = self.model_configs[self.current_model]
        
        if self.current_model == "model1":
            config.update({
                "lora_rank": self.lora_rank.value(),
                "lora_alpha": self.lora_alpha.value(),
                "learning_rate": self.learning_rate.value(),
                "batch_size": self.batch_size.value(),
                "use_4bit": self.use_4bit.isChecked()
            })
            
        elif self.current_model == "model2":
            config.update({
                "bert_size": self.bert_size.currentText(),
                "hidden_size": self.hidden_size.value(),
                "epochs": self.epochs.value(),
                "dropout": self.dropout.value()
            })
            
        elif self.current_model == "model3":
            config.update({
                "rule_strength": self.rule_strength.value(),
                "confidence_threshold": self.confidence_threshold.value(),
                "use_predefined": self.use_predefined.isChecked(),
                "use_pattern": self.use_pattern.isChecked()
            })
            
        # 通用配置
        config.update({
            "max_length": self.max_length.value(),
            "use_gpu": self.use_gpu.isChecked(),
            "num_threads": self.num_threads.value()
        })
        
    def load_model(self):
        """加载模型"""
        self.model_status.setText("加载中...")
        self.model_status.setStyleSheet("color: #ffa726; font-weight: bold;")
        
        # 使用真实模型加载
        try:
            model_manager = get_model_manager()
            
            # 根据当前模型类型映射到推理引擎类型
            model_type_map = {
                "model1": "codelama_finetuned",  # CodeLlama + LoRA
                "model2": "bert_finetuned",      # BERT微调
                "model3": "codelama_base"        # 规则方法(使用未微调的CodeLlama)
            }
            
            engine_type = model_type_map.get(self.current_model, "codelama_finetuned")
            
            # 实际加载模型
            success = model_manager.load_model(engine_type)
            
            if success:
                self.on_model_loaded()
            else:
                self.on_model_load_failed()
                
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.on_model_load_failed(str(e))
        
    def on_model_loaded(self):
        """模型加载完成"""
        self.model_status.setText("已加载")
        self.model_status.setStyleSheet("color: #4caf50; font-weight: bold;")
        
    def on_model_load_failed(self, error_msg=""):
        """模型加载失败"""
        status = f"加载失败"
        if error_msg:
            status += f": {error_msg}"
        self.model_status.setText(status)
        self.model_status.setStyleSheet("color: #f44336; font-weight: bold;")
        
    def test_model(self):
        """测试模型"""
        self.model_status.setText("测试中...")
        self.model_status.setStyleSheet("color: #2196f3; font-weight: bold;")
        
        # 实际测试模型
        try:
            model_manager = get_model_manager()
            
            # 检查模型是否已加载
            model_type_map = {
                "model1": "codelama_finetuned",
                "model2": "bert_finetuned", 
                "model3": "codelama_base"
            }
            
            engine_type = model_type_map.get(self.current_model, "codelama_finetuned")
            
            if not model_manager.is_model_loaded(engine_type):
                self.model_status.setText("模型未加载")
                self.model_status.setStyleSheet("color: #ff9800; font-weight: bold;")
                return
                
            # 测试文本
            test_text = "Apple Inc. is a technology company founded by Steve Jobs. Tim Cook is the current CEO."
            
            # 进行预测
            result = model_manager.predict_with_model(engine_type, test_text)
            
            if result.get("success", False):
                self.model_status.setText("测试完成 ✓")
                self.model_status.setStyleSheet("color: #4caf50; font-weight: bold;")
            else:
                self.model_status.setText(f"测试失败: {result.get('error', '未知错误')}")
                self.model_status.setStyleSheet("color: #f44336; font-weight: bold;")
                
        except Exception as e:
            logger.error(f"模型测试失败: {e}")
            self.model_status.setText(f"测试失败: {str(e)}")
            self.model_status.setStyleSheet("color: #f44336; font-weight: bold;")
        
    def reset_config(self):
        """重置配置"""
        self.load_default_configs()
        self.load_config_to_ui()
        
    def get_current_config(self):
        """获取当前配置"""
        self.collect_config_from_ui()
        return self.model_configs.get(self.current_model, {}).copy()
        
    def get_current_model_id(self):
        """获取当前模型ID"""
        return self.current_model
        
    def save_config(self, filepath):
        """保存配置到文件"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.model_configs, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False
            
    def load_config(self, filepath):
        """从文件加载配置"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.model_configs = json.load(f)
            self.load_config_to_ui()
            return True
        except Exception:
            return False