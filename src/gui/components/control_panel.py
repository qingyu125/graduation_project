# -*- coding: utf-8 -*-
"""
控制面板组件
控制整个处理流程的启动、暂停、停止等操作
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QGroupBox, QLabel, QCheckBox, QSpinBox, 
                             QDoubleSpinBox, QComboBox, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QFont, QIcon
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
from .enhanced_processor import RealProcessingWorker


class ProcessingWorker(QThread):
    """文档处理工作线程"""
    
    progress_updated = pyqtSignal(int, str)  # 进度更新 (progress, message)
    result_generated = pyqtSignal(dict)  # 生成结果
    processing_finished = pyqtSignal(list)  # 处理完成
    error_occurred = pyqtSignal(str)  # 错误发生
    status_changed = pyqtSignal(str)  # 状态变化
    
    def __init__(self, text: str, model_config: dict, process_options: dict):
        super().__init__()
        self.text = text
        self.model_config = model_config
        self.process_options = process_options
        self.is_paused = False
        self.is_stopped = False
        self.results = []
        
    def run(self):
        """运行文档处理流程"""
        try:
            self.status_changed.emit("开始处理文档...")
            
            # 步骤1: 文本预处理
            self.progress_updated.emit(10, "正在进行文本预处理...")
            self.msleep(200)  # 模拟处理时间
            processed_text = self.preprocess_text(self.text)
            
            # 步骤2: 文本到伪代码转换
            self.progress_updated.emit(25, "正在转换文本为伪代码...")
            self.msleep(500)  # 模拟处理时间
            pseudocode = self.text_to_pseudocode(processed_text)
            
            # 步骤3: 要素抽取
            self.progress_updated.emit(50, "正在进行要素抽取...")
            self.msleep(400)  # 模拟处理时间
            elements = self.extract_elements(pseudocode)
            
            # 步骤4: 知识融合
            self.progress_updated.emit(70, "正在进行知识融合...")
            self.msleep(300)  # 模拟处理时间
            fused_knowledge = self.fuse_knowledge(elements)
            
            # 步骤5: 推理验证
            self.progress_updated.emit(90, "正在进行推理验证...")
            self.msleep(200)  # 模拟处理时间
            verified_results = self.reason_and_verify(fused_knowledge, processed_text)
            
            self.progress_updated.emit(100, "处理完成")
            self.processing_finished.emit(verified_results)
            
        except Exception as e:
            self.error_occurred.emit(f"处理过程中发生错误: {str(e)}")
            
    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if self.is_stopped:
            return ""
            
        # 模拟文本清理和处理
        processed = text.strip().replace('\n\n\n', '\n\n')
        return processed
        
    def text_to_pseudocode(self, text: str) -> str:
        """文本到伪代码转换"""
        if self.is_stopped:
            return ""
            
        # 模拟伪代码生成
        model_type = self.model_config.get('type', 'codelama_lora')
        if model_type == 'codelama_lora':
            return self.generate_codelama_pseudocode(text)
        elif model_type == 'bert_finetune':
            return self.generate_bert_pseudocode(text)
        else:
            return self.generate_rule_pseudocode(text)
            
    def generate_codelama_pseudocode(self, text: str) -> str:
        """生成CodeLlama伪代码"""
        pseudocode = f"""
# CodeLlama-7B + LoRA 处理流程
def process_document(text):
    # 实体识别
    entities = extract_entities(text)
    
    # 关系抽取
    relations = extract_relations(entities, text)
    
    # 结果整合
    results = integrate_results(entities, relations)
    
    return results
"""
        return pseudocode
        
    def generate_bert_pseudocode(self, text: str) -> str:
        """生成BERT伪代码"""
        pseudocode = f"""
# BERT微调模型处理流程  
def process_document(text):
    # 文本编码
    encoded = bert_encode(text)
    
    # 关系分类
    predictions = bert_classify(encoded)
    
    # 后处理
    results = post_process(predictions)
    
    return results
"""
        return pseudocode
        
    def generate_rule_pseudocode(self, text: str) -> str:
        """生成规则方法伪代码"""
        pseudocode = f"""
# 规则方法处理流程
def process_document(text):
    # 模式匹配
    patterns = match_patterns(text)
    
    # 规则应用
    relations = apply_rules(patterns)
    
    # 结果筛选
    results = filter_results(relations)
    
    return results
"""
        return pseudocode
        
    def extract_elements(self, pseudocode: str) -> List[Dict]:
        """要素抽取"""
        if self.is_stopped:
            return []
            
        # 模拟要素抽取
        elements = [
            {
                'type': 'entity',
                'value': 'John Smith',
                'position': [0, 10],
                'confidence': 0.95
            },
            {
                'type': 'relation', 
                'value': 'works_for',
                'confidence': 0.88
            },
            {
                'type': 'entity',
                'value': 'TechCorp',
                'position': [25, 33],
                'confidence': 0.92
            }
        ]
        return elements
        
    def fuse_knowledge(self, elements: List[Dict]) -> Dict:
        """知识融合"""
        if self.is_stopped:
            return {}
            
        # 模拟知识融合
        fused = {
            'entities': [elem for elem in elements if elem['type'] == 'entity'],
            'relations': [elem for elem in elements if elem['type'] == 'relation'],
            'context': '公司关系',
            'confidence_score': 0.85
        }
        return fused
        
    def reason_and_verify(self, fused_knowledge: Dict, original_text: str) -> List[Dict]:
        """推理验证"""
        if self.is_stopped:
            return []
            
        # 模拟推理验证过程，生成最终结果
        results = []
        
        # 生成关系三元组
        entities = fused_knowledge.get('entities', [])
        relations = fused_knowledge.get('relations', [])
        
        for i, relation in enumerate(relations):
            if i < len(entities) - 1:
                result = {
                    'text': original_text[:50] + "...",
                    'head_entity': entities[i].get('value', 'Unknown'),
                    'relation': relation.get('value', 'unknown'),
                    'tail_entity': entities[i + 1].get('value', 'Unknown'),
                    'confidence': relation.get('confidence', 0.5),
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'model_used': self.model_config.get('type', 'unknown'),
                    'original_text': original_text,
                    'pseudocode': self.get_pseudocode(),
                    'elements': entities,
                    'process_info': {
                        'text_preprocessing': 'completed',
                        'pseudocode_generation': 'completed',
                        'element_extraction': 'completed',
                        'knowledge_fusion': 'completed',
                        'reasoning_verification': 'completed'
                    }
                }
                results.append(result)
                
        return results
        
    def get_pseudocode(self) -> str:
        """获取伪代码"""
        return f"""
# {self.model_config.get('type', 'unknown')} 处理结果
def final_processing():
    # 处理完成
    return processed_results
"""
        
    def pause(self):
        """暂停处理"""
        self.is_paused = True
        self.status_changed.emit("已暂停")
        
    def resume(self):
        """恢复处理"""
        self.is_paused = False
        self.status_changed.emit("正在处理")
        
    def stop(self):
        """停止处理"""
        self.is_stopped = True
        self.status_changed.emit("已停止")


class ControlPanel(QWidget):
    """控制面板，管理整个处理流程"""
    
    # 信号定义
    processing_started = pyqtSignal(str, dict, dict)  # 开始处理 (text, model_config, options)
    processing_paused = pyqtSignal()  # 暂停处理
    processing_resumed = pyqtSignal()  # 恢复处理
    processing_stopped = pyqtSignal()  # 停止处理
    settings_changed = pyqtSignal(dict)  # 设置变化
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.enhanced_worker = None
        self.is_processing = False
        self.process_options = {}
        self.flow_visualizer = None
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("处理控制")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # 主控制按钮区域
        control_group = QGroupBox("处理控制")
        control_layout = QVBoxLayout(control_group)
        
        # 主要控制按钮
        main_buttons = QHBoxLayout()
        
        self.start_btn = QPushButton("开始处理")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.start_btn.clicked.connect(self.start_processing)
        
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.pause_processing)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_processing)
        
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_processing)
        
        main_buttons.addWidget(self.start_btn)
        main_buttons.addWidget(self.pause_btn)
        main_buttons.addWidget(self.stop_btn)
        main_buttons.addWidget(self.reset_btn)
        control_layout.addLayout(main_buttons)
        
        # 处理选项
        options_group = QGroupBox("处理选项")
        options_layout = QVBoxLayout(options_group)
        
        # 自动保存结果
        self.auto_save_cb = QCheckBox("自动保存结果")
        self.auto_save_cb.setChecked(True)
        self.auto_save_cb.toggled.connect(self.on_setting_changed)
        
        # 显示详细日志
        self.detailed_log_cb = QCheckBox("显示详细处理日志")
        self.detailed_log_cb.setChecked(True)
        self.detailed_log_cb.toggled.connect(self.on_setting_changed)
        
        # 批量处理
        self.batch_process_cb = QCheckBox("启用批量处理")
        self.batch_process_cb.setChecked(False)
        self.batch_process_cb.toggled.connect(self.on_setting_changed)
        
        options_layout.addWidget(self.auto_save_cb)
        options_layout.addWidget(self.detailed_log_cb)
        options_layout.addWidget(self.batch_process_cb)
        
        # 高级设置
        advanced_layout = QHBoxLayout()
        
        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setRange(1, 8)
        self.max_workers_spin.setValue(2)
        self.max_workers_spin.valueChanged.connect(self.on_setting_changed)
        
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(10, 300)
        self.timeout_spin.setValue(60)
        self.timeout_spin.setSuffix(" 秒")
        self.timeout_spin.valueChanged.connect(self.on_setting_changed)
        
        advanced_layout.addWidget(QLabel("最大工作线程:"))
        advanced_layout.addWidget(self.max_workers_spin)
        advanced_layout.addWidget(QLabel("超时时间:"))
        advanced_layout.addWidget(self.timeout_spin)
        advanced_layout.addStretch()
        
        options_layout.addLayout(advanced_layout)
        control_layout.addWidget(options_group)
        
        # 状态显示
        status_group = QGroupBox("处理状态")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.process_time_label = QLabel("处理时间: 00:00")
        self.process_time_label.setStyleSheet("color: #666;")
        self.process_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.process_time_label)
        control_layout.addWidget(status_group)
        
        layout.addWidget(control_group)
        layout.addStretch()
        
        # 计时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_process_time)
        self.start_time = None
        
    def start_processing(self):
        """开始处理"""
        if self.is_processing:
            return
            
        # 验证输入
        text = getattr(self.parent(), 'get_text', lambda: "")()
        if not text or not text.strip():
            QMessageBox.warning(self, "警告", "请先输入要处理的文本")
            return
            
        # 获取模型配置和设置
        model_config = self.get_model_config()
        process_options = self.get_process_options()
        
        # 获取流程可视化器
        if hasattr(self.parent(), 'flow_visualizer'):
            self.flow_visualizer = self.parent().flow_visualizer
        else:
            self.flow_visualizer = None
            
        # 更新UI状态
        self.is_processing = True
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("处理中...")
        self.status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        
        # 启动计时器
        self.start_time = time.time()
        self.timer.start(1000)
        
        # 启动真实工作线程
        self.enhanced_worker = RealProcessingWorker(
            text, model_config, process_options, self.flow_visualizer
        )
        
        # 连接信号
        self.enhanced_worker.step_started.connect(self.on_step_started)
        self.enhanced_worker.step_completed.connect(self.on_step_completed)
        self.enhanced_worker.step_error.connect(self.on_step_error)
        self.enhanced_worker.progress_updated.connect(self.on_progress_updated)
        self.enhanced_worker.processing_finished.connect(self.on_processing_finished)
        self.enhanced_worker.error_occurred.connect(self.on_processing_error)
        self.enhanced_worker.status_changed.connect(self.on_status_changed)
        
        self.enhanced_worker.start()
        self.processing_started.emit(text, model_config, process_options)
        
    def pause_processing(self):
        """暂停处理"""
        if self.enhanced_worker and self.enhanced_worker.isRunning():
            if self.enhanced_worker.is_paused:
                self.enhanced_worker.resume()
                self.pause_btn.setText("暂停")
                self.status_label.setText("处理中...")
                self.processing_resumed.emit()
            else:
                self.enhanced_worker.pause()
                self.pause_btn.setText("恢复")
                self.status_label.setText("已暂停")
                self.processing_paused.emit()
                
    def stop_processing(self):
        """停止处理"""
        if self.enhanced_worker and self.enhanced_worker.isRunning():
            self.enhanced_worker.stop()
            self.enhanced_worker.wait()
            
        self.reset_processing()
        self.processing_stopped.emit()
        
    def reset_processing(self):
        """重置处理状态"""
        self.is_processing = False
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("暂停")
        self.stop_btn.setEnabled(False)
        self.status_label.setText("就绪")
        self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.process_time_label.setText("处理时间: 00:00")
        
        if self.enhanced_worker:
            self.enhanced_worker = None
            
        if self.start_time:
            self.timer.stop()
            self.start_time = None
            
    def on_progress_updated(self, progress: int, message: str):
        """进度更新处理"""
        # 可以连接进度面板的更新
        pass
        
    def on_step_started(self, step_name: str):
        """步骤开始"""
        if self.flow_visualizer:
            self.flow_visualizer.start_step(step_name)
            
    def on_step_completed(self, step_name: str, result):
        """步骤完成"""
        if self.flow_visualizer:
            self.flow_visualizer.complete_step(step_name, result)
            
    def on_step_error(self, step_name: str, error_message: str):
        """步骤错误"""
        if self.flow_visualizer:
            self.flow_visualizer.error_step(step_name, error_message)
            
    def on_processing_finished(self, results: List[Dict]):
        """处理完成"""
        if self.flow_visualizer:
            self.flow_visualizer.complete_flow()
            
        self.reset_processing()
        QMessageBox.information(
            self, "处理完成", 
            f"文档处理完成，共生成 {len(results)} 条关系"
        )
        
    def on_processing_error(self, error_message: str):
        """处理错误"""
        self.reset_processing()
        QMessageBox.critical(self, "处理错误", error_message)
        
    def on_status_changed(self, status: str):
        """状态变化"""
        self.status_label.setText(status)
        
    def update_process_time(self):
        """更新处理时间"""
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            self.process_time_label.setText(f"处理时间: {minutes:02d}:{seconds:02d}")
            
    def on_setting_changed(self):
        """设置变化"""
        self.process_options = self.get_process_options()
        self.settings_changed.emit(self.process_options)
        
    def get_model_config(self) -> dict:
        """获取模型配置"""
        # 从父窗口获取模型选择器的配置
        parent = self.parent()
        if hasattr(parent, 'get_model_config'):
            return parent.get_model_config()
        return {}
        
    def get_process_options(self) -> dict:
        """获取处理选项"""
        return {
            'auto_save': self.auto_save_cb.isChecked(),
            'detailed_log': self.detailed_log_cb.isChecked(),
            'batch_process': self.batch_process_cb.isChecked(),
            'max_workers': self.max_workers_spin.value(),
            'timeout': self.timeout_spin.value()
        }
        
    def is_currently_processing(self) -> bool:
        """检查是否正在处理"""
        return self.is_processing
        
    def get_processing_status(self) -> str:
        """获取处理状态"""
        return self.status_label.text()