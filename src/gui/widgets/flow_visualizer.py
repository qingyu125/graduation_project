# -*- coding: utf-8 -*-
"""
流程可视化组件
展示DocRED完整的处理流程：文本→伪代码→要素抽取→知识融合→推理验证
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QFrame, QProgressBar, QTextEdit, QGroupBox,
                             QSplitter, QTabWidget, QScrollArea, QPushButton)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPainter, QPen, QBrush, QColor
import json
from typing import Dict, List, Optional
from datetime import datetime


class FlowStepWidget(QFrame):
    """流程步骤显示组件"""
    
    def __init__(self, step_name: str, step_description: str, parent=None):
        super().__init__(parent)
        self.step_name = step_name
        self.step_description = step_description
        self.status = "pending"  # pending, processing, completed, error
        self.result = None
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(2)
        self.setMinimumHeight(80)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 步骤名称
        self.name_label = QLabel(self.step_name)
        self.name_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.name_label)
        
        # 状态标签
        self.status_label = QLabel("等待中")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # 结果显示
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setMaximumHeight(40)
        layout.addWidget(self.result_label)
        
        # 更新样式
        self.update_style()
        
    def set_status(self, status: str):
        """设置状态"""
        self.status = status
        self.update_status_label()
        self.update_style()
        
    def update_status_label(self):
        """更新状态标签"""
        status_texts = {
            "pending": "等待中",
            "processing": "处理中...",
            "completed": "已完成",
            "error": "处理错误"
        }
        self.status_label.setText(status_texts.get(self.status, "未知状态"))
        
    def set_result(self, result):
        """设置结果"""
        self.result = result
        if result:
            if isinstance(result, dict):
                result_text = json.dumps(result, ensure_ascii=False, indent=2)
            elif isinstance(result, list):
                result_text = f"共 {len(result)} 项结果"
            else:
                result_text = str(result)
            # 截断过长的结果
            if len(result_text) > 100:
                result_text = result_text[:97] + "..."
            self.result_label.setText(result_text)
            
    def update_style(self):
        """更新样式"""
        colors = {
            "pending": "#cccccc",     # 灰色
            "processing": "#2196F3",  # 蓝色
            "completed": "#4CAF50",   # 绿色
            "error": "#F44336"        # 红色
        }
        
        color = colors.get(self.status, "#cccccc")
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {color}33;
                border: 2px solid {color};
                border-radius: 8px;
            }}
            QLabel {{
                color: {color};
            }}
        """)


class FlowVisualizer(QWidget):
    """流程可视化组件"""
    
    # 信号定义
    step_completed = pyqtSignal(str, object)  # 步骤完成
    flow_completed = pyqtSignal(dict)  # 流程完成
    error_occurred = pyqtSignal(str, str)  # 错误发生
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.flow_steps = []
        self.current_step_index = 0
        self.step_results = {}
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("DocRED 处理流程")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # 流程显示区域
        flow_group = QGroupBox("处理流程")
        flow_layout = QVBoxLayout(flow_group)
        
        # 步骤容器
        self.steps_container = QVBoxLayout()
        flow_layout.addLayout(self.steps_container)
        
        # 整体进度
        progress_layout = QHBoxLayout()
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 5)
        self.overall_progress.setValue(0)
        progress_layout.addWidget(QLabel("整体进度:"))
        progress_layout.addWidget(self.overall_progress)
        flow_layout.addLayout(progress_layout)
        
        layout.addWidget(flow_group)
        
        # 详细结果区域
        detail_group = QGroupBox("处理详情")
        detail_layout = QVBoxLayout(detail_group)
        
        # 结果标签页
        self.detail_tabs = QTabWidget()
        
        # 伪代码详情
        self.pseudocode_detail = QTextEdit()
        self.pseudocode_detail.setReadOnly(True)
        self.pseudocode_detail.setFont(QFont("Consolas", 9))
        self.detail_tabs.addTab(self.pseudocode_detail, "伪代码")
        
        # 要素详情
        self.elements_detail = QTextEdit()
        self.elements_detail.setReadOnly(True)
        self.elements_detail.setFont(QFont("Consolas", 9))
        self.detail_tabs.addTab(self.elements_detail, "抽取要素")
        
        # 融合结果
        self.fusion_detail = QTextEdit()
        self.fusion_detail.setReadOnly=True
        self.fusion_detail.setFont(QFont("Consolas", 9))
        self.detail_tabs.addTab(self.fusion_detail, "知识融合")
        
        # 验证结果
        self.verification_detail = QTextEdit()
        self.verification_detail.setReadOnly(True)
        self.verification_detail.setFont(QFont("Consolas", 9))
        self.detail_tabs.addTab(self.verification_detail, "推理验证")
        
        detail_layout.addWidget(self.detail_tabs)
        layout.addWidget(detail_group)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("重置流程")
        self.reset_btn.clicked.connect(self.reset_flow)
        self.export_btn = QPushButton("导出详情")
        self.export_btn.clicked.connect(self.export_details)
        
        control_layout.addWidget(self.reset_btn)
        control_layout.addWidget(self.export_btn)
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
    def initialize_flow(self, text: str, model_config: dict):
        """初始化流程"""
        self.clear_flow()
        
        # 定义流程步骤
        steps = [
            ("文本预处理", "清理和标准化输入文本"),
            ("文本到伪代码", "将自然语言转换为结构化伪代码"),
            ("要素抽取", "从伪代码中抽取实体和关系要素"),
            ("知识融合", "整合和关联抽取的要素信息"),
            ("推理验证", "推理和验证关系三元组的正确性")
        ]
        
        # 创建步骤组件
        for step_name, step_desc in steps:
            step_widget = FlowStepWidget(step_name, step_desc)
            self.steps_container.addWidget(step_widget)
            self.flow_steps.append(step_widget)
            
        # 添加连接线（可选）
        self.add_flow_connections()
        
        # 更新进度
        self.overall_progress.setValue(0)
        
        # 存储初始信息
        self.step_results["initial_text"] = text
        self.step_results["model_config"] = model_config
        
    def add_flow_connections(self):
        """添加流程连接线"""
        # 这里可以添加视觉连接线，暂时使用占位符
        for i in range(len(self.flow_steps) - 1):
            connector = QLabel("↓")
            connector.setAlignment(Qt.AlignmentFlag.AlignCenter)
            connector.setFont(QFont("Arial", 16))
            self.steps_container.addWidget(connector)
            
    def start_step(self, step_name: str):
        """开始处理步骤"""
        step_index = self.get_step_index(step_name)
        if step_index >= 0 and step_index < len(self.flow_steps):
            self.current_step_index = step_index
            self.flow_steps[step_index].set_status("processing")
            
    def complete_step(self, step_name: str, result):
        """完成步骤"""
        step_index = self.get_step_index(step_name)
        if step_index >= 0 and step_index < len(self.flow_steps):
            # 设置状态和结果
            self.flow_steps[step_index].set_status("completed")
            self.flow_steps[step_index].set_result(result)
            
            # 存储结果
            self.step_results[step_name] = result
            
            # 更新进度
            completed_steps = sum(1 for step in self.flow_steps if step.status == "completed")
            self.overall_progress.setValue(completed_steps)
            
            # 更新详情显示
            self.update_details_display(step_name, result)
            
            # 发送信号
            self.step_completed.emit(step_name, result)
            
    def error_step(self, step_name: str, error_message: str):
        """步骤出错"""
        step_index = self.get_step_index(step_name)
        if step_index >= 0 and step_index < len(self.flow_steps):
            self.flow_steps[step_index].set_status("error")
            self.flow_steps[step_index].set_result(error_message)
            self.error_occurred.emit(step_name, error_message)
            
    def update_details_display(self, step_name: str, result):
        """更新详情显示"""
        if step_name == "文本到伪代码":
            self.pseudocode_detail.setText(str(result) if result else "无伪代码")
        elif step_name == "要素抽取":
            if isinstance(result, list):
                elements_text = "抽取的要素:\n"
                for i, elem in enumerate(result):
                    elements_text += f"{i+1}. {elem}\n"
                self.elements_detail.setText(elements_text)
            else:
                self.elements_detail.setText(str(result))
        elif step_name == "知识融合":
            if isinstance(result, dict):
                fusion_text = f"知识融合结果:\n"
                fusion_text += f"实体数量: {len(result.get('entities', []))}\n"
                fusion_text += f"关系数量: {len(result.get('relations', []))}\n"
                fusion_text += f"置信度: {result.get('confidence_score', 0.0):.3f}\n"
                self.fusion_detail.setText(fusion_text)
            else:
                self.fusion_detail.setText(str(result))
        elif step_name == "推理验证":
            if isinstance(result, list):
                verification_text = f"验证结果 ({len(result)} 条):\n"
                for i, res in enumerate(result):
                    verification_text += f"{i+1}. {res}\n"
                self.verification_detail.setText(verification_text)
            else:
                self.verification_detail.setText(str(result))
                
    def get_step_index(self, step_name: str) -> int:
        """获取步骤索引"""
        step_mapping = {
            "文本预处理": 0,
            "文本到伪代码": 1,
            "要素抽取": 2,
            "知识融合": 3,
            "推理验证": 4
        }
        return step_mapping.get(step_name, -1)
        
    def complete_flow(self):
        """完成整个流程"""
        self.overall_progress.setValue(len(self.flow_steps))
        self.flow_completed.emit(self.step_results)
        
    def reset_flow(self):
        """重置流程"""
        self.clear_flow()
        
    def clear_flow(self):
        """清空流程"""
        # 清空步骤组件
        for step in self.flow_steps:
            step.setParent(None)
            step.deleteLater()
        self.flow_steps.clear()
        
        # 清空详情显示
        self.pseudocode_detail.clear()
        self.elements_detail.clear()
        self.fusion_detail.clear()
        self.verification_detail.clear()
        
        # 重置进度
        self.overall_progress.setValue(0)
        
        # 清空结果
        self.step_results.clear()
        self.current_step_index = 0
        
    def export_details(self):
        """导出详情"""
        if not self.step_results:
            return
            
        from PyQt6.QtWidgets import QFileDialog
        from PyQt6.QtCore import QStandardPaths
        import os
        
        default_path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        filename, _ = QFileDialog.getSaveFileName(
            self, "导出流程详情", 
            os.path.join(default_path, f"docred_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
            "JSON文件 (*.json)"
        )
        
        if filename:
            try:
                export_data = {
                    'export_time': datetime.now().isoformat(),
                    'flow_steps': self.step_results
                }
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"导出失败: {str(e)}")
                
    def get_flow_results(self) -> dict:
        """获取流程结果"""
        return self.step_results.copy()
        
    def is_flow_completed(self) -> bool:
        """检查流程是否完成"""
        return self.overall_progress.value() == len(self.flow_steps)
