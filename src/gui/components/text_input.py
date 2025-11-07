# -*- coding: utf-8 -*-
"""
文本输入组件
支持多行文本输入、文件导入、文本预处理
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
                             QPushButton, QGroupBox, QLabel, QFileDialog,
                             QLineEdit, QComboBox, QCheckBox, QSpinBox,
                             QProgressBar, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QFont, QTextCursor
import re
import os
from typing import List, Optional


class TextProcessorWorker(QThread):
    """文本处理工作线程"""
    
    progress_updated = pyqtSignal(int)  # 进度更新
    text_processed = pyqtSignal(str)  # 文本处理完成
    error_occurred = pyqtSignal(str)  # 错误发生
    
    def __init__(self, text: str, process_options: dict):
        super().__init__()
        self.text = text
        self.process_options = process_options
        
    def run(self):
        """运行文本处理"""
        try:
            # 模拟文本处理过程
            lines = self.text.split('\n')
            total_lines = len(lines)
            processed_text = ""
            
            for i, line in enumerate(lines):
                # 应用各种处理选项
                processed_line = self.process_line(line)
                processed_text += processed_line + '\n'
                
                # 更新进度
                progress = int((i + 1) / total_lines * 100)
                self.progress_updated.emit(progress)
                
                # 短暂延迟以显示进度
                self.msleep(10)
                
            self.text_processed.emit(processed_text)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
            
    def process_line(self, line: str) -> str:
        """处理单行文本"""
        options = self.process_options
        
        # 清理空白字符
        if options.get('clean_whitespace', True):
            line = re.sub(r'\s+', ' ', line.strip())
            
        # 移除特殊字符
        if options.get('remove_special_chars', False):
            line = re.sub(r'[^\w\s\-\.\,\!\?\;\:\(\)]', '', line)
            
        # 转换大小写
        if options.get('convert_case', 'none') == 'upper':
            line = line.upper()
        elif options.get('convert_case', 'none') == 'lower':
            line = line.lower()
            
        # 添加标点符号
        if options.get('add_punctuation', False) and line and not line.endswith('.'):
            line += '.'
            
        return line


class TextInputWidget(QWidget):
    """文本输入组件"""
    
    # 信号定义
    text_changed = pyqtSignal(str)  # 文本变化信号
    file_imported = pyqtSignal(str)  # 文件导入信号
    processing_finished = pyqtSignal(str)  # 处理完成信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_file = None
        self.processed_text = ""
        self.worker = None
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        
        # 标题区域
        title_layout = QHBoxLayout()
        title_label = QLabel("文本输入与预处理")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        # 字数统计
        self.char_count_label = QLabel("字符数: 0")
        self.char_count_label.setStyleSheet("color: #666;")
        title_layout.addWidget(self.char_count_label)
        
        layout.addLayout(title_layout)
        
        # 文本输入区域
        input_group = QGroupBox("文本输入")
        input_layout = QVBoxLayout(input_group)
        
        # 文本编辑框
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("请在此输入要处理的文本...\n\n支持格式：\n- 普通文本\n- 段落分隔\n- JSON格式的文档数据")
        self.text_edit.setFont(QFont("Consolas", 10))
        self.text_edit.textChanged.connect(self.on_text_changed)
        input_layout.addWidget(self.text_edit)
        
        # 输入控制按钮
        input_buttons = QHBoxLayout()
        
        self.import_file_btn = QPushButton("导入文件")
        self.import_file_btn.clicked.connect(self.import_file)
        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self.clear_text)
        self.sample_btn = QPushButton("加载示例")
        self.sample_btn.clicked.connect(self.load_sample)
        
        input_buttons.addWidget(self.import_file_btn)
        input_buttons.addWidget(self.clear_btn)
        input_buttons.addWidget(self.sample_btn)
        input_buttons.addStretch()
        
        input_layout.addLayout(input_buttons)
        layout.addWidget(input_group)
        
        # 预处理选项
        preprocess_group = QGroupBox("预处理选项")
        preprocess_layout = QVBoxLayout(preprocess_group)
        
        # 第一行选项
        options_row1 = QHBoxLayout()
        
        self.clean_whitespace_cb = QCheckBox("清理空白字符")
        self.clean_whitespace_cb.setChecked(True)
        self.remove_special_chars_cb = QCheckBox("移除特殊字符")
        self.add_punctuation_cb = QCheckBox("自动添加标点")
        
        options_row1.addWidget(self.clean_whitespace_cb)
        options_row1.addWidget(self.remove_special_chars_cb)
        options_row1.addWidget(self.add_punctuation_cb)
        options_row1.addStretch()
        
        preprocess_layout.addLayout(options_row1)
        
        # 第二行选项
        options_row2 = QHBoxLayout()
        
        self.case_conversion_cb = QComboBox()
        self.case_conversion_cb.addItems(["不转换", "转大写", "转小写"])
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(100, 10000)
        self.max_length_spin.setValue(2000)
        self.max_length_spin.setSuffix(" 字符")
        self.max_length_label = QLabel("最大长度:")
        
        options_row2.addWidget(QLabel("大小写转换:"))
        options_row2.addWidget(self.case_conversion_cb)
        options_row2.addWidget(self.max_length_label)
        options_row2.addWidget(self.max_length_spin)
        options_row2.addStretch()
        
        preprocess_layout.addLayout(options_row2)
        
        # 处理控制
        process_buttons = QHBoxLayout()
        
        self.preprocess_btn = QPushButton("预处理文本")
        self.preprocess_btn.clicked.connect(self.preprocess_text)
        self.reset_btn = QPushButton("重置选项")
        self.reset_btn.clicked.connect(self.reset_options)
        
        process_buttons.addWidget(self.preprocess_btn)
        process_buttons.addWidget(self.reset_btn)
        process_buttons.addStretch()
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        process_buttons.addWidget(self.progress_bar)
        
        preprocess_layout.addLayout(process_buttons)
        layout.addWidget(preprocess_group)
        
    def import_file(self):
        """导入文件"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择文本文件", "", 
            "文本文件 (*.txt *.md);;JSON文件 (*.json);;所有文件 (*)"
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 如果是JSON文件，尝试解析并提取文本
                if filename.endswith('.json'):
                    import json
                    try:
                        data = json.loads(content)
                        # 尝试从各种可能的字段提取文本
                        text_content = ""
                        if isinstance(data, dict):
                            text_fields = ['text', 'content', 'document', 'raw_text']
                            for field in text_fields:
                                if field in data:
                                    text_content = str(data[field])
                                    break
                            if not text_content:
                                # 如果没有找到文本字段，尝试提取所有字符串值
                                for key, value in data.items():
                                    if isinstance(value, str) and len(value) > len(text_content):
                                        text_content = value
                        elif isinstance(data, list):
                            # 如果是列表，连接所有字符串
                            text_content = ' '.join([str(item) for item in data if isinstance(item, str)])
                        else:
                            text_content = str(data)
                            
                        self.text_edit.setText(text_content)
                        
                    except json.JSONDecodeError:
                        # 如果JSON解析失败，直接使用文件内容
                        self.text_edit.setText(content)
                else:
                    self.text_edit.setText(content)
                    
                self.current_file = filename
                self.file_imported.emit(filename)
                self.update_char_count()
                
            except Exception as e:
                QMessageBox.warning(self, "导入失败", f"无法读取文件: {str(e)}")
                
    def clear_text(self):
        """清空文本"""
        self.text_edit.clear()
        self.current_file = None
        self.processed_text = ""
        self.update_char_count()
        
    def load_sample(self):
        """加载示例文本"""
        sample_text = """John Smith, the CEO of TechCorp, announced the company's new AI initiative during the quarterly meeting.
The initiative aims to revolutionize customer service through machine learning technologies.
Mary Johnson, the CTO, will lead the project with a team of 15 engineers.
The project timeline is set for 18 months with a budget of $5 million.
Investors are optimistic about the potential returns from this strategic move."""
        
        self.text_edit.setText(sample_text)
        self.update_char_count()
        
    def on_text_changed(self):
        """文本变化处理"""
        self.update_char_count()
        self.text_changed.emit(self.text_edit.toPlainText())
        
    def update_char_count(self):
        """更新字符数统计"""
        text = self.text_edit.toPlainText()
        char_count = len(text)
        self.char_count_label.setText(f"字符数: {char_count}")
        
    def preprocess_text(self):
        """预处理文本"""
        text = self.text_edit.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "警告", "请先输入要处理的文本")
            return
            
        # 检查文本长度
        max_length = self.max_length_spin.value()
        if len(text) > max_length:
            reply = QMessageBox.question(
                self, "确认", 
                f"文本长度 ({len(text)}) 超过限制 ({max_length})，是否截断？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                text = text[:max_length]
            else:
                return
                
        # 收集处理选项
        process_options = {
            'clean_whitespace': self.clean_whitespace_cb.isChecked(),
            'remove_special_chars': self.remove_special_chars_cb.isChecked(),
            'add_punctuation': self.add_punctuation_cb.isChecked(),
            'convert_case': ['none', 'upper', 'lower'][self.case_conversion_cb.currentIndex()]
        }
        
        # 启动处理工作线程
        self.worker = TextProcessorWorker(text, process_options)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.text_processed.connect(self.on_processing_finished)
        self.worker.error_occurred.connect(self.on_processing_error)
        
        self.progress_bar.setVisible(True)
        self.preprocess_btn.setEnabled(False)
        self.worker.start()
        
    def update_progress(self, value):
        """更新处理进度"""
        self.progress_bar.setValue(value)
        
    def on_processing_finished(self, processed_text):
        """处理完成"""
        self.processed_text = processed_text
        self.text_edit.setText(processed_text)
        self.progress_bar.setVisible(False)
        self.preprocess_btn.setEnabled(True)
        self.processing_finished.emit(processed_text)
        
    def on_processing_error(self, error_message):
        """处理出错"""
        QMessageBox.critical(self, "处理错误", f"文本处理失败: {error_message}")
        self.progress_bar.setVisible(False)
        self.preprocess_btn.setEnabled(True)
        
    def reset_options(self):
        """重置预处理选项"""
        self.clean_whitespace_cb.setChecked(True)
        self.remove_special_chars_cb.setChecked(False)
        self.add_punctuation_cb.setChecked(False)
        self.case_conversion_cb.setCurrentIndex(0)
        self.max_length_spin.setValue(2000)
        
    def get_text(self) -> str:
        """获取当前文本"""
        return self.text_edit.toPlainText()
        
    def get_processed_text(self) -> str:
        """获取预处理后的文本"""
        return self.processed_text
        
    def get_current_file(self) -> Optional[str]:
        """获取当前文件路径"""
        return self.current_file
        
    def set_text(self, text: str):
        """设置文本内容"""
        self.text_edit.setText(text)
        self.update_char_count()
        
    def is_empty(self) -> bool:
        """检查是否为空"""
        return not self.text_edit.toPlainText().strip()