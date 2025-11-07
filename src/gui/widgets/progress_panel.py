# -*- coding: utf-8 -*-
"""
进度面板控件
显示处理进度、状态信息和实时日志
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QProgressBar, 
                             QLabel, QTextEdit, QPushButton, QGroupBox)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QTextCursor
import time
from datetime import datetime


class ProgressPanel(QWidget):
    """进度面板，显示处理进度和状态信息"""
    
    # 信号定义
    progress_updated = pyqtSignal(int, str)  # 进度更新信号 (progress, message)
    status_changed = pyqtSignal(str)  # 状态变化信号
    log_updated = pyqtSignal(str)  # 日志更新信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_progress = 0
        self.start_time = None
        self.setup_ui()
        self.setup_timer()
        
    def setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("处理进度")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # 进度条区域
        progress_group = QGroupBox("当前进度")
        progress_layout = QVBoxLayout(progress_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setMinimumHeight(30)
        progress_layout.addWidget(self.progress_bar)
        
        # 状态信息
        status_layout = QHBoxLayout()
        self.status_label = QLabel("就绪")
        self.status_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.time_label = QLabel("00:00")
        self.time_label.setFont(QFont("Arial", 9))
        status_layout.addWidget(QLabel("状态:"))
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(QLabel("用时:"))
        status_layout.addWidget(self.time_label)
        progress_layout.addLayout(status_layout)
        
        layout.addWidget(progress_group)
        
        # 日志显示区域
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        # 日志控制按钮
        log_buttons = QHBoxLayout()
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.clicked.connect(self.clear_log)
        self.save_log_btn = QPushButton("保存日志")
        self.save_log_btn.clicked.connect(self.save_log)
        log_buttons.addWidget(self.clear_log_btn)
        log_buttons.addWidget(self.save_log_btn)
        log_buttons.addStretch()
        log_layout.addLayout(log_buttons)
        
        layout.addWidget(log_group)
        
    def setup_timer(self):
        """设置计时器"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_elapsed_time)
        
    def start_progress(self):
        """开始进度跟踪"""
        self.current_progress = 0
        self.start_time = time.time()
        self.progress_bar.setValue(0)
        self.status_label.setText("处理中...")
        self.update_progress(0, "开始处理...")
        self.timer.start(1000)  # 每秒更新一次
        self.log_info("开始文档处理")
        
    def stop_progress(self):
        """停止进度跟踪"""
        self.current_progress = 0
        self.start_time = None
        self.progress_bar.setValue(100)
        self.status_label.setText("完成")
        self.timer.stop()
        self.log_info("处理完成")
        
    def update_progress(self, progress, message=""):
        """更新进度"""
        self.current_progress = min(100, max(0, progress))
        self.progress_bar.setValue(self.current_progress)
        
        if message:
            self.status_label.setText(message)
            self.progress_updated.emit(self.current_progress, message)
            
    def update_status(self, status):
        """更新状态"""
        self.status_label.setText(status)
        self.status_changed.emit(status)
        
    def log_info(self, message):
        """添加信息日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [INFO] {message}"
        self.log_text.append(log_entry)
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)
        self.log_updated.emit(log_entry)
        
    def log_error(self, message):
        """添加错误日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [ERROR] {message}"
        self.log_text.append(log_entry)
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)
        self.log_updated.emit(log_entry)
        
    def log_warning(self, message):
        """添加警告日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [WARNING] {message}"
        self.log_text.append(log_entry)
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)
        self.log_updated.emit(log_entry)
        
    def update_elapsed_time(self):
        """更新已用时间"""
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            self.time_label.setText(f"{minutes:02d}:{seconds:02d}")
            
    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
        
    def save_log(self):
        """保存日志到文件"""
        from PyQt6.QtWidgets import QFileDialog
        from PyQt6.QtCore import QStandardPaths
        import os
        
        default_path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存日志", default_path, "文本文件 (*.txt)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                self.log_info(f"日志已保存到: {filename}")
            except Exception as e:
                self.log_error(f"保存日志失败: {str(e)}")
                
    def get_progress(self):
        """获取当前进度"""
        return self.current_progress
        
    def get_status(self):
        """获取当前状态"""
        return self.status_label.text()
        
    def get_logs(self):
        """获取所有日志"""
        return self.log_text.toPlainText()