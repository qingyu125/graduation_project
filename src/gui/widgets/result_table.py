# -*- coding: utf-8 -*-
"""
结果表格控件
以表格形式展示关系抽取结果，支持数据导出和详细查看
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
                             QTableWidgetItem, QPushButton, QGroupBox, QLabel,
                             QHeaderView, QTextEdit, QSplitter, QTabWidget)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette
import pandas as pd
import json
from datetime import datetime


class ResultTable(QWidget):
    """结果表格，显示关系抽取结果"""
    
    # 信号定义
    selection_changed = pyqtSignal(int, int)  # 选择变化信号
    export_requested = pyqtSignal(str)  # 导出请求信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_data = []
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        
        # 标题区域
        title_layout = QHBoxLayout()
        title_label = QLabel("关系抽取结果")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        # 统计信息
        self.stat_label = QLabel("共 0 条结果")
        self.stat_label.setFont(QFont("Arial", 10))
        self.stat_label.setStyleSheet("color: #666;")
        title_layout.addWidget(self.stat_label)
        
        layout.addLayout(title_layout)
        
        # 主内容区域
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(main_splitter)
        
        # 结果表格
        table_group = QGroupBox("抽取结果")
        table_layout = QVBoxLayout(table_group)
        
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(6)
        self.result_table.setHorizontalHeaderLabels([
            "文本片段", "头实体", "关系", "尾实体", "置信度", "处理时间"
        ])
        
        # 设置表格属性
        self.result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.result_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        
        self.result_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.itemSelectionChanged.connect(self.on_selection_changed)
        
        table_layout.addWidget(self.result_table)
        
        # 表格控制按钮
        table_buttons = QHBoxLayout()
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh_table)
        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self.clear_results)
        
        table_buttons.addWidget(self.refresh_btn)
        table_buttons.addWidget(self.clear_btn)
        table_buttons.addStretch()
        
        # 导出按钮
        self.export_csv_btn = QPushButton("导出CSV")
        self.export_csv_btn.clicked.connect(lambda: self.export_results("csv"))
        self.export_txt_btn = QPushButton("导出TXT")
        self.export_txt_btn.clicked.connect(lambda: self.export_results("txt"))
        self.export_json_btn = QPushButton("导出JSON")
        self.export_json_btn.clicked.connect(lambda: self.export_results("json"))
        
        table_buttons.addWidget(self.export_csv_btn)
        table_buttons.addWidget(self.export_txt_btn)
        table_buttons.addWidget(self.export_json_btn)
        
        table_layout.addLayout(table_buttons)
        main_splitter.addWidget(table_group)
        
        # 详细信息面板
        detail_group = QGroupBox("详细信息")
        detail_layout = QVBoxLayout(detail_group)
        
        self.detail_tabs = QTabWidget()
        
        # 文本详情标签页
        text_detail_widget = QWidget()
        text_detail_layout = QVBoxLayout(text_detail_widget)
        self.text_detail = QTextEdit()
        self.text_detail.setReadOnly(True)
        self.text_detail.setMaximumHeight(100)
        text_detail_layout.addWidget(self.text_detail)
        self.detail_tabs.addTab(text_detail_widget, "原始文本")
        
        # 处理详情标签页
        process_detail_widget = QWidget()
        process_detail_layout = QVBoxLayout(process_detail_widget)
        self.process_detail = QTextEdit()
        self.process_detail.setReadOnly(True)
        self.process_detail.setMaximumHeight(100)
        process_detail_layout.addWidget(self.process_detail)
        self.detail_tabs.addTab(process_detail_widget, "处理详情")
        
        # 伪代码标签页
        pseudocode_detail_widget = QWidget()
        pseudocode_detail_layout = QVBoxLayout(pseudocode_detail_widget)
        self.pseudocode_detail = QTextEdit()
        self.pseudocode_detail.setReadOnly(True)
        self.pseudocode_detail.setMaximumHeight(100)
        pseudocode_detail_layout.addWidget(self.pseudocode_detail)
        self.detail_tabs.addTab(pseudocode_detail_widget, "伪代码")
        
        detail_layout.addWidget(self.detail_tabs)
        main_splitter.addWidget(detail_group)
        
        # 设置分割器比例
        main_splitter.setSizes([400, 200])
        
    def add_result(self, result_data):
        """添加一条结果"""
        self.results_data.append(result_data)
        self.refresh_table()
        
    def set_results(self, results_list):
        """设置多个结果"""
        self.results_data = results_list
        self.refresh_table()
        
    def refresh_table(self):
        """刷新表格显示"""
        self.result_table.setRowCount(len(self.results_data))
        
        for row, result in enumerate(self.results_data):
            # 文本片段
            text_item = QTableWidgetItem(result.get('text', ''))
            text_item.setToolTip(result.get('text', ''))
            self.result_table.setItem(row, 0, text_item)
            
            # 头实体
            head_item = QTableWidgetItem(result.get('head_entity', ''))
            self.result_table.setItem(row, 1, head_item)
            
            # 关系
            relation_item = QTableWidgetItem(result.get('relation', ''))
            self.result_table.setItem(row, 2, relation_item)
            
            # 尾实体
            tail_item = QTableWidgetItem(result.get('tail_entity', ''))
            self.result_table.setItem(row, 3, tail_item)
            
            # 置信度
            confidence = result.get('confidence', 0.0)
            confidence_item = QTableWidgetItem(f"{confidence:.3f}")
            if confidence > 0.8:
                confidence_item.setBackground(QColor(200, 255, 200))
            elif confidence > 0.6:
                confidence_item.setBackground(QColor(255, 255, 200))
            else:
                confidence_item.setBackground(QColor(255, 200, 200))
            self.result_table.setItem(row, 4, confidence_item)
            
            # 处理时间
            timestamp = result.get('timestamp', datetime.now().strftime("%H:%M:%S"))
            time_item = QTableWidgetItem(timestamp)
            self.result_table.setItem(row, 5, time_item)
            
        # 更新统计信息
        self.update_statistics()
        
    def update_statistics(self):
        """更新统计信息"""
        count = len(self.results_data)
        self.stat_label.setText(f"共 {count} 条结果")
        
    def on_selection_changed(self):
        """表格选择变化处理"""
        current_row = self.result_table.currentRow()
        if current_row >= 0 and current_row < len(self.results_data):
            result = self.results_data[current_row]
            self.show_detail(result)
            self.selection_changed.emit(current_row, -1)
            
    def show_detail(self, result):
        """显示详细信息"""
        # 原始文本
        self.text_detail.setText(result.get('original_text', result.get('text', '')))
        
        # 处理详情
        process_info = f"头实体: {result.get('head_entity', '')}\n"
        process_info += f"关系: {result.get('relation', '')}\n"
        process_info += f"尾实体: {result.get('tail_entity', '')}\n"
        process_info += f"置信度: {result.get('confidence', 0.0):.3f}\n"
        process_info += f"处理模型: {result.get('model_used', '')}\n"
        process_info += f"处理时间: {result.get('timestamp', '')}"
        self.process_detail.setText(process_info)
        
        # 伪代码
        pseudocode = result.get('pseudocode', '无伪代码信息')
        self.pseudocode_detail.setText(pseudocode)
        
    def clear_results(self):
        """清空所有结果"""
        self.results_data.clear()
        self.result_table.setRowCount(0)
        self.text_detail.clear()
        self.process_detail.clear()
        self.pseudocode_detail.clear()
        self.update_statistics()
        
    def export_results(self, format_type):
        """导出结果"""
        if not self.results_data:
            return
            
        from PyQt6.QtWidgets import QFileDialog
        from PyQt6.QtCore import QStandardPaths
        import os
        
        # 设置文件扩展名和过滤器
        filters = {
            'csv': ("CSV文件 (*.csv)", ".csv"),
            'txt': ("文本文件 (*.txt)", ".txt"),
            'json': ("JSON文件 (*.json)", ".json")
        }
        
        filter_str, extension = filters[format_type]
        default_path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        default_name = f"docred_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}{extension}"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, f"导出{format_type.upper()}", 
            os.path.join(default_path, default_name), filter_str
        )
        
        if filename:
            try:
                if format_type == 'csv':
                    self.export_to_csv(filename)
                elif format_type == 'txt':
                    self.export_to_txt(filename)
                elif format_type == 'json':
                    self.export_to_json(filename)
                self.export_requested.emit(filename)
            except Exception as e:
                print(f"导出失败: {str(e)}")
                
    def export_to_csv(self, filename):
        """导出CSV格式"""
        # 准备数据
        export_data = []
        for result in self.results_data:
            export_data.append({
                '文本片段': result.get('text', ''),
                '头实体': result.get('head_entity', ''),
                '关系': result.get('relation', ''),
                '尾实体': result.get('tail_entity', ''),
                '置信度': result.get('confidence', 0.0),
                '处理时间': result.get('timestamp', ''),
                '模型': result.get('model_used', ''),
                '原始文本': result.get('original_text', ''),
                '伪代码': result.get('pseudocode', '')
            })
            
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        
    def export_to_txt(self, filename):
        """导出TXT格式"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("DocRED 关系抽取结果\n")
            f.write("=" * 50 + "\n\n")
            
            for i, result in enumerate(self.results_data, 1):
                f.write(f"结果 {i}:\n")
                f.write(f"文本片段: {result.get('text', '')}\n")
                f.write(f"头实体: {result.get('head_entity', '')}\n")
                f.write(f"关系: {result.get('relation', '')}\n")
                f.write(f"尾实体: {result.get('tail_entity', '')}\n")
                f.write(f"置信度: {result.get('confidence', 0.0):.3f}\n")
                f.write(f"处理模型: {result.get('model_used', '')}\n")
                f.write(f"处理时间: {result.get('timestamp', '')}\n")
                f.write(f"原始文本: {result.get('original_text', '')}\n")
                f.write(f"伪代码:\n{result.get('pseudocode', '')}\n")
                f.write("-" * 30 + "\n\n")
                
    def export_to_json(self, filename):
        """导出JSON格式"""
        export_data = {
            'export_time': datetime.now().isoformat(),
            'total_results': len(self.results_data),
            'results': self.results_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
            
    def get_results_count(self):
        """获取结果数量"""
        return len(self.results_data)
        
    def get_selected_result(self):
        """获取当前选中的结果"""
        current_row = self.result_table.currentRow()
        if current_row >= 0 and current_row < len(self.results_data):
            return self.results_data[current_row]
        return None
        
    def get_all_results(self):
        """获取所有结果"""
        return self.results_data.copy()