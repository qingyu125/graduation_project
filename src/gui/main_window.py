# -*- coding: utf-8 -*-
"""
DocRED 主窗口
完整的PyQt6用户界面，集成所有功能模块
"""

from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QSplitter, 
                             QWidget, QMenuBar, QStatusBar, QToolBar,
                             QMessageBox, QFileDialog, QApplication, QTextEdit, QLabel)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QIcon, QPixmap, QKeySequence

from .widgets import ProgressPanel, ResultTable, ModelSelector, FlowVisualizer
from .components import TextInputWidget, ControlPanel, AboutDialog

import sys
import json
import os
from datetime import datetime
from typing import Dict, List, Optional


class DocRedMainWindow(QMainWindow):
    """DocRED关系抽取系统主窗口"""
    
    # 信号定义
    text_processed = pyqtSignal(str, dict)  # 文本处理完成
    model_loaded = pyqtSignal(str)  # 模型加载完成
    error_occurred = pyqtSignal(str)  # 错误发生
    
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.is_processing = False
        self.setup_ui()
        self.setup_menu()
        self.setup_toolbar()
        self.setup_statusbar()
        self.connect_signals()
        
        # 应用样式
        self.apply_styles()
        
    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("DocRED 关系抽取系统 v1.0.0")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # 主分割器
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # 左侧面板（输入和配置）
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # 中间面板（流程可视化）
        center_panel = self.create_center_panel()
        main_splitter.addWidget(center_panel)
        
        # 右侧面板（结果和进度）
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # 设置分割器比例
        main_splitter.setSizes([400, 500, 600])
        
    def create_left_panel(self) -> QWidget:
        """创建左侧面板"""
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 文本输入组件
        self.text_input = TextInputWidget()
        layout.addWidget(self.text_input, 2)
        
        # 控制面板
        self.control_panel = ControlPanel()
        layout.addWidget(self.control_panel, 1)
        
        return left_widget
        
    def create_center_panel(self) -> QWidget:
        """创建中间面板（流程可视化）"""
        center_widget = QWidget()
        layout = QVBoxLayout(center_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 流程可视化组件
        self.flow_visualizer = FlowVisualizer()
        layout.addWidget(self.flow_visualizer, 3)
        
        return center_widget
        
    def create_right_panel(self) -> QWidget:
        """创建右侧面板"""
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 模型选择器
        self.model_selector = ModelSelector()
        layout.addWidget(self.model_selector, 1)
        
        # 结果显示区域
        results_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 进度面板
        self.progress_panel = ProgressPanel()
        results_splitter.addWidget(self.progress_panel)
        
        # 结果表格
        self.result_table = ResultTable()
        results_splitter.addWidget(self.result_table)
        
        # 设置分割器比例
        results_splitter.setSizes([250, 450])
        
        layout.addWidget(results_splitter, 3)
        
        return right_widget
        
    def setup_menu(self):
        """设置菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件(&F)')
        
        # 新建动作
        new_action = QAction('新建(&N)', self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.new_document)
        file_menu.addAction(new_action)
        
        # 打开文件
        open_action = QAction('打开(&O)', self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        # 保存结果
        save_action = QAction('保存结果(&S)', self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # 退出动作
        exit_action = QAction('退出(&X)', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 编辑菜单
        edit_menu = menubar.addMenu('编辑(&E)')
        
        # 复制结果
        copy_action = QAction('复制结果(&C)', self)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(self.copy_results)
        edit_menu.addAction(copy_action)
        
        # 清空结果
        clear_action = QAction('清空结果(&L)', self)
        clear_action.triggered.connect(self.clear_results)
        edit_menu.addAction(clear_action)
        
        edit_menu.addSeparator()
        
        # 设置
        settings_action = QAction('设置(&T)', self)
        settings_action.triggered.connect(self.show_settings)
        edit_menu.addAction(settings_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu('工具(&T)')
        
        # 导出配置
        export_config_action = QAction('导出配置(&E)', self)
        export_config_action.triggered.connect(self.export_config)
        tools_menu.addAction(export_config_action)
        
        # 导入配置
        import_config_action = QAction('导入配置(&I)', self)
        import_config_action.triggered.connect(self.import_config)
        tools_menu.addAction(import_config_action)
        
        tools_menu.addSeparator()
        
        # 清理缓存
        clean_cache_action = QAction('清理缓存(&C)', self)
        clean_cache_action.triggered.connect(self.clean_cache)
        tools_menu.addAction(clean_cache_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助(&H)')
        
        # 用户手册
        manual_action = QAction('用户手册(&M)', self)
        manual_action.triggered.connect(self.show_manual)
        help_menu.addAction(manual_action)
        
        # API文档
        api_action = QAction('API文档(&A)', self)
        api_action.triggered.connect(self.show_api_docs)
        help_menu.addAction(api_action)
        
        help_menu.addSeparator()
        
        # 关于
        about_action = QAction('关于(&A)', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_toolbar(self):
        """设置工具栏"""
        toolbar = self.addToolBar('主工具栏')
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        
        # 新建
        new_action = QAction('新建', self)
        new_action.triggered.connect(self.new_document)
        toolbar.addAction(new_action)
        
        # 打开
        open_action = QAction('打开', self)
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        # 开始处理
        start_action = QAction('开始处理', self)
        start_action.triggered.connect(self.start_processing)
        toolbar.addAction(start_action)
        
        # 暂停/恢复
        self.pause_action = QAction('暂停', self)
        self.pause_action.setEnabled(False)
        self.pause_action.triggered.connect(self.toggle_pause)
        toolbar.addAction(self.pause_action)
        
        # 停止
        stop_action = QAction('停止', self)
        stop_action.setEnabled(False)
        stop_action.triggered.connect(self.stop_processing)
        toolbar.addAction(stop_action)
        
        toolbar.addSeparator()
        
        # 保存结果
        save_action = QAction('保存', self)
        save_action.triggered.connect(self.save_results)
        toolbar.addAction(save_action)
        
        # 关于
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        toolbar.addAction(about_action)
        
    def setup_statusbar(self):
        """设置状态栏"""
        statusbar = self.statusBar()
        
        # 状态信息
        self.status_label = QLabel("就绪")
        statusbar.addWidget(self.status_label)
        
        # 分隔符
        statusbar.addPermanentWidget(QLabel("|"))
        
        # 模型信息
        self.model_label = QLabel("模型: 未加载")
        statusbar.addPermanentWidget(self.model_label)
        
        # 分隔符
        statusbar.addPermanentWidget(QLabel("|"))
        
        # 结果统计
        self.results_label = QLabel("结果: 0")
        statusbar.addPermanentWidget(self.results_label)
        
    def connect_signals(self):
        """连接信号"""
        # 模型选择器信号
        self.model_selector.model_changed.connect(self.on_model_changed)
        self.model_selector.config_changed.connect(self.on_config_changed)
        
        # 控制面板信号
        self.control_panel.processing_started.connect(self.on_processing_started)
        self.control_panel.processing_paused.connect(self.on_processing_paused)
        self.control_panel.processing_resumed.connect(self.on_processing_resumed)
        self.control_panel.processing_stopped.connect(self.on_processing_stopped)
        
        # 文本输入信号
        self.text_input.text_changed.connect(self.on_text_changed)
        self.text_input.file_imported.connect(self.on_file_imported)
        
        # 结果表格信号
        self.result_table.export_requested.connect(self.on_export_requested)
        
        # 流程可视化信号
        self.flow_visualizer.step_completed.connect(self.on_step_completed)
        self.flow_visualizer.flow_completed.connect(self.on_flow_completed)
        self.flow_visualizer.error_occurred.connect(self.on_flow_error)
        
    def apply_styles(self):
        """应用样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #e0e0e0;
                border: 1px solid #999;
                border-radius: 3px;
                padding: 5px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
            QPushButton:disabled {
                background-color: #f0f0f0;
                color: #999;
            }
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 3px;
                background-color: white;
            }
            QTableWidget {
                gridline-color: #ddd;
                selection-background-color: #cce8ff;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                padding: 4px;
            }
        """)
        
    def new_document(self):
        """新建文档"""
        if self.is_processing:
            QMessageBox.warning(self, "警告", "正在处理中，无法创建新文档")
            return
            
        if not self.result_table.get_results_count() == 0:
            reply = QMessageBox.question(
                self, "确认", 
                "当前结果将被清空，是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
                
        self.text_input.clear_text()
        self.result_table.clear_results()
        self.progress_panel.clear_log()
        self.status_label.setText("新文档已创建")
        self.current_file = None
        
    def open_file(self):
        """打开文件"""
        if self.is_processing:
            QMessageBox.warning(self, "警告", "正在处理中，无法打开文件")
            return
            
        filename, _ = QFileDialog.getOpenFileName(
            self, "打开文档", "", 
            "文本文件 (*.txt *.md);;JSON文件 (*.json);;所有文件 (*)"
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if filename.endswith('.json'):
                    # 尝试解析JSON
                    try:
                        import json
                        data = json.loads(content)
                        # 提取文本内容
                        if isinstance(data, dict) and 'text' in data:
                            content = data['text']
                        elif isinstance(data, list):
                            content = ' '.join([str(item) for item in data if isinstance(item, str)])
                    except:
                        pass
                        
                self.text_input.set_text(content)
                self.current_file = filename
                self.status_label.setText(f"已打开: {os.path.basename(filename)}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"打开文件失败: {str(e)}")
                
    def save_results(self):
        """保存结果"""
        if self.result_table.get_results_count() == 0:
            QMessageBox.information(self, "提示", "没有结果可保存")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存结果", 
            f"docred_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON文件 (*.json);;CSV文件 (*.csv);;文本文件 (*.txt)"
        )
        
        if filename:
            try:
                if filename.endswith('.json'):
                    export_data = {
                        'export_time': datetime.now().isoformat(),
                        'total_results': self.result_table.get_results_count(),
                        'results': self.result_table.get_all_results()
                    }
                    with open(filename, 'w', encoding='utf-8') as f:
                        import json
                        json.dump(export_data, f, ensure_ascii=False, indent=2)
                elif filename.endswith('.csv'):
                    # 转换为CSV
                    import pandas as pd
                    results = self.result_table.get_all_results()
                    df_data = []
                    for result in results:
                        df_data.append({
                            '文本片段': result.get('text', ''),
                            '头实体': result.get('head_entity', ''),
                            '关系': result.get('relation', ''),
                            '尾实体': result.get('tail_entity', ''),
                            '置信度': result.get('confidence', 0.0)
                        })
                    df = pd.DataFrame(df_data)
                    df.to_csv(filename, index=False, encoding='utf-8-sig')
                elif filename.endswith('.txt'):
                    # 转换为TXT
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write("DocRED 关系抽取结果\n")
                        f.write("=" * 50 + "\n\n")
                        for i, result in enumerate(self.result_table.get_all_results(), 1):
                            f.write(f"结果 {i}:\n")
                            f.write(f"文本片段: {result.get('text', '')}\n")
                            f.write(f"头实体: {result.get('head_entity', '')}\n")
                            f.write(f"关系: {result.get('relation', '')}\n")
                            f.write(f"尾实体: {result.get('tail_entity', '')}\n")
                            f.write(f"置信度: {result.get('confidence', 0.0):.3f}\n")
                            f.write("-" * 30 + "\n\n")
                            
                self.status_label.setText(f"结果已保存: {os.path.basename(filename)}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
                
    def copy_results(self):
        """复制结果到剪贴板"""
        results = self.result_table.get_all_results()
        if not results:
            QMessageBox.information(self, "提示", "没有结果可复制")
            return
            
        # 准备复制内容
        content = "DocRED 关系抽取结果\n"
        content += "=" * 30 + "\n"
        for i, result in enumerate(results, 1):
            content += f"{i}. {result.get('head_entity', '')}"
            content += f" --{result.get('relation', '')}--> "
            content += f"{result.get('tail_entity', '')}"
            content += f" (置信度: {result.get('confidence', 0.0):.3f})\n"
            
        # 复制到剪贴板
        clipboard = QApplication.clipboard()
        clipboard.setText(content)
        self.status_label.setText("结果已复制到剪贴板")
        
    def clear_results(self):
        """清空结果"""
        self.result_table.clear_results()
        self.results_label.setText("结果: 0")
        self.progress_panel.clear_log()
        self.status_label.setText("结果已清空")
        
    def export_config(self):
        """导出配置"""
        config = self.model_selector.get_current_config()
        filename, _ = QFileDialog.getSaveFileName(
            self, "导出配置", "docred_config.json", "JSON文件 (*.json)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                self.status_label.setText(f"配置已导出: {os.path.basename(filename)}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出配置失败: {str(e)}")
                
    def import_config(self):
        """导入配置"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "导入配置", "", "JSON文件 (*.json)"
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # 这里可以添加配置验证逻辑
                self.status_label.setText(f"配置已导入: {os.path.basename(filename)}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导入配置失败: {str(e)}")
                
    def clean_cache(self):
        """清理缓存"""
        reply = QMessageBox.question(
            self, "确认", 
            "这将清理所有临时文件和缓存，确定继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # 清理临时文件
            import tempfile
            import shutil
            try:
                # 这里可以添加具体的清理逻辑
                self.status_label.setText("缓存清理完成")
                QMessageBox.information(self, "完成", "缓存已清理")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"清理过程中发生错误: {str(e)}")
                
    def show_settings(self):
        """显示设置对话框"""
        # 这里可以创建设置对话框
        QMessageBox.information(self, "设置", "设置功能开发中...")
        
    def show_manual(self):
        """显示用户手册"""
        QMessageBox.information(
            self, "用户手册", 
            "用户手册可以在以下位置找到：\n"
            "- 菜单栏: 帮助 > 用户手册\n"
            "- 工具栏: 帮助按钮\n"
            "- 在线文档: https://docred.github.io/manual"
        )
        
    def show_api_docs(self):
        """显示API文档"""
        QMessageBox.information(
            self, "API文档", 
            "API文档可以在以下位置找到：\n"
            "- 源码中的docstring注释\n"
            "- 在线文档: https://docred.github.io/api\n"
            "- GitHub: https://github.com/docred/extract-system"
        )
        
    def show_about(self):
        """显示关于对话框"""
        about_dialog = AboutDialog(self)
        about_dialog.exec()
        
    def start_processing(self):
        """开始处理"""
        if not self.text_input.is_empty():
            self.control_panel.start_processing()
        else:
            QMessageBox.warning(self, "警告", "请先输入要处理的文本")
            
    def toggle_pause(self):
        """切换暂停/恢复"""
        if self.is_processing:
            self.control_panel.pause_processing()
            
    def stop_processing(self):
        """停止处理"""
        if self.is_processing:
            self.control_panel.stop_processing()
            
    # 信号处理方法
    def on_model_changed(self, model_id: str):
        """模型变化处理"""
        self.model_label.setText(f"模型: {model_id}")
        self.status_label.setText(f"已切换到模型: {model_id}")
        
    def on_config_changed(self, config: dict):
        """配置变化处理"""
        self.model_loaded.emit("config_changed")
        
    def on_processing_started(self, text: str, model_config: dict, options: dict):
        """处理开始"""
        self.is_processing = True
        self.progress_panel.start_progress()
        self.pause_action.setEnabled(True)
        
        # 初始化流程可视化
        self.flow_visualizer.initialize_flow(text, model_config)
        
        # 停止按钮在ControlPanel中处理
        
    def on_processing_paused(self):
        """处理暂停"""
        self.pause_action.setText("恢复")
        
    def on_processing_resumed(self):
        """处理恢复"""
        self.pause_action.setText("暂停")
        
    def on_processing_stopped(self):
        """处理停止"""
        self.is_processing = False
        self.progress_panel.stop_progress()
        self.pause_action.setEnabled(False)
        self.pause_action.setText("暂停")
        
    def on_text_changed(self, text: str):
        """文本变化"""
        char_count = len(text)
        self.status_label.setText(f"文本长度: {char_count} 字符")
        
    def on_file_imported(self, filename: str):
        """文件导入"""
        self.current_file = filename
        self.status_label.setText(f"已导入文件: {os.path.basename(filename)}")
        
    def on_export_requested(self, filename: str):
        """导出请求"""
        self.status_label.setText(f"结果已导出: {os.path.basename(filename)}")
        
    # 便捷方法
    def get_model_config(self) -> dict:
        """获取当前模型配置"""
        return self.model_selector.get_current_config()
        
    def get_text(self) -> str:
        """获取当前文本"""
        return self.text_input.get_text()
        
    def add_result(self, result: dict):
        """添加结果"""
        self.result_table.add_result(result)
        self.results_label.setText(f"结果: {self.result_table.get_results_count()}")
        
    def log_info(self, message: str):
        """添加信息日志"""
        self.progress_panel.log_info(message)
        
    def log_error(self, message: str):
        """添加错误日志"""
        self.progress_panel.log_error(message)
        
    # 流程可视化信号处理
    def on_step_completed(self, step_name: str, result):
        """步骤完成"""
        self.log_info(f"步骤完成: {step_name}")
        self.progress_panel.log_info(f"步骤 '{step_name}' 已完成")
        
    def on_flow_completed(self, flow_results: dict):
        """流程完成"""
        self.log_info("整个处理流程已完成")
        self.progress_panel.log_info("所有处理步骤已完成")
        
    def on_flow_error(self, step_name: str, error_message: str):
        """流程错误"""
        error_msg = f"流程步骤 '{step_name}' 发生错误: {error_message}"
        self.log_error(error_msg)
        self.progress_panel.log_error(error_msg)
        QMessageBox.critical(self, "处理错误", error_msg)
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.is_processing:
            reply = QMessageBox.question(
                self, "确认退出", 
                "正在处理中，确定要退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                if self.control_panel.is_currently_processing():
                    self.control_panel.stop_processing()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()