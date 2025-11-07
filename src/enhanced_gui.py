# -*- coding: utf-8 -*-
"""
DocRED 增强GUI启动文件
展示完整的处理流程和三个模型选择
"""

import sys
import os
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# 添加src路径到sys.path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gui.main_window import DocRedMainWindow


def main():
    """主函数"""
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("DocRED 关系抽取系统")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("DocRED Project")
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 设置默认字体
    font = QFont("Arial", 10)
    app.setFont(font)
    
    # 创建主窗口
    try:
        main_window = DocRedMainWindow()
        main_window.show()
        
        # 显示欢迎消息
        QMessageBox.information(
            main_window, 
            "欢迎使用 DocRED 关系抽取系统",
            "这是一个完整的文档级关系抽取系统，支持三种模型：\n\n"
            "1. CodeLlama-7B + LoRA (推荐)\n"
            "2. BERT + 微调\n"
            "3. 传统规则方法\n\n"
            "请在左侧输入要处理的文本，然后选择模型并开始处理。"
        )
        
        # 运行应用
        return app.exec()
        
    except Exception as e:
        QMessageBox.critical(
            None, 
            "启动错误", 
            f"无法启动应用: {str(e)}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
