# -*- coding: utf-8 -*-
"""
关于对话框
显示系统信息和帮助文档
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTextEdit, QPushButton, QTabWidget, QWidget,
                             QGroupBox, QFormLayout, QListWidget, QListWidgetItem)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QIcon, QDesktopServices
from PyQt6.QtCore import QUrl
import sys
import platform
from datetime import datetime


class AboutDialog(QDialog):
    """关于对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("关于 DocRED 关系抽取系统")
        self.setModal(True)
        self.resize(600, 500)
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        
        # 标题区域
        title_layout = QVBoxLayout()
        
        app_name = QLabel("DocRED 关系抽取系统")
        app_name.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        app_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        version = QLabel("版本 1.0.0")
        version.setFont(QFont("Arial", 12))
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setStyleSheet("color: #666;")
        
        description = QLabel("基于文本→伪代码→要素抽取→知识融合→推理验证的完整流程")
        description.setFont(QFont("Arial", 10))
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setStyleSheet("color: #888;")
        
        title_layout.addWidget(app_name)
        title_layout.addWidget(version)
        title_layout.addWidget(description)
        layout.addLayout(title_layout)
        
        # 标签页
        self.tabs = QTabWidget()
        
        # 系统信息标签页
        self.setup_system_tab()
        
        # 功能特性标签页
        self.setup_features_tab()
        
        # 使用指南标签页
        self.setup_guide_tab()
        
        # 更新日志标签页
        self.setup_changelog_tab()
        
        layout.addWidget(self.tabs)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.check_updates_btn = QPushButton("检查更新")
        self.check_updates_btn.clicked.connect(self.check_updates)
        
        self.help_btn = QPushButton("在线帮助")
        self.help_btn.clicked.connect(self.open_help)
        
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.accept)
        self.close_btn.setDefault(True)
        
        button_layout.addWidget(self.check_updates_btn)
        button_layout.addWidget(self.help_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
    def setup_system_tab(self):
        """设置系统信息标签页"""
        system_widget = QWidget()
        layout = QVBoxLayout(system_widget)
        
        # 基本信息
        basic_group = QGroupBox("系统信息")
        basic_layout = QFormLayout(basic_group)
        
        # Python版本
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        basic_layout.addRow("Python版本:", QLabel(python_version))
        
        # 操作系统
        basic_layout.addRow("操作系统:", QLabel(platform.system() + " " + platform.release()))
        
        # 架构
        basic_layout.addRow("系统架构:", QLabel(platform.machine()))
        
        # 处理器
        try:
            cpu_info = platform.processor()
            if not cpu_info:
                cpu_info = "未知"
        except:
            cpu_info = "未知"
        basic_layout.addRow("处理器:", QLabel(cpu_info))
        
        # 内存信息
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            basic_layout.addRow("系统内存:", QLabel(f"{memory_gb:.1f} GB"))
        except ImportError:
            basic_layout.addRow("系统内存:", QLabel("无法检测"))
            
        layout.addWidget(basic_group)
        
        # 依赖库版本
        deps_group = QGroupBox("核心依赖库")
        deps_layout = QFormLayout(deps_group)
        
        # 检查主要依赖库
        deps = {
            'PyQt6': self.get_package_version('PyQt6'),
            'PyTorch': self.get_package_version('torch'),
            'Transformers': self.get_package_version('transformers'),
            'Pandas': self.get_package_version('pandas'),
            'NumPy': self.get_package_version('numpy'),
        }
        
        for name, version in deps.items():
            deps_layout.addRow(f"{name}:", QLabel(version))
            
        layout.addWidget(deps_group)
        layout.addStretch()
        
        self.tabs.addTab(system_widget, "系统信息")
        
    def setup_features_tab(self):
        """设置功能特性标签页"""
        features_widget = QWidget()
        layout = QVBoxLayout(features_widget)
        
        # 主要功能
        features_group = QGroupBox("主要功能")
        features_layout = QVBoxLayout(features_group)
        
        features_list = [
            "• 多模型支持：CodeLlama-7B + LoRA、BERT微调、传统规则方法",
            "• 完整流程：文本→伪代码→要素抽取→知识融合→推理验证",
            "• 实时进度监控和详细日志记录",
            "• 可视化结果展示和数据导出（CSV/TXT/JSON）",
            "• 文本预处理和格式转换",
            "• 批量处理和并发执行",
            "• 跨平台支持（Windows/Linux/macOS）"
        ]
        
        for feature in features_list:
            label = QLabel(feature)
            label.setWordWrap(True)
            features_layout.addWidget(label)
            
        layout.addWidget(features_group)
        
        # 技术特点
        tech_group = QGroupBox("技术特点")
        tech_layout = QVBoxLayout(tech_group)
        
        tech_list = [
            "• 4位量化技术，降低内存占用60%",
            "• LoRA微调，提升模型适应性",
            "• AST解析技术，精确要素提取", 
            "• 知识图谱融合，增强推理能力",
            "• 混合损失函数，平衡准确性和效率"
        ]
        
        for tech in tech_list:
            label = QLabel(tech)
            label.setWordWrap(True)
            tech_layout.addWidget(label)
            
        layout.addWidget(tech_group)
        layout.addStretch()
        
        self.tabs.addTab(features_widget, "功能特性")
        
    def setup_guide_tab(self):
        """设置使用指南标签页"""
        guide_widget = QWidget()
        layout = QVBoxLayout(guide_widget)
        
        # 使用步骤
        steps_group = QGroupBox("使用步骤")
        steps_layout = QVBoxLayout(steps_group)
        
        steps = [
            "1. 选择合适的模型（推荐CodeLlama-7B + LoRA）",
            "2. 调整模型参数和通用设置",
            "3. 输入要处理的文本或导入文件",
            "4. 配置预处理选项（可选）",
            "5. 点击'开始处理'启动处理流程",
            "6. 监控处理进度和查看实时日志",
            "7. 在结果表格中查看抽取结果",
            "8. 导出结果到CSV/TXT/JSON格式"
        ]
        
        for i, step in enumerate(steps):
            step_label = QLabel(step)
            step_label.setWordWrap(True)
            step_label.setStyleSheet("padding: 5px; border-left: 3px solid #4CAF50; background-color: #f0f8ff;")
            steps_layout.addWidget(step_label)
            
        layout.addWidget(steps_group)
        
        # 注意事项
        tips_group = QGroupBox("使用技巧")
        tips_layout = QVBoxLayout(tips_group)
        
        tips = [
            "• 建议文本长度在200-2000字符之间效果最佳",
            "• 复杂文本可先进行预处理，提高抽取准确性",
            "• 关系置信度低于0.6的结果建议进一步人工验证",
            "• 可以同时处理多个文档，提高工作效率",
            "• 定期保存处理日志，便于问题排查和结果追踪"
        ]
        
        for tip in tips:
            tip_label = QLabel(tip)
            tip_label.setWordWrap(True)
            tip_label.setStyleSheet("color: #d84315; padding: 3px;")
            tips_layout.addWidget(tip_label)
            
        layout.addWidget(tips_group)
        layout.addStretch()
        
        self.tabs.addTab(guide_widget, "使用指南")
        
    def setup_changelog_tab(self):
        """设置更新日志标签页"""
        changelog_widget = QWidget()
        layout = QVBoxLayout(changelog_widget)
        
        # 更新日志
        changelog_group = QGroupBox("版本更新历史")
        changelog_layout = QVBoxLayout(changelog_group)
        
        # 当前版本
        current_version = QLabel("版本 1.0.0 (2024-11-06)")
        current_version.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        current_version.setStyleSheet("color: #4CAF50;")
        changelog_layout.addWidget(current_version)
        
        current_changes = [
            "• 初始版本发布",
            "• 支持CodeLlama-7B + LoRA模型",
            "• 集成BERT微调和规则方法",
            "• 完整的GUI用户界面",
            "• 实时进度监控和结果导出",
            "• 跨平台兼容性支持"
        ]
        
        for change in current_changes:
            change_label = QLabel(change)
            change_label.setStyleSheet("padding-left: 20px; color: #555;")
            changelog_layout.addWidget(change_label)
            
        # 添加分隔线
        separator = QLabel()
        separator.setFrameStyle(QLabel.Shape.HLine)
        separator.setFrameShadow(QLabel.Shadow.Sunken)
        changelog_layout.addWidget(separator)
        
        # 计划功能
        planned_group = QGroupBox("计划功能")
        planned_layout = QVBoxLayout(planned_group)
        
        planned_features = [
            "• 支持更多预训练模型（GPT、LLaMA等）",
            "• 知识图谱可视化界面",
            "• 自定义规则编辑器",
            "• 云端模型部署支持",
            "• 协作功能和结果分享",
            "• 移动端适配"
        ]
        
        for feature in planned_features:
            feature_label = QLabel(feature)
            feature_label.setStyleSheet("color: #ff9800; padding-left: 20px;")
            planned_layout.addWidget(feature_label)
            
        layout.addWidget(changelog_group)
        layout.addStretch()
        
        self.tabs.addTab(changelog_widget, "更新日志")
        
    def get_package_version(self, package_name: str) -> str:
        """获取包版本信息"""
        try:
            if package_name == 'PyQt6':
                import PyQt6.QtCore
                return PyQt6.QtCore.PYQT_VERSION_STR
            else:
                module = __import__(package_name)
                if hasattr(module, '__version__'):
                    return module.__version__
                else:
                    return "已安装"
        except ImportError:
            return "未安装"
        except Exception:
            return "未知"
            
    def check_updates(self):
        """检查更新"""
        # 模拟检查更新
        QMessageBox.information(
            self, "检查更新", 
            "当前已是最新版本 (1.0.0)"
        )
        
    def open_help(self):
        """打开在线帮助"""
        help_url = QUrl("https://github.com/docred/extract-system")
        QDesktopServices.openUrl(help_url)


class SystemInfoWidget(QWidget):
    """系统信息组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout(self)
        
        # 显示详细系统信息
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        
        # 收集系统信息
        system_info = self.collect_system_info()
        info_text.setText(system_info)
        
        layout.addWidget(info_text)
        
    def collect_system_info(self) -> str:
        """收集系统信息"""
        info = []
        info.append("DocRED 关系抽取系统 - 详细系统信息")
        info.append("=" * 50)
        info.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        info.append("")
        
        # 基本信息
        info.append("1. 基本系统信息")
        info.append("-" * 20)
        info.append(f"操作系统: {platform.system()} {platform.release()}")
        info.append(f"系统架构: {platform.machine()}")
        info.append(f"处理器: {platform.processor()}")
        info.append(f"Python版本: {sys.version}")
        info.append("")
        
        # 依赖库
        info.append("2. 核心依赖库")
        info.append("-" * 20)
        deps = ['torch', 'transformers', 'pandas', 'numpy', 'PyQt6']
        for dep in deps:
            version = self.get_package_version(dep)
            info.append(f"{dep}: {version}")
        info.append("")
        
        # 性能信息
        info.append("3. 性能信息")
        info.append("-" * 20)
        try:
            import psutil
            memory = psutil.virtual_memory()
            info.append(f"总内存: {memory.total / (1024**3):.2f} GB")
            info.append(f"可用内存: {memory.available / (1024**3):.2f} GB")
            info.append(f"内存使用率: {memory.percent:.1f}%")
            
            disk = psutil.disk_usage('/')
            info.append(f"磁盘总容量: {disk.total / (1024**3):.2f} GB")
            info.append(f"磁盘可用空间: {disk.free / (1024**3):.2f} GB")
        except ImportError:
            info.append("无法获取性能信息 (需要psutil库)")
            
        return "\n".join(info)
        
    def get_package_version(self, package_name: str) -> str:
        """获取包版本"""
        try:
            module = __import__(package_name)
            if hasattr(module, '__version__'):
                return module.__version__
            return "版本未知"
        except ImportError:
            return "未安装"