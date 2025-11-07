# -*- coding: utf-8 -*-
"""
DocRED GUI 模块
完整的PyQt6用户界面系统
"""

from .main_window import DocRedMainWindow
from .widgets import ProgressPanel, ResultTable, ModelSelector
from .components import TextInputWidget, ControlPanel, AboutDialog

__version__ = "1.0.0"
__author__ = "DocRED Team"

__all__ = [
    'DocRedMainWindow',
    'ProgressPanel',
    'ResultTable', 
    'ModelSelector',
    'TextInputWidget',
    'ControlPanel',
    'AboutDialog'
]