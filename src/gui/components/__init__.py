# -*- coding: utf-8 -*-
"""
DocRED GUI 组件模块
"""

from .text_input import TextInputWidget
from .control_panel import ControlPanel
from .about_dialog import AboutDialog
from .enhanced_processor import EnhancedProcessingWorker

__all__ = [
    'TextInputWidget',
    'ControlPanel', 
    'AboutDialog',
    'EnhancedProcessingWorker'
]