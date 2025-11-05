import torch
import torch.nn as nn

class LogicConsistencyLoss(nn.Module):
    def __init__(self, logic_weight=0.3):
        super().__init__()
        self.logic_weight = logic_weight
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, outputs, labels, logic_labels):
        """
        混合损失：交叉熵损失 + 逻辑一致性损失
        - outputs: 模型输出的logits
        - labels: 标准自回归标签（-100掩码）
        - logic_labels: 逻辑一致性标签（1=一致，0=不一致）
        """
        # 基础交叉熵损失
        ce_loss = self.cross_entropy(
            outputs.logits.transpose(1, 2), 
            labels
        )
        
        # 逻辑一致性损失（惩罚违反规则的样本）
        # 从输出中提取关系分类的logits（简化版）
        rel_logits = outputs.logits[:, -1, :]  # 假设最后一个token是关系判断
        logic_loss = self.cross_entropy(
            rel_logits, 
            logic_labels.long()
        )
        
        # 总损失
        total_loss = ce_loss + self.logic_weight * logic_loss
        return total_loss