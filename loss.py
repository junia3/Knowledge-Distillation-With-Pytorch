import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self, T, alpha):
        super().__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, student_pred, target, teacher_pred):
        student_loss = nn.CrossEntropyLoss()(student_pred, target)
        soft_target = F.softmax(teacher_pred/self.T, dim=1) # softmax on teacher prediction
        soft_pred = F.log_softmax(student_pred/self.T, dim=1) # log softmax on student prediction
        teacher_loss = nn.KLDivLoss(reduction="batchmean")(soft_pred, soft_target) # KL divergence -> cross entropy loss
        loss = self.alpha*self.T*self.T*teacher_loss + (1-self.alpha)*student_loss
        return loss