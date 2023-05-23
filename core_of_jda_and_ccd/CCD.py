import torch
import torch.nn as nn
import torch.nn.functional as F
class CCDLoss(nn.KLDivLoss):
    def __init__(self, temperature, alpha=None, beta=None, p=0.5,reduction='batchmean',apply_norm=False, **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.alpha = alpha
        self.apply_norm = apply_norm
        self.beta = 1 - alpha if beta is None else beta
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.p=p
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction, **kwargs)
        self.kl_loss=nn.KLDivLoss(reduction="none")
        self.momentum=0.99


    def norm(self,input,eps=1e-6):
        input = (input - input.mean(1,keepdim=True))/(input.std(1,keepdim=True)+eps)
        return input
    def forward(self, student_output, teacher_output, *args, **kwargs):
        if self.norm:
            student_output = self.norm(student_output)
            teacher_output = self.norm(teacher_output)
        b1_indices = torch.arange(student_output.shape[0]) % 2 == 0
        b2_indices = torch.arange(student_output.shape[0]) % 2 != 0
        original_soft_loss = super().forward(torch.log_softmax(student_output[b1_indices] / self.temperature, dim=1),
                                    torch.softmax(teacher_output[b1_indices] / self.temperature, dim=1))
        b1=teacher_output[b1_indices]
        b2=teacher_output[b2_indices]
        cosine=F.cosine_similarity(b1,b2)+1
        augmented_soft_loss = self.kl_loss(torch.log_softmax(student_output[b2_indices] / self.temperature, dim=1),
                                    torch.softmax(teacher_output[b2_indices] / self.temperature, dim=1))*cosine.unsqueeze(-1)
        augmented_soft_loss = augmented_soft_loss.sum(-1).mean()
        soft_loss=(original_soft_loss+augmented_soft_loss)/2
        return (self.temperature ** 2) * soft_loss
