import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleGradientLoss(nn.Module):
    def __init__(self, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        
    def forward(self, pred, target):
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
            
        total_loss = 0
        for s in self.scales:
            # Calcular gradientes a diferentes distancias (escalas)
            dx_pred = torch.abs(pred[:, :, :, :-s] - pred[:, :, :, s:])
            dy_pred = torch.abs(pred[:, :, :-s, :] - pred[:, :, s:, :])
            
            dx_target = torch.abs(target[:, :, :, :-s] - target[:, :, :, s:])
            dy_target = torch.abs(target[:, :, :-s, :] - target[:, :, s:, :])
            
            total_loss += F.l1_loss(dx_pred, dx_target) + F.l1_loss(dy_pred, dy_target)
            
        return total_loss / len(self.scales)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        
    def forward(self, pred, target):
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
            
        pred = pred.float()
        target = target.float()
            
        # Simplificación de SSIM para eficiencia en entrenamiento
        mu_x = F.avg_pool2d(pred, self.window_size, stride=1, padding=self.window_size//2)
        mu_y = F.avg_pool2d(target, self.window_size, stride=1, padding=self.window_size//2)
        
        sigma_x = F.avg_pool2d(pred * pred, self.window_size, stride=1, padding=self.window_size//2) - mu_x * mu_x
        sigma_y = F.avg_pool2d(target * target, self.window_size, stride=1, padding=self.window_size//2) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(pred * target, self.window_size, stride=1, padding=self.window_size//2) - mu_x * mu_y
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
        return 1 - ssim_map.mean()

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        """
        alpha: peso para L1 (MAE) - Valores globales.
        beta: peso para MultiScaleGradient - Bordes y nitidez.
        gamma: peso para SSIM - Estructura y suavidad.
        """
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.grad_loss = MultiScaleGradientLoss()
        self.ssim_loss = SSIMLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, student_pred, teacher_pred):
        if student_pred.shape != teacher_pred.shape:
            teacher_pred = F.interpolate(
                teacher_pred.unsqueeze(1) if teacher_pred.dim() == 3 else teacher_pred,
                size=student_pred.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            if teacher_pred.shape[1] == 1 and student_pred.dim() == 3:
                teacher_pred = teacher_pred.squeeze(1)
            
        l1 = self.l1_loss(student_pred, teacher_pred)
        grad = self.grad_loss(student_pred, teacher_pred)
        ssim = self.ssim_loss(student_pred, teacher_pred)
        
        return self.alpha * l1 + self.beta * grad + self.gamma * ssim
