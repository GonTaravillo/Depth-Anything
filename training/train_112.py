import argparse
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from depth_anything.dpt import DepthAnything
from dataset import OpenImagesDataset
from loss import KnowledgeDistillationLoss

def main():
    parser = argparse.ArgumentParser(description='Train Depth Anything Nano at 112x112 via KD')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=10, help='Total number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for L1 loss')
    parser.add_argument('--beta', type=float, default=0.3, help='Weight for Gradient loss')
    parser.add_argument('--gamma', type=float, default=0.2, help='Weight for SSIM loss')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "data/open_images_v7/train/data"
    print(f"Usando dispositivo: {device}")

    # 1. Dataset (carrega 518x518 originalmente)
    print("Cargando dataset...")
    if not os.path.exists(data_dir):
        print(f"Error: Directorio de dataset '{data_dir}' no encontrado. Por favor, asegúrate de tener los datos.")
        return
        
    dataset = OpenImagesDataset(data_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 2. Modelos
    print("Cargando modelo Teacher (Base 518x518)...")
    teacher = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitb14').to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    print("Cargando modelo Student (Nano)...")
    student_config = {
        'encoder': 'vitn',
        'use_bn': False,
        'use_clstoken': False,
        'localhub': True,
        'act_layer': torch.nn.ReLU
    }
    student = DepthAnything(student_config).to(device)
    
    if args.resume:
        print(f" -> Cargando pesos del checkpoint: {args.resume}")
        student.load_state_dict(torch.load(args.resume, map_location=device))
    else:
        print(" -> Inicializando weights desde cero para 112x112")
        with torch.no_grad():
            student.depth_head.scratch.output_conv2[2].bias.fill_(20.0)
        
    student.train()
    
    # 3. Optimizador
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = KnowledgeDistillationLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    os.makedirs("checkpoints", exist_ok=True)
    
    # 4. Bucle de Entrenamiento
    print("Comenzando el re-entrenamiento (Knowledge Distillation) para 112x112...")
    for epoch in range(args.epochs):
        student.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, images_518 in enumerate(progress_bar):
            images_518 = images_518.to(device)
            
            # --- PROFESOR (518x518) ---
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    # Prediccion en alta resolucion [Batch, H, W]
                    teacher_preds_518 = teacher(images_518)
            
            # Redimensionar la imagen de entrada a 112x112 para el Alumno
            images_112 = F.interpolate(images_518, size=(112, 112), mode='bilinear', align_corners=False)
            
            # Redimensionar la verdad fundamental (teacher) a 112x112 para calcular la Loss
            teacher_preds_112 = F.interpolate(teacher_preds_518.unsqueeze(1), size=(112, 112), mode='bilinear', align_corners=False).squeeze(1)
            
            # --- ALUMNO (112x112) ---
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                student_preds_112 = student(images_112)
                loss = criterion(student_preds_112, teacher_preds_112)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} finalizada. Pérdida promedio (a 112x112): {avg_loss:.4f}")
        
        checkpoint_path = f"checkpoints/depth_anything_nano_112x112_epoch_{epoch+1}.pth"
        torch.save(student.state_dict(), checkpoint_path)
        print(f"Checkpoint guardado en {checkpoint_path}")

if __name__ == "__main__":
    main()
