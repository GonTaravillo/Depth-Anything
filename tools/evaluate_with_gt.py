import cv2
import numpy as np
import os
from PIL import Image
import onnxruntime as ort
from transformers import pipeline
from fpdf import FPDF

try:
    from datasets import load_dataset
except ImportError:
    print("Please run: pip install datasets")
    exit(1)

def normalize_image(image_rgb):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = image_rgb.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

def align_depth(pred, gt):
    mask = gt > 0
    if not np.any(mask): return pred
    pred_valid, gt_valid = pred[mask], gt[mask]
    A = np.vstack([pred_valid, np.ones_like(pred_valid)]).T
    try:
        m, c = np.linalg.lstsq(A, gt_valid, rcond=None)[0]
        return pred * m + c
    except:
        return pred

def compute_metrics(pred, gt):
    mask = gt > 0
    if not np.any(mask): return 0.0, 0.0
    pred_valid, gt_valid = pred[mask], gt[mask]
    pred_valid = np.clip(pred_valid, 1e-3, 80)
    rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    return rmse, abs_rel

def main():
    print("Loading NYU Depth V2 validation set...")
    dataset = load_dataset("sayakpaul/nyu_depth_v2", split="validation", streaming=True, trust_remote_code=True)
    
    onnx_path = 'depth_anything_nano_112x112_trained_clean.onnx'
    if not os.path.exists(onnx_path):
        onnx_path = 'rpi_deploy/depth_anything_nano_112x112_trained_clean.onnx'
             
    print("Loading ONNX Nano model...")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, session_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    print("Loading Depth-Anything ViT-B...")
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf", device=-1)

    metrics_vitb = {"rmse": [], "abs_rel": []}
    metrics_nano = {"rmse": [], "abs_rel": []}

    num_samples = 10
    dataset_iter = iter(dataset)
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Reporte de Evaluación Ground Truth (NYU Depth V2)", ln=True, align='C')
    pdf.ln(10)

    for i in range(num_samples):
        print(f"\n--- Processing Image {i+1}/{num_samples} ---")
        try:
            sample = next(dataset_iter)
        except StopIteration:
            break
            
        pil_image, pil_depth = sample["image"], sample["depth_map"]
        gt_depth_np = np.array(pil_depth, dtype=np.float32)
        image_rgb = np.array(pil_image)
        
        # 1. ViT-Base
        depth_vitb_pil = pipe(pil_image)["depth"]
        depth_vitb_np = np.array(depth_vitb_pil, dtype=np.float32)
        depth_vitb_resized = cv2.resize(depth_vitb_np, (gt_depth_np.shape[1], gt_depth_np.shape[0]))
        depth_vitb_aligned = align_depth(depth_vitb_resized, gt_depth_np)
        rmse_v, absrel_v = compute_metrics(depth_vitb_aligned, gt_depth_np)
        metrics_vitb["rmse"].append(rmse_v)
        metrics_vitb["abs_rel"].append(absrel_v)

        # 2. Nano
        image_resized_112 = cv2.resize(image_rgb, (112, 112))
        outputs = session.run(None, {input_name: normalize_image(image_resized_112)})
        depth_nano = np.squeeze(outputs[0])
        depth_nano_resized = cv2.resize(depth_nano, (gt_depth_np.shape[1], gt_depth_np.shape[0]))
        depth_nano_aligned = align_depth(depth_nano_resized, gt_depth_np)
        rmse_n, absrel_n = compute_metrics(depth_nano_aligned, gt_depth_np)
        metrics_nano["rmse"].append(rmse_n)
        metrics_nano["abs_rel"].append(absrel_n)

        # 3. Visualization
        def colorize(d):
            d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8) * 255.0
            return cv2.applyColorMap(d_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
            
        color_gt = colorize(gt_depth_np)
        color_vitb = colorize(depth_vitb_aligned)
        color_nano = colorize(depth_nano_aligned)
        orig_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        target_h = 480
        target_w = int(orig_bgr.shape[1] * (target_h / orig_bgr.shape[0]))
        orig_bgr = cv2.resize(orig_bgr, (target_w, target_h))
        color_gt = cv2.resize(color_gt, (target_w, target_h))
        color_vitb = cv2.resize(color_vitb, (target_w, target_h))
        color_nano = cv2.resize(color_nano, (target_w, target_h))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(orig_bgr, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(color_gt, "Ground Truth", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(color_vitb, "ViT-B", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(color_nano, "Nano", (10, 30), font, 1, (255, 255, 255), 2)
        
        sep = np.zeros((target_h, 10, 3), dtype=np.uint8)
        combined = cv2.hconcat([orig_bgr, sep, color_gt, sep, color_vitb, sep, color_nano])
        img_path = f'gt_comparison_pdf_{i}.png'
        cv2.imwrite(img_path, combined)
        
        # Add to PDF
        pdf.add_page()
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, f"Escena {i+1}", ln=True)
        pdf.set_font("helvetica", "", 12)
        pdf.cell(0, 8, f"ViT-Base -> RMSE: {rmse_v:.3f} | AbsRel: {absrel_v:.3f}", ln=True)
        pdf.cell(0, 8, f"Nano FP32 -> RMSE: {rmse_n:.3f} | AbsRel: {absrel_n:.3f}", ln=True)
        pdf.ln(5)
        # combined image is very wide, let's fit to width
        pdf.image(img_path, x=10, w=190)

    # Summary Page
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Resumen Global (10 Imágenes)", ln=True)
    pdf.ln(5)
    pdf.set_font("helvetica", "", 12)
    mean_rmse_v = np.mean(metrics_vitb["rmse"])
    mean_abs_v = np.mean(metrics_vitb["abs_rel"])
    mean_rmse_n = np.mean(metrics_nano["rmse"])
    mean_abs_n = np.mean(metrics_nano["abs_rel"])
    
    pdf.cell(0, 10, f"ViT-Base FP32 - Promedio RMSE: {mean_rmse_v:.3f} | Promedio AbsRel: {mean_abs_v:.3f}", ln=True)
    pdf.cell(0, 10, f"Nano FP32     - Promedio RMSE: {mean_rmse_n:.3f} | Promedio AbsRel: {mean_abs_n:.3f}", ln=True)
    
    pdf_path = "reporte_metricas_nyu.pdf"
    pdf.output(pdf_path)
    print(f"\nReporte guardado en: {pdf_path}")

if __name__ == "__main__":
    main()
