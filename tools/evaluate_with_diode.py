import cv2
import numpy as np
import os
import glob
import onnxruntime as ort
from PIL import Image
from transformers import pipeline

def normalize_image(image_rgb):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    img = image_rgb.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1) # HWC to CHW
    img = np.expand_dims(img, axis=0) # CHW to BCHW
    return img

def align_depth(pred, gt):
    """Align prediction to Ground Truth using least squares (scale + shift)"""
    mask = (gt > 0) & (gt < 100) # DIODE can have far invalid ranges
    if not np.any(mask):
        return pred
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    
    A = np.vstack([pred_valid, np.ones_like(pred_valid)]).T
    try:
        m, c = np.linalg.lstsq(A, gt_valid, rcond=None)[0]
        return pred * m + c
    except:
        return pred

def compute_metrics(pred, gt):
    """Compute RMSE and AbsRel on valid pixels"""
    mask = (gt > 0) & (gt < 100)
    if not np.any(mask):
        return 0.0, 0.0
    
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    pred_valid = np.clip(pred_valid, 1e-3, 100)
    
    rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    return rmse, abs_rel

def colorize(d):
    d_min, d_max = d.min(), d.max()
    d_norm = (d - d_min) / (d_max - d_min + 1e-8) * 255.0
    return cv2.applyColorMap(d_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)

def main():
    # TODO: Cambia esta ruta a la carpeta donde tengas (o vayas a tener) los archivos de DIODE
    # DIODE guarda pares de archivos: "nombre.png" y "nombre_depth.npy"
    diode_folder = 'diode_samples/' 
    
    if not os.path.exists(diode_folder):
        print(f"Carpeta {diode_folder} no encontrada. Por favor, crea la carpeta y mete algunas imágenes.")
        return

    # Buscar imágenes de DIODE recursivamente (suelen ser PNG)
    image_files = glob.glob(os.path.join(diode_folder, "**/*.png"), recursive=True)
    if len(image_files) == 0:
        print("No se encontraron imágenes .png en la carpeta.")
        return

    onnx_path = 'depth_anything_nano_112x112_trained_clean.onnx'
    if not os.path.exists(onnx_path):
        onnx_path = 'rpi_deploy/depth_anything_nano_112x112_trained_clean.onnx'
             
    print("Cargando modelo ONNX Nano...")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, session_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    print("Cargando Depth-Anything ViT-B (HuggingFace)...")
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf", device=-1)

    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Reporte de Evaluación Ground Truth (DIODE)", ln=True, align='C')
    pdf.ln(10)

    metrics_vitb = {"rmse": [], "abs_rel": []}
    metrics_nano = {"rmse": [], "abs_rel": []}

    num_samples = 10
    image_files = image_files[:num_samples] # Procesar 10 imágenes

    for i, img_path in enumerate(image_files):
        depth_path = img_path.replace('.png', '_depth.npy')
        
        if not os.path.exists(depth_path):
            print(f"No se encontró Ground Truth para {img_path}. Saltando...")
            continue
            
        print(f"\n--- Procesando {os.path.basename(img_path)} ---")
        
        # Cargar imagen y Ground Truth
        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gt_depth_np = np.load(depth_path).squeeze() # Cargar el array .npy denso de DIODE
        
        # 1. ViT-Base Inference
        pil_image = Image.fromarray(image_rgb)
        depth_vitb_pil = pipe(pil_image)["depth"]
        depth_vitb_np = np.array(depth_vitb_pil, dtype=np.float32)
        
        depth_vitb_resized = cv2.resize(depth_vitb_np, (gt_depth_np.shape[1], gt_depth_np.shape[0]))
        depth_vitb_aligned = align_depth(depth_vitb_resized, gt_depth_np)
        
        rmse_v, absrel_v = compute_metrics(depth_vitb_aligned, gt_depth_np)
        metrics_vitb["rmse"].append(rmse_v)
        metrics_vitb["abs_rel"].append(absrel_v)
        print(f"ViT-Base -> RMSE: {rmse_v:.3f}, AbsRel: {absrel_v:.3f}")

        # 2. Nano FP32 Inference
        image_resized_112 = cv2.resize(image_rgb, (112, 112))
        input_tensor = normalize_image(image_resized_112)
        outputs = session.run(None, {input_name: input_tensor})
        depth_nano = np.squeeze(outputs[0])
        
        depth_nano_resized = cv2.resize(depth_nano, (gt_depth_np.shape[1], gt_depth_np.shape[0]))
        depth_nano_aligned = align_depth(depth_nano_resized, gt_depth_np)
        
        rmse_n, absrel_n = compute_metrics(depth_nano_aligned, gt_depth_np)
        metrics_nano["rmse"].append(rmse_n)
        metrics_nano["abs_rel"].append(absrel_n)
        print(f"Nano FP32 -> RMSE: {rmse_n:.3f}, AbsRel: {absrel_n:.3f}")

        # 3. Visualización
        color_gt = colorize(gt_depth_np)
        color_vitb = colorize(depth_vitb_aligned)
        color_nano = colorize(depth_nano_aligned)
        
        target_h = 480
        target_w = int(image_bgr.shape[1] * (target_h / image_bgr.shape[0]))
        
        orig_r = cv2.resize(image_bgr, (target_w, target_h))
        color_gt_r = cv2.resize(color_gt, (target_w, target_h))
        color_vitb_r = cv2.resize(color_vitb, (target_w, target_h))
        color_nano_r = cv2.resize(color_nano, (target_w, target_h))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(orig_r, "Original (DIODE)", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(color_gt_r, "Ground Truth", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(color_vitb_r, f"ViT-B (RMSE:{rmse_v:.2f})", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(color_nano_r, f"Nano (RMSE:{rmse_n:.2f})", (10, 30), font, 1, (255, 255, 255), 2)
        
        separator = np.zeros((target_h, 10, 3), dtype=np.uint8)
        combined = cv2.hconcat([orig_r, separator, color_gt_r, separator, color_vitb_r, separator, color_nano_r])
        
        out_path = f'diode_comparison_pdf_{i}.png'
        cv2.imwrite(out_path, combined)
        
        pdf.add_page()
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, f"Escena {i+1}: {os.path.basename(img_path)}", ln=True)
        pdf.set_font("helvetica", "", 12)
        pdf.cell(0, 8, f"ViT-Base -> RMSE: {rmse_v:.3f} | AbsRel: {absrel_v:.3f}", ln=True)
        pdf.cell(0, 8, f"Nano FP32 -> RMSE: {rmse_n:.3f} | AbsRel: {absrel_n:.3f}", ln=True)
        pdf.ln(5)
        pdf.image(out_path, x=10, w=190)

    if len(metrics_vitb["rmse"]) > 0:
        pdf.add_page()
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, f"Resumen Global ({len(image_files)} Imágenes)", ln=True)
        pdf.ln(5)
        pdf.set_font("helvetica", "", 12)
        
        mean_rmse_v = np.mean(metrics_vitb["rmse"])
        mean_abs_v = np.mean(metrics_vitb["abs_rel"])
        mean_rmse_n = np.mean(metrics_nano["rmse"])
        mean_abs_n = np.mean(metrics_nano["abs_rel"])
        
        pdf.cell(0, 10, f"ViT-Base FP32 - Promedio RMSE: {mean_rmse_v:.3f} | Promedio AbsRel: {mean_abs_v:.3f}", ln=True)
        pdf.cell(0, 10, f"Nano FP32     - Promedio RMSE: {mean_rmse_n:.3f} | Promedio AbsRel: {mean_abs_n:.3f}", ln=True)
        
        pdf_path = "reporte_metricas_diode.pdf"
        pdf.output(pdf_path)
        print(f"\nReporte PDF guardado en: {pdf_path}")

if __name__ == "__main__":
    main()
