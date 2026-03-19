import sys
import os
import torch
import torch.nn as nn

# Asegurarse de que se puede importar depth_anything
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from depth_anything.dpt import DepthAnything

def verify_activations(model, expected_act_name):
    print(f"\nVerificando si el modelo usa {expected_act_name}...")
    
    # Check DPTHead output_conv
    output_conv = model.depth_head.scratch.output_conv2
    act_found = False
    for module in output_conv:
        if isinstance(module, (nn.ReLU, nn.GELU)):
            print(f" - Encontrada activación en output_conv2: {type(module).__name__}")
            if type(module).__name__ == expected_act_name:
                act_found = True
            else:
                print(f"   ❌ ERROR: Se esperaba {expected_act_name}")
                return False

    # Check a fusion block
    fusion_block = model.depth_head.scratch.refinenet1
    res_unit = fusion_block.resConfUnit1
    print(f" - Encontrada activación en refinenet1.resConfUnit1: {type(res_unit.activation).__name__}")
    if type(res_unit.activation).__name__ != expected_act_name:
         print(f"   ❌ ERROR: Se esperaba {expected_act_name}")
         return False
    
    # Check DPT_DINOv2 main activation
    print(f" - Encontrada activación principal en DPT_DINOv2: {type(model.activation).__name__}")
    if type(model.activation).__name__ != expected_act_name:
         print(f"   ❌ ERROR: Se esperaba {expected_act_name}")
         return False

    return True

def test_teacher_student_activations():
    # 1. Test Student (ReLU)
    student_config = {
        'encoder': 'vitn',
        'use_bn': False,
        'use_clstoken': False,
        'localhub': True,
        'act_layer': torch.nn.ReLU
    }
    print("Iniciando Verificación de STUDENT...")
    student = DepthAnything(student_config)
    if verify_activations(student, "ReLU"):
        print("✅ Verificación de STUDENT (ReLU) EXITOSA")
    else:
        print("❌ Verificación de STUDENT (ReLU) FALLIDA")

    # 2. Test Teacher (GELU)
    teacher_config = {
        'encoder': 'vits', # Usamos vits para el test por ser mas ligero que vitb
        'use_bn': False,
        'use_clstoken': False,
        'localhub': True,
        'act_layer': torch.nn.GELU
    }
    print("\nIniciando Verificación de TEACHER...")
    teacher = DepthAnything(teacher_config)
    if verify_activations(teacher, "GELU"):
        print("✅ Verificación de TEACHER (GELU) EXITOSA")
    else:
        print("❌ Verificación de TEACHER (GELU) FALLIDA")

if __name__ == '__main__':
    test_teacher_student_activations()
