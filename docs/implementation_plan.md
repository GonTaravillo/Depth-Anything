# Análisis y Plan para ESP32-S3 (Korvo-2)

Para reducir el modelo "al extremo" y hacerlo funcionar en un ESP32-S3, nos enfrentamos a dos retos gigantescos:
1. **Memoria de Hardware:** El ESP32-S3-Korvo-2 tiene normalmente entre 2MB y 8MB de PSRAM. La versión más pequeña de Depth Anything (ViT-Small) tiene ~24.4 Millones de parámetros. En \`float32\` ocupa casi 100 MB y en \`int8\` unos 24 MB. Literalmente **no cabe en la memoria** del chip a menos que reemplacemos el encoder DINOv2 por uno microscópico.
2. **Operadores Soportados (ESP-DL):** Según tu archivo \[operator_support_state.md\](file:///home/gonzalo/Documents/PFG/Depth-Anything/operator_support_state.md), el acelerador neuronal de Espressif tiene restricciones severas en cómo procesa los datos.

## Análisis de Operadores Incompatibles

Al revisar \[depth_anything/dpt.py\](file:///home/gonzalo/Documents/PFG/Depth-Anything/depth_anything/dpt.py), \[depth_anything/blocks.py\](file:///home/gonzalo/Documents/PFG/Depth-Anything/depth_anything/blocks.py) y la arquitectura interna de DINOv2, he localizado las siguientes incompatibilidades y cómo solucionarlas:

### 1. Activación GELU (¡Incompatible!)
* **Problema:** En el archivo \`dpt.py\` (línea 67) y dentro de todos los bloques transformer del modelo de Meta (DINOv2), se utiliza la función de activación matemática \`GELU\`. Si revisas tu archivo de soporte, no existe. Solamente tienes soportados Elu, Relu, LeakyRelu, PRelu, Swish y Sigmoid.
* **Solución:** Hay que modificar el código para sustituir `nn.GELU()` por `nn.ReLU()` o `nn.SiLU()` (que matemáticamente es equivalente a Swish, soportado por ESP). Esto requerirá reentrenar el modelo, ya que cambiar la activación por defecto destruye los pesos preentrenados.

### 2. Operadores de Convolución y Multiplicación de Matrices en Float32
* **Problema:** Según tu lista, los operadores \`Conv\`, \`ConvTranspose\`, \`Gemm\` y \`MatMul\` **tienen una cruz verde (❌) en la columna de float32**. El chip bloquea la ejecución en coma flotante nativa para capas pesadas.
* **Solución:** Tienes que **cuantizar el modelo a \`int8\` o \`int16\`** forzosamente ("symmetric quantization"). No podrás correr el modelo en coma flotante.

### 3. Operador Resize / Interpolate (Punto Crítico)
* **Problema:** En el código (ej. línea 133 de \`dpt.py\` usan \`F.interpolate(..., align_corners=True)\`). Según tu archivo, el operador \`Resize\` **solo tiene tic (✔) en \`int8\`** y cruces en int16 y float32. ¡Toda la red que hace \`interpolate\` tiene que estar en \`int8\`!
* **Solución:** Además de cuantizar, al exportar a ONNX tendrás que asegurarte de usar opset 18 (como sugiere tu documento) y comprobar que el flag \`align_corners=True\` es bien asimilado por el Resize bilineal de ESP-DL.

### 4. Batch Normalization (\`nn.BatchNorm2d\`)
* **Problema:** En \[depth_anything/blocks.py\](file:///home/gonzalo/Documents/PFG/Depth-Anything/depth_anything/blocks.py) se declara \`nn.BatchNorm2d\` si habilitas el flag \`use_bn=True\`. ESP-DL no tiene un operador para "LayerNormalization" o BatchNorm independiente 2D.
* **Solución:** Al exportar a ONNX utilizar una utilidad de PyTorch que fusione los pesos dinámicos de las convoluciones con el BatchNorm (\`fuse_conv_bn\`) de forma que desaparezca de la gráfica y solo quede la Convolución pura. (Curiosamente `LayerNormalization`, usado en los ViT, sí está).

## Plan de Modificación Sugerido (Prueba de Concepto)

### [MODIFY] [depth_anything/dpt.py](file:///home/gonzalo/Documents/PFG/Depth-Anything/depth_anything/dpt.py)
Se modificarán las arquitecturas reemplazando \`GELU\` por \`ReLU\` en el \`readout_projects\`. 
Más importante aún, para que el modelo "quepa", habría que programar un "Mock DINOv2" (un modelo pequeñito de convoluciones estilo MobileNetV2 de < 2 Millones de parámetros) que suplante a la bestia de *vits*.

### [MODIFY] [depth_anything/blocks.py](file:///home/gonzalo/Documents/PFG/Depth-Anything/depth_anything/blocks.py)
Me aseguraré de forzar el parámetro \`bn=False\` (eliminando el uso de BatchNorm2d explícito para evitar problemas en exportación), ya que está implementado de esa manera como seguridad pero es mejor quitarlo de raíz para este hardware.

## Verificación
No podemos verificar el comportamiento de la placa física aquí, pero podemos generar los archivos de código modificados preparados para ser cuantizados con ESP-PPQ y exportados a ONNX (opset 18) de manera segura.

¿Quieres que proceda a realizar estas modificaciones de refactorización de código en los archivos [dpt.py](file:///home/gonzalo/Documents/PFG/Depth-Anything/depth_anything/dpt.py) y [blocks.py](file:///home/gonzalo/Documents/PFG/Depth-Anything/depth_anything/blocks.py) para dejarlo como base pura "Friendy for ESP32", o prefieres que intentemos primero instanciar un "Dummy Encoder" para suplir al gigantesco DINOv2 que trae por defecto para que directamente quepa en los 8MB?
