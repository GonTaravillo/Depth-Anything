# Reducción de Arquitectura para Depth Anything y DINOv2 (Orientado a Microcontroladores como ESP32-S3)

Esta guía detalla los archivos y parámetros específicos que deben ser modificados para reducir enormemente la carga computacional y el uso de memoria RAM (PSRAM/SRAM) de los modelos Depth-Anything y su *backbone* DINOv2.

## 1. Archivos a Modificar

1. **`depth_anything/dpt.py`**: Contiene la definición de la cabeza del modelo (la parte que decodifica y genera el mapa de profundidad) y la carga del *backbone* DINOv2 (`class DPT_DINOv2` y `class DPTHead`).
2. **`depth_anything/blocks.py`**: Define los bloques de construcción individuales del decodificador, concretamente los bloques de fusión (`FeatureFusionBlock`) y convolucionales residuales (`ResidualConvUnit`).
3. **Código fuente de DINOv2 (Vision Transformer)**: Dado que DINOv2 se carga a través de `torch.hub.load`, para modificar su arquitectura interna deberás buscar el código local (o extraer el archivo `vision_transformer.py` del repositorio de DINOv2) e instanciar la clase `VisionTransformer` manualmente en el proyecto en lugar de descargar el modelo automáticamente.

---

## 2. Parámetros a Modificar y su Impacto

### A. En el *Backbone* DINOv2 (`vision_transformer.py`)

El *encoder* original es demasiado grande para un ESP32. Para hacerlo a medida se reducen estos hiperparámetros:

*   **`embed_dim` (Dimensión de Embedding)**:
    *   **Cambio:** Reducir de `384` (ViT-Small) a `192`, `128` o incluso `64`.
    *   **Impacto en Rendimiento:** Disminuye drásticamente los parámetros de las capas lineales (QKV y MLP). Menos uso de memoria y menor tiempo de inferencia (MACs).
    *   **Impacto en el Resultado:** El modelo tendrá menos "capacidad" mental para representar características. Puede que no entienda texturas complejas o dependencias sutiles.
*   **`depth` (Número de Capas/Bloques del Transformador)**:
    *   **Cambio:** Reducir de `12` capas a `4` o `6`.
    *   **Impacto en Rendimiento:** El tiempo de inferencia se reduce de forma casi lineal (la mitad de capas = casi la mitad de retraso/latencia).
    *   **Impacto en el Resultado:** Menor capacidad para procesar la información a nivel muy abstracto. El modelo puede perder entendimiento global profundo de la geometría de la escena.
*   **`num_heads` (Cabezas de Atención)**:
    *   **Cambio:** Reducir de `6` a `2` o `3` (debe ser divisor de `embed_dim`).
    *   **Impacto en Rendimiento:** Acelera el cálculo paralelo de atención y ahorra algo de memoria RAM durante la inferencia.
    *   **Impacto en el Resultado:** Reduce la capacidad del modelo de enfocarse en múltiples aspectos visuales a la vez (ej. texturas, bordes y colores individualmente).
*   **`patch_size` (Tamaño del Parche)**:
    *   **Cambio:** Aumentar de `14` a `16` o `32`.
    *   **Impacto en Rendimiento:** **Crítico para hardware limitado**. La complejidad de la atención es cuadrática ($O(N^2)$). Pasar de 14 a 32 reduce las secuencias de atención y ahorra muchísima RAM dinámica.
    *   **Impacto en el Resultado:** Los mapas de profundidad serán mucho más "cuadriculados" o borrosos espacialmente. Se perderán detalles de objetos muy pequeños y bordes.

### B. En el Decodificador Depth Anything (`depth_anything/dpt.py`)

*   **`features` (en la clase `DPTHead`)**:
    *   **Cambio:** El valor por defecto es `256`. Puedes probar reduciéndolo a `128`, `64` o `32`.
    *   **Impacto en Rendimiento:** Controla el tamaño (canales) de todas las convoluciones en el bloque refinenet. Cae drásticamente el uso de memoria de los mapas de características intermedios.
    *   **Impacto en el Resultado:** El modelo podría tener problemas para reconstruir los bordes de alta resolución de la imagen y generará gradientes (transiciones de cercano a lejano) de profundidad más toscos o escalonados.

### C. En los Bloques de Fusión (`depth_anything/blocks.py`)

*   **Simplificar `ResidualConvUnit`**:
    *   **Cambio:** Por defecto, esta clase aplica dos convoluciones estándar de 3x3 por cada escala espacial. Para reducir carga, recorta esto eliminando la segunda convolución `conv2` o reemplazándolas por **Depthwise Separable Convolutions**.
    *   **Impacto en Rendimiento:** Las convoluciones de 3x3 consumen muchos ciclos. Transformar o quitar una reduce significativamente la latencia.
    *   **Impacto en el Resultado:** El alisado (smoothness) del mapa de profundidad final y la resolución de contornos de la imagen empeorará visiblemente sin estas capas de refinamiento extra.

---

## 3. Consideración Crítica: Re-entrenamiento

Al modificar de forma estructural la arquitectura (dimensiones, tamaños, número de capas, eliminación de convoluciones), **los pesos pre-entrenados del modelo original descargado dejarán de funcionar.** Las matrices matemáticas de los pesos dejarán de encajar.

Será estrictamente necesario entrenar el modelo modificado. La mejor técnica es el **Knowledge Distillation** (Destilación de Conocimiento), donde instanciarás el modelo original pesado (Profesor) y el nuevo modelo miniatura que has creado modificado las variables de arriba (Estudiante). Se entrena al modelo estudiante para que intente igualar todos los *outputs* e imágenes de profundidad producidos por el Profesor.
