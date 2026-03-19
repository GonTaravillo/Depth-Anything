# Historia Técnica del PFG: Desarrollo de Depth Anything Nano en Antigravity

Este documento recopila la trayectoria completa del proyecto, desde la concepción de la idea de portabilidad hasta el entrenamiento final del modelo optimizado. 

---

## HITO 1: Concepción y Portabilidad (ESP32-S3 Korvo)
**Sesiones:** `d263008c`, `385bb27d`

**Q&A Clave:**
- **USER:** ¿Puedo correr Depth Anything en un ESP32-S3?
- **ANTIGRAVITY:** Es un gran reto técnico. Necesitamos resolver tres bloqueos: 
  1. La memoria (PSRAM limitada), 
  2. Las operaciones no soportadas (GELU), 
  3. El tamaño del encoder (DINOv2).
- **USER:** ¿Cómo eliminamos GELU?
- **ANTIGRAVITY:** Proponemos sustituir GELU por ReLU en todo el modelo. El ESP32 tiene aceleración por hardware para ReLU, lo que hará el modelo mucho más rápido y ligero.

---

## HITO 2: Reducción de Arquitectura (Creación de "Nano")
**Sesiones:** `7364b0fc`, `af3f2deb`

**Q&A Clave:**
- **USER:** El modelo original es demasiado pesado. ¿Cómo lo reducimos?
- **ANTIGRAVITY:** Hemos "podado" las capas del encoder DINOv2 y reducido las dimensiones del DPTHead.
- **RESULTADO:** Pasamos de un modelo masivo a una versión de **3.73M de parámetros**, compatible con una exportación INT8 que cabe en 4MB de memoria.

---

## HITO 3: Configuración de Activaciones y Dataset
**Sesiones:** `6a322df9`, `93fc12e9`

**Q&A Clave:**
- **USER:** ¿Cómo entrenamos si el Maestro usa GELU y el Alumno usa ReLU?
- **ANTIGRAVITY:** Implementamos una lógica condicional en `dpt.py`. El Maestro mantiene su precisión con GELU, mientras que el Alumno aprende usando ReLU para ser eficiente en el microcontrolador.
- **USER:** ¿Qué datos usamos?
- **ANTIGRAVITY:** Configuramos **FiftyOne** para descargar y preparar el dataset **Open Images V7**. Esto nos permite tener miles de imágenes variadas para que el modelo "entienda" el mundo real.

---

## HITO 4: Resolución del Colapso (The "Dying ReLU" Fix)
**Sesión:** `2186c175` (Actual)

**Q&A Clave:**
- **USER:** Las imágenes salen negras al entrenar. ¿Qué pasa?
- **ANTIGRAVITY:** Identificamos el "Dying ReLU". Las capas finales estaban cortando todas las predicciones a cero.
- **SOLUCIÓN:** 
  1. Corregimos la inicialización del Maestro (usando `from_pretrained`).
  2. Eliminamos activaciones redundantes al final del Alumno.
  3. Inicializamos el sesgo de salida a `+20.0` para que el modelo empiece "viendo" algo.

---

## HITO 5: Calidad Visual y "Edge Sharpening"
**Sesión Actual**

**Q&A Clave:**
- **USER:** Los mapas de profundidad se ven como acuarelas (borrosos).
- **ANTIGRAVITY:** Implementamos una **Pérdida de Gradiente Multi-escala**. Ahora el modelo compara bordes a distancias de 1, 2 y 4 píxeles simultáneamente.
- **USER:** ¿Mejorará el resultado?
- **ANTIGRAVITY:** Sí. Al subir el peso de los gradientes al 50%, el modelo se centra obsesivamente en definir los contornos de los objetos, algo crítico para la navegación robótica.

---

## HITO 6: Futuro y Despliegue (Raspberry vs ESP32)
**Sesión Actual**

**Q&A Clave:**
- **USER:** ¿Puedo comparar con una Raspberry Pi Zero 2W?
- **ANTIGRAVITY:** ¡Claro! Es la comparativa perfecta para un PFG. Verás que la Pi es mucho más rápida gracias a NEON, mientras que el ESP32 es el rey del bajo consumo (Battery-first).

---
*Este documento es una síntesis de cientos de interacciones centradas en la excelencia técnica para un Proyecto de Fin de Grado de ingeniería.*
