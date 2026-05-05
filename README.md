# 🎯 Death Ray — Sistema de Apuntado Láser por Visión por Computador

> Proyecto de visión por computador para detectar y rastrear insectos automáticamente
> mediante un modelo YOLOv8, controlando un láser simulado (y opcionalmente físico con
> servomotores vía Arduino).

---

## 1. Descripción del Proyecto

**Death Ray** es un sistema que combina:

- **Generación de dataset sintético**: crea imágenes artificiales con insectos (mariposas,
  mariquitas, cucarachas) sobre fondos variados, con etiquetas YOLO automáticas.
- **Entrenamiento de modelo**: fine-tuning de YOLOv8n sobre el dataset generado.
- **Simulación en tiempo real**: ventana Pygame con insectos moviéndose, inferencia YOLO
  y un láser virtual que persigue automáticamente el tipo de insecto seleccionado.
- **Control de Arduino** (opcional): envío de ángulos a dos servomotores para apuntar
  un láser físico.

---

## 2. Instalación

### Requisitos previos

- Python 3.10 o superior
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/JDRBdev/bugs_catcher.git
cd bugs_catcher

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## 3. Preparar las imágenes de insectos

Coloca en la carpeta `assets/` los archivos PNG con **fondo transparente** (modo RGBA):

| Archivo | Clase | ID |
|---|---|---|
| `mariposa.png` | Mariposa | 0 |
| `mariquita.png` | Mariquita | 1 |
| `cucaracha.png` | Cucaracha | 2 |
| `cruz.png` | Marcador de esquina | 3 |
| `laser.png` | Punto láser | 4 |

> **Consejo**: Puedes crear imágenes PNG con fondo transparente en GIMP, Photoshop,
> Canva, o buscarlas en sitios como [OpenClipart](https://openclipart.org/).

La configuración de nombres de archivo se puede cambiar en `.env`:

```
img_clases=[mariposa.png, mariquita.png, cucaracha.png]
img_limite=cruz.png
```

---

## 4. Generar el Dataset

```bash
# Generación completa (1000 imágenes por defecto, configurable en .env)
python generate_dataset.py

# Previsualización rápida (1 imagen sin guardar)
python generate_dataset.py --dry-run
```

El script genera:
- `dataset/images/img1.jpg` … `imgN.jpg` — imágenes sintéticas
- `dataset/labels/img1.txt` … `imgN.txt` — etiquetas YOLO (clase x_c y_c w h)
- `dataset/train/` y `dataset/val/` — split 80/20 automático
- `dataset/dataset.yaml` — configuración para YOLOv8

### Parámetros configurables (`.env`)

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `num_imagenes` | Número de imágenes a generar | 1000 |
| `max_objetos` | Máximo de insectos por imagen | 20 |
| `min_objetos` | Mínimo de insectos por imagen | 10 |
| `max_width` | Ancho de las imágenes (px) | 640 |
| `max_height` | Alto de las imágenes (px) | 480 |

---

## 5. Entrenar el Modelo

```bash
python train_model.py
```

Parámetros de entrenamiento:
- Modelo base: `yolov8n.pt` (nano, ligero)
- Épocas: 50 (con early stopping a 10)
- Tamaño de imagen: 640×640
- Batch size: 16

El mejor modelo se guarda en `models/best.pt`.

---

## 6. Ejecutar la Simulación

```bash
python simulate.py
```

### Controles del teclado

| Tecla | Acción |
|---|---|
| `1` | Apuntar a mariposas |
| `2` | Apuntar a mariquitas |
| `3` | Apuntar a cucarachas (por defecto) |
| `Q` | Salir |

La simulación funciona **sin el modelo entrenado** (modo demo) mostrando solo los
insectos animados. Con el modelo, ejecuta inferencia YOLO cada 5 fotogramas y mueve
el punto láser verde hacia cada objetivo detectado.

---

## 7. Conectar el Arduino (Opcional)

### Hardware necesario

- Arduino Uno/Nano
- 2 servomotores (pan + tilt)
- Cables de conexión
- Soporte mecánico para montar los servos y el puntero láser

### Cargar el sketch

1. Abre `arduino/servo_control.ino` en el IDE de Arduino.
2. Conecta el servo horizontal al **pin 9** y el vertical al **pin 10**.
3. Sube el sketch a tu Arduino.

### Configurar la conexión serie

Edita `.env` con el puerto serie de tu Arduino:

```
# Linux/macOS
serial_port=/dev/ttyUSB0

# Windows
serial_port=COM3

serial_baud=9600
```

El sistema enviará comandos `"X,Y\n"` con los ángulos (0–180°) para cada servo.

---

## 8. Estructura del Proyecto

```
bugs_catcher/
├── .env                    # Configuración del proyecto
├── requirements.txt        # Dependencias Python con versiones fijadas
├── README.md               # Esta documentación
├── generate_dataset.py     # Generador de dataset sintético
├── train_model.py          # Entrenamiento del modelo YOLOv8
├── simulate.py             # Simulación con Pygame + YOLO
├── arduino/
│   └── servo_control.ino  # Sketch Arduino para control de servos
├── assets/
│   ├── mariposa.png        # Imagen de mariposa (RGBA PNG)
│   ├── mariquita.png       # Imagen de mariquita (RGBA PNG)
│   ├── cucaracha.png       # Imagen de cucaracha (RGBA PNG)
│   ├── cruz.png            # Marcador de esquina (RGBA PNG)
│   └── laser.png           # Punto láser (RGBA PNG)
├── dataset/
│   ├── images/             # Imágenes generadas (jpg)
│   ├── labels/             # Etiquetas YOLO (txt)
│   ├── train/              # Subset de entrenamiento (80%)
│   ├── val/                # Subset de validación (20%)
│   └── dataset.yaml        # Configuración YOLO del dataset
└── models/
    └── best.pt             # Mejor modelo entrenado
```

---

## Licencia

MIT — véase el repositorio para más detalles.

