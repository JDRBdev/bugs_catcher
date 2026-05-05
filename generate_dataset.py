"""
generate_dataset.py
--------------------
Generador de dataset sintético para el proyecto "Death Ray".

Genera imágenes de escenas con insectos, marcadores de esquina y puntos láser,
junto con etiquetas en formato YOLO.

Uso:
    python generate_dataset.py [--dry-run]

    --dry-run: genera y muestra 1 imagen sin guardarla en disco.
"""

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageEnhance

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clase IDs
# ---------------------------------------------------------------------------
CLASS_NAMES = ["mariposa", "mariquita", "cucaracha", "cruz", "laser"]
CLASS_MARIPOSA = 0
CLASS_MARIQUITA = 1
CLASS_CUCARACHA = 2
CLASS_CRUZ = 3
CLASS_LASER = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Carga la configuración desde el archivo .env."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        logger.error("No se encontró el archivo .env en %s", env_path)
        sys.exit(1)
    load_dotenv(env_path)

    raw_clases = os.getenv("img_clases", "[mariposa.png, mariquita.png, cucaracha.png]")
    # Parsear lista estilo "[a.png, b.png]" o "a.png, b.png"
    raw_clases = raw_clases.strip().strip("[]")
    clases = [s.strip().strip("\"'") for s in raw_clases.split(",") if s.strip()]

    return {
        "num_clases": int(os.getenv("num_clases", 3)),
        "max_width": int(os.getenv("max_width", 640)),
        "max_height": int(os.getenv("max_height", 480)),
        "img_limite": os.getenv("img_limite", "cruz.png"),
        "img_clases": clases,
        "num_imagenes": int(os.getenv("num_imagenes", 1000)),
        "max_objetos": int(os.getenv("max_objetos", 20)),
        "min_objetos": int(os.getenv("min_objetos", 10)),
    }


def load_asset(assets_dir: Path, filename: str) -> Image.Image | None:
    """Carga una imagen de assets con manejo de errores."""
    path = assets_dir / filename
    if not path.exists():
        logger.warning("Asset no encontrado: %s", path)
        return None
    try:
        img = Image.open(path).convert("RGBA")
        return img
    except Exception as exc:
        logger.warning("No se pudo cargar %s: %s", path, exc)
        return None


def paste_rgba(background: Image.Image, overlay: Image.Image, x: int, y: int) -> None:
    """Pega una imagen RGBA sobre un fondo PIL en la posición (x, y)."""
    if overlay.mode != "RGBA":
        overlay = overlay.convert("RGBA")
    background.paste(overlay, (x, y), overlay)


def apply_transforms(img: Image.Image, base_size: int = 80) -> Image.Image:
    """Aplica transformaciones aleatorias a una imagen de insecto."""
    # Escala aleatoria: 0.5x a 1.5x del tamaño base
    scale = random.uniform(0.5, 1.5)
    new_size = max(1, int(base_size * scale))
    img = img.resize((new_size, new_size), Image.LANCZOS)

    # Rotación aleatoria 0-360
    angle = random.uniform(0, 360)
    img = img.rotate(angle, expand=True, resample=Image.BICUBIC)

    # Flip horizontal/vertical aleatorio
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Variación de brillo ligera
    if random.random() < 0.5:
        factor = random.uniform(0.8, 1.2)
        # Solo afecta los canales RGB, no el alfa
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
        rgb = ImageEnhance.Brightness(rgb).enhance(factor)
        r2, g2, b2 = rgb.split()
        img = Image.merge("RGBA", (r2, g2, b2, a))

    return img


def create_background(width: int, height: int) -> Image.Image:
    """Crea un fondo blanco/gris aleatorio."""
    shade = random.randint(220, 255)
    bg = Image.new("RGBA", (width, height), (shade, shade, shade, 255))
    return bg


def generate_image(
    cfg: dict,
    insect_assets: list[Image.Image | None],
    cruz_asset: Image.Image | None,
    laser_asset: Image.Image | None,
    width: int,
    height: int,
) -> tuple[Image.Image, list[tuple[int, float, float, float, float]]]:
    """
    Genera una imagen sintética y sus etiquetas YOLO.

    Devuelve (imagen_PIL, lista_de_etiquetas).
    Cada etiqueta: (class_id, x_center_norm, y_center_norm, w_norm, h_norm)
    """
    bg = create_background(width, height)
    labels: list[tuple[int, float, float, float, float]] = []

    corner_size = 40

    # --- Marcadores de esquina (cruz.png) ---
    if cruz_asset is not None:
        cruz_img = cruz_asset.resize((corner_size, corner_size), Image.LANCZOS)
        corners = [
            (0, 0),
            (width - corner_size, 0),
            (0, height - corner_size),
            (width - corner_size, height - corner_size),
        ]
        for cx, cy in corners:
            paste_rgba(bg, cruz_img, cx, cy)
            xc = (cx + corner_size / 2) / width
            yc = (cy + corner_size / 2) / height
            labels.append((CLASS_CRUZ, xc, yc, corner_size / width, corner_size / height))

    # --- Insectos ---
    num_objetos = random.randint(cfg["min_objetos"], cfg["max_objetos"])
    max_objetos = cfg["max_objetos"]

    # Grid imaginario para evitar solapamientos
    cols = int(np.ceil(np.sqrt(max_objetos)))
    rows = int(np.ceil(max_objetos / cols))
    cell_w = width // cols
    cell_h = height // rows

    # Elegir celdas aleatorias sin repetición
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(all_cells)
    chosen_cells = all_cells[:num_objetos]

    # Solo usamos los assets de insectos que estén disponibles
    valid_insects = [
        (i, asset) for i, asset in enumerate(insect_assets) if asset is not None
    ]

    if valid_insects:
        for row, col in chosen_cells:
            class_id, base_asset = random.choice(valid_insects)
            transformed = apply_transforms(base_asset)

            # Centro de la celda con algo de aleatoriedad
            cell_cx = col * cell_w + cell_w // 2 + random.randint(-cell_w // 4, cell_w // 4)
            cell_cy = row * cell_h + cell_h // 2 + random.randint(-cell_h // 4, cell_h // 4)

            tw, th = transformed.size
            px = cell_cx - tw // 2
            py = cell_cy - th // 2

            # Asegurar que no salga de los bordes
            px = max(0, min(px, width - tw))
            py = max(0, min(py, height - th))

            paste_rgba(bg, transformed, px, py)

            xc = (px + tw / 2) / width
            yc = (py + th / 2) / height
            labels.append((class_id, xc, yc, tw / width, th / height))

    # --- Puntos láser (ocasionalmente 1-3) ---
    if laser_asset is not None and random.random() < 0.4:
        num_lasers = random.randint(1, 3)
        laser_size = 20
        laser_img = laser_asset.resize((laser_size, laser_size), Image.LANCZOS)
        for _ in range(num_lasers):
            lx = random.randint(0, width - laser_size)
            ly = random.randint(0, height - laser_size)
            paste_rgba(bg, laser_img, lx, ly)
            xc = (lx + laser_size / 2) / width
            yc = (ly + laser_size / 2) / height
            labels.append((CLASS_LASER, xc, yc, laser_size / width, laser_size / height))

    return bg.convert("RGB"), labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generador de dataset para Death Ray")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Genera 1 imagen de prueba y la muestra sin guardar en disco",
    )
    args = parser.parse_args()

    cfg = load_config()
    logger.info("Configuración cargada: %s", cfg)

    assets_dir = Path(__file__).parent / "assets"
    if not assets_dir.exists():
        logger.error("No se encontró el directorio de assets: %s", assets_dir)
        sys.exit(1)

    # Cargar assets
    cruz_asset = load_asset(assets_dir, cfg["img_limite"])
    laser_asset = load_asset(assets_dir, "laser.png")

    insect_assets: list[Image.Image | None] = []
    for class_filename in cfg["img_clases"]:
        insect_assets.append(load_asset(assets_dir, class_filename))

    width = cfg["max_width"]
    height = cfg["max_height"]
    num_imagenes = cfg["num_imagenes"]

    # --- Dry run ---
    if args.dry_run:
        logger.info("Modo dry-run: generando 1 imagen de prueba...")
        img, labels = generate_image(cfg, insect_assets, cruz_asset, laser_asset, width, height)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Dibujar bounding boxes para visualización
        for cls_id, xc, yc, w, h in labels:
            x1 = int((xc - w / 2) * width)
            y1 = int((yc - h / 2) * height)
            x2 = int((xc + w / 2) * width)
            y2 = int((yc + h / 2) * height)
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)][cls_id]
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img_cv,
                CLASS_NAMES[cls_id],
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        cv2.imshow("Dry Run Preview", img_cv)
        logger.info("Presiona cualquier tecla para cerrar la ventana de previsualización.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # --- Generación completa ---
    dataset_dir = Path(__file__).parent / "dataset"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generando %d imágenes...", num_imagenes)

    for i in range(1, num_imagenes + 1):
        img, labels = generate_image(cfg, insect_assets, cruz_asset, laser_asset, width, height)

        # Guardar imagen
        img_path = images_dir / f"img{i}.jpg"
        img.save(str(img_path), "JPEG", quality=95)

        # Guardar etiquetas
        label_path = labels_dir / f"img{i}.txt"
        with open(label_path, "w") as f:
            for cls_id, xc, yc, w, h in labels:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        if i % 100 == 0:
            logger.info("Progreso: %d/%d imágenes generadas", i, num_imagenes)

    logger.info("Todas las imágenes generadas.")

    # --- Split train/val 80/20 ---
    all_indices = list(range(1, num_imagenes + 1))
    random.shuffle(all_indices)
    split = int(len(all_indices) * 0.8)
    train_indices = all_indices[:split]
    val_indices = all_indices[split:]

    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    for sub in ["images", "labels"]:
        (train_dir / sub).mkdir(parents=True, exist_ok=True)
        (val_dir / sub).mkdir(parents=True, exist_ok=True)

    for idx in train_indices:
        shutil.copy(images_dir / f"img{idx}.jpg", train_dir / "images" / f"img{idx}.jpg")
        shutil.copy(labels_dir / f"img{idx}.txt", train_dir / "labels" / f"img{idx}.txt")

    for idx in val_indices:
        shutil.copy(images_dir / f"img{idx}.jpg", val_dir / "images" / f"img{idx}.jpg")
        shutil.copy(labels_dir / f"img{idx}.txt", val_dir / "labels" / f"img{idx}.txt")

    logger.info(
        "Split: %d train, %d val", len(train_indices), len(val_indices)
    )

    # --- Generar dataset.yaml ---
    yaml_path = dataset_dir / "dataset.yaml"
    abs_train = str((train_dir / "images").resolve())
    abs_val = str((val_dir / "images").resolve())

    with open(yaml_path, "w") as f:
        f.write(f"train: {abs_train}\n")
        f.write(f"val: {abs_val}\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    logger.info("dataset.yaml guardado en %s", yaml_path)
    logger.info("¡Dataset generado con éxito!")


if __name__ == "__main__":
    main()
