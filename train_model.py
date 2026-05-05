"""
train_model.py
--------------
Entrenamiento del modelo YOLO para el proyecto "Death Ray".

Fine-tunes YOLOv8n sobre el dataset sintético generado por generate_dataset.py.

Uso:
    python train_model.py
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Carga la configuración desde .env."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        logger.error("No se encontró el archivo .env en %s", env_path)
        sys.exit(1)
    load_dotenv(env_path)
    return {
        "num_clases": int(os.getenv("num_clases", 3)),
    }


def main() -> None:
    cfg = load_config()
    logger.info("Configuración cargada: %s", cfg)

    dataset_yaml = Path(__file__).parent / "dataset" / "dataset.yaml"
    if not dataset_yaml.exists():
        logger.error(
            "No se encontró dataset.yaml. Ejecuta primero generate_dataset.py."
        )
        sys.exit(1)

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error(
            "ultralytics no está instalado. Ejecuta: pip install ultralytics"
        )
        sys.exit(1)

    logger.info("Cargando modelo base YOLOv8n...")
    model = YOLO("yolov8n.pt")

    logger.info("Iniciando entrenamiento...")
    results = model.train(
        data=str(dataset_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        project=str(models_dir),
        name="death_ray",
        exist_ok=True,
    )

    # Copiar mejor modelo al directorio raíz de models/
    best_src = models_dir / "death_ray" / "weights" / "best.pt"
    best_dst = models_dir / "best.pt"
    if best_src.exists():
        import shutil
        shutil.copy(best_src, best_dst)
        logger.info("Mejor modelo guardado en %s", best_dst)
    else:
        logger.warning("No se encontró best.pt en %s", best_src)

    # Métricas finales
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        logger.info("--- Métricas de entrenamiento ---")
        for key, value in metrics.items():
            logger.info("  %s: %s", key, value)
    else:
        logger.info("Entrenamiento completado. Revisa los resultados en %s", models_dir)


if __name__ == "__main__":
    main()
