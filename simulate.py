"""
simulate.py
-----------
Simulación del sistema "Death Ray" usando Pygame y YOLO.

Muestra una ventana con insectos moviéndose, ejecuta inferencia YOLO
y controla un láser simulado (y opcionalmente un Arduino con servos).

Uso:
    python simulate.py

Teclas:
    1 = apuntar a mariposas
    2 = apuntar a mariquitas
    3 = apuntar a cucarachas (por defecto)
    Q = salir
"""

import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

SCREEN_W = 640
SCREEN_H = 480
TARGET_FPS = 30
INFERENCE_EVERY = 5  # fotogramas entre inferencias

CLASS_NAMES = ["mariposa", "mariquita", "cucaracha", "cruz", "laser"]
CLASS_COLORS = {
    0: (255, 200, 0),    # mariposa - amarillo
    1: (255, 50, 50),    # mariquita - rojo
    2: (50, 200, 50),    # cucaracha - verde
    3: (100, 100, 255),  # cruz - azul
    4: (255, 255, 255),  # laser - blanco
}

LASER_DWELL_TIME = 2.0  # segundos en cada objetivo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    return {
        "serial_port": os.getenv("serial_port", ""),
        "serial_baud": int(os.getenv("serial_baud", 9600)),
    }


def map_pixel_to_servo(
    px: float,
    py: float,
    screen_w: int = SCREEN_W,
    screen_h: int = SCREEN_H,
) -> tuple[int, int]:
    """
    Mapea coordenadas de píxel a ángulos de servo (0-180).

    px, py: coordenadas de píxel en la pantalla.
    Retorna (angle_x, angle_y) en grados (0-180 cada uno).
    """
    angle_x = int(np.clip(px / screen_w * 180, 0, 180))
    angle_y = int(np.clip(py / screen_h * 180, 0, 180))
    return angle_x, angle_y


def try_connect_arduino(port: str, baud: int):
    """Intenta conectarse al Arduino. Retorna el objeto serial o None."""
    if not port:
        return None
    try:
        import serial  # type: ignore
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # Esperar al reset del Arduino
        logger.info("Arduino conectado en %s @ %d baud", port, baud)
        return ser
    except Exception as exc:
        logger.warning("No se pudo conectar al Arduino: %s", exc)
        return None


def send_servo_command(ser, angle_x: int, angle_y: int) -> None:
    """Envía un comando de ángulos al Arduino."""
    if ser is None:
        return
    try:
        cmd = f"{angle_x},{angle_y}\n".encode()
        ser.write(cmd)
    except Exception as exc:
        logger.warning("Error enviando comando al Arduino: %s", exc)


# ---------------------------------------------------------------------------
# Insecto simulado
# ---------------------------------------------------------------------------

class Insect:
    """Representa un insecto que se mueve en la pantalla."""

    SIZE = 60

    def __init__(self, class_id: int, surface, screen_w: int, screen_h: int):
        self.class_id = class_id
        self.surface = surface
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.x = float(random.randint(0, screen_w - self.SIZE))
        self.y = float(random.randint(0, screen_h - self.SIZE))
        speed = random.uniform(1.0, 3.5)
        angle = random.uniform(0, 2 * np.pi)
        self.vx = speed * np.cos(angle)
        self.vy = speed * np.sin(angle)
        self.born = time.time()
        self.lifespan = random.uniform(5.0, 15.0)

    @property
    def alive(self) -> bool:
        return time.time() - self.born < self.lifespan

    @property
    def rect(self):
        import pygame
        return pygame.Rect(int(self.x), int(self.y), self.SIZE, self.SIZE)

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.SIZE / 2, self.y + self.SIZE / 2

    def update(self) -> None:
        self.x += self.vx
        self.y += self.vy
        if self.x < 0 or self.x > self.screen_w - self.SIZE:
            self.vx *= -1
            self.x = max(0, min(self.x, self.screen_w - self.SIZE))
        if self.y < 0 or self.y > self.screen_h - self.SIZE:
            self.vy *= -1
            self.y = max(0, min(self.y, self.screen_h - self.SIZE))

    def draw(self, screen) -> None:
        import pygame
        if self.surface:
            screen.blit(self.surface, (int(self.x), int(self.y)))
        else:
            color = CLASS_COLORS.get(self.class_id, (200, 200, 200))
            pygame.draw.ellipse(screen, color, self.rect)
            # Etiqueta
            font = pygame.font.SysFont(None, 18)
            label = font.render(CLASS_NAMES[self.class_id][:3], True, (255, 255, 255))
            screen.blit(label, (int(self.x), int(self.y)))


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        import pygame
    except ImportError:
        logger.error("pygame no está instalado. Ejecuta: pip install pygame")
        sys.exit(1)

    cfg = load_config()

    # Cargar modelo YOLO
    model = None
    model_path = Path(__file__).parent / "models" / "best.pt"
    if model_path.exists():
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            logger.info("Modelo YOLO cargado desde %s", model_path)
        except Exception as exc:
            logger.warning("No se pudo cargar el modelo YOLO: %s", exc)
    else:
        logger.warning(
            "No se encontró models/best.pt. Ejecuta train_model.py primero. "
            "La simulación funcionará sin inferencia."
        )

    # Conectar Arduino (opcional)
    arduino = try_connect_arduino(cfg["serial_port"], cfg["serial_baud"])

    # Inicializar Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Death Ray — Simulación")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)
    font_small = pygame.font.SysFont(None, 20)

    # Cargar assets de insectos (opcional)
    assets_dir = Path(__file__).parent / "assets"
    insect_surfaces = {}
    insect_filenames = ["mariposa.png", "mariquita.png", "cucaracha.png"]
    for cls_id, fname in enumerate(insect_filenames):
        path = assets_dir / fname
        if path.exists():
            try:
                surf = pygame.image.load(str(path)).convert_alpha()
                surf = pygame.transform.scale(surf, (Insect.SIZE, Insect.SIZE))
                insect_surfaces[cls_id] = surf
            except Exception:
                insect_surfaces[cls_id] = None
        else:
            insect_surfaces[cls_id] = None

    # Estado de la simulación
    insects: list[Insect] = []
    target_class = 2  # cucaracha por defecto
    frame_count = 0
    detections: list[tuple] = []  # (class_id, x1, y1, x2, y2, conf)

    # Estado del láser
    laser_pos: tuple[float, float] | None = None
    laser_targets: list[tuple[float, float]] = []
    laser_target_idx = 0
    laser_arrived_at: float | None = None

    last_spawn = time.time()
    spawn_interval = random.uniform(1.0, 3.0)

    running = True
    while running:
        dt = clock.tick(TARGET_FPS) / 1000.0

        # --- Eventos ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_1:
                    target_class = 0
                    logger.info("Objetivo: mariposas")
                elif event.key == pygame.K_2:
                    target_class = 1
                    logger.info("Objetivo: mariquitas")
                elif event.key == pygame.K_3:
                    target_class = 2
                    logger.info("Objetivo: cucarachas")

        # --- Spawn de insectos ---
        now = time.time()
        if now - last_spawn > spawn_interval:
            cls_id = random.randint(0, 2)
            surf = insect_surfaces.get(cls_id)
            insects.append(Insect(cls_id, surf, SCREEN_W, SCREEN_H))
            last_spawn = now
            spawn_interval = random.uniform(0.5, 2.5)

        # Limitar número de insectos a 15
        if len(insects) > 15:
            insects.pop(0)

        # Eliminar insectos que hayan caducado
        insects = [ins for ins in insects if ins.alive]

        # --- Actualizar posiciones ---
        for ins in insects:
            ins.update()

        # --- Inferencia YOLO (cada N fotogramas) ---
        if model is not None and frame_count % INFERENCE_EVERY == 0:
            # Capturar pantalla como array numpy
            frame_surface = screen.copy()
            frame_array = pygame.surfarray.array3d(frame_surface)
            # surfarray devuelve (W, H, 3); convertir a (H, W, 3)
            frame_array = np.transpose(frame_array, (1, 0, 2))
            try:
                results = model.predict(frame_array, verbose=False)
                detections = []
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        detections.append((cls, x1, y1, x2, y2, conf))
            except Exception as exc:
                logger.debug("Error en inferencia: %s", exc)

        # --- Actualizar objetivos del láser ---
        laser_targets = [
            ((d[1] + d[3]) / 2, (d[2] + d[4]) / 2)
            for d in detections
            if d[0] == target_class
        ]
        # Si no hay detecciones, usar posiciones reales de insectos del target_class
        if not laser_targets:
            laser_targets = [
                ins.center for ins in insects if ins.class_id == target_class
            ]

        if laser_targets:
            if laser_target_idx >= len(laser_targets):
                laser_target_idx = 0

            target_pos = laser_targets[laser_target_idx]

            if laser_pos is None:
                laser_pos = target_pos
                laser_arrived_at = None
            else:
                # Mover láser suavemente hacia el objetivo
                lx, ly = laser_pos
                tx, ty = target_pos
                dist = np.sqrt((tx - lx) ** 2 + (ty - ly) ** 2)
                speed_px = 200 * dt  # píxeles por segundo
                if dist < speed_px:
                    laser_pos = target_pos
                    if laser_arrived_at is None:
                        laser_arrived_at = now
                    elif now - laser_arrived_at >= LASER_DWELL_TIME:
                        # Pasar al siguiente objetivo
                        laser_target_idx = (laser_target_idx + 1) % len(laser_targets)
                        laser_arrived_at = None
                else:
                    ratio = speed_px / dist
                    laser_pos = (lx + (tx - lx) * ratio, ly + (ty - ly) * ratio)
                    laser_arrived_at = None

            # Enviar ángulos al Arduino
            if laser_pos:
                ax, ay = map_pixel_to_servo(laser_pos[0], laser_pos[1])
                send_servo_command(arduino, ax, ay)
        else:
            laser_pos = None
            laser_target_idx = 0

        # --- Dibujar ---
        screen.fill((40, 40, 40))

        # Insectos
        for ins in insects:
            ins.draw(screen)

        # Bounding boxes de detecciones YOLO
        for cls_id, x1, y1, x2, y2, conf in detections:
            color = CLASS_COLORS.get(cls_id, (200, 200, 200))
            pygame.draw.rect(screen, color, (x1, y1, x2 - x1, y2 - y1), 2)
            label = font_small.render(
                f"{CLASS_NAMES[cls_id]} {conf:.2f}", True, color
            )
            screen.blit(label, (x1, max(0, y1 - 16)))

        # Punto láser
        if laser_pos is not None:
            lx, ly = int(laser_pos[0]), int(laser_pos[1])
            pygame.draw.circle(screen, (0, 255, 0), (lx, ly), 8)
            pygame.draw.circle(screen, (0, 180, 0), (lx, ly), 4)

        # HUD
        fps = clock.get_fps()
        fps_text = font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))

        target_text = font_small.render(
            f"Objetivo: {CLASS_NAMES[target_class]}  [1=mariposa 2=mariquita 3=cucaracha Q=salir]",
            True, (200, 200, 200),
        )
        screen.blit(target_text, (10, SCREEN_H - 25))

        if model is None:
            no_model = font_small.render("Sin modelo YOLO — modo demo", True, (255, 150, 50))
            screen.blit(no_model, (10, 40))

        pygame.display.flip()
        frame_count += 1

    if arduino:
        arduino.close()
    pygame.quit()
    logger.info("Simulación finalizada.")


if __name__ == "__main__":
    main()
