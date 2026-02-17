import cv2
import numpy as np
import math

# ------------------------------
# 1) Parámetros a AJUSTAR
# ------------------------------

VIDEO_PATH = "peixos2.mp4"  # ruta al vídeo de los peces

# Medidas reales del área visible de la imagen (en metros)
# Por ejemplo, si sabes que el ancho visible del acuario son 1.2 m
SCENE_WIDTH_M = 1.2   # ancho real correspondiente al frame
SCENE_HEIGHT_M = 0.7  # alto real correspondiente al frame (opcional, pero útil)

# ------------------------------
# 2) Funciones auxiliares
# ------------------------------

def center_of_rect(rect):
    """Devuelve el centro de un rectángulo (x, y, w, h)."""
    x, y, w, h = rect
    cx = x + w / 2.0
    cy = y + h / 2.0
    return (cx, cy)

def draw_arrow(image, p_start, p_end, color=(0, 0, 255), thickness=2):
    """Dibuja una flecha desde p_start hasta p_end."""
    p_start = (int(p_start[0]), int(p_start[1]))
    p_end = (int(p_end[0]), int(p_end[1]))
    cv2.arrowedLine(image, p_start, p_end, color, thickness, tipLength=0.3)

# ------------------------------
# 3) Inicialización del vídeo
# ------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("No se puede abrir el vídeo: {}".format(VIDEO_PATH))

# Leemos primer frame para seleccionar las ROIs
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("No se pudo leer el primer frame del vídeo")

frame_h, frame_w = first_frame.shape[:2]

# Cálculo de metros por píxel (suponemos que todo el ancho del frame corresponde a SCENE_WIDTH_M)
meters_per_pixel_x = SCENE_WIDTH_M / frame_w
meters_per_pixel_y = SCENE_HEIGHT_M / frame_h

# Obtenemos FPS del vídeo para pasar de píxeles/frame a píxeles/segundo
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    # Valor por defecto en caso de fallo; conviene medir o fijar el FPS real
    fps = 30.0

# ------------------------------
# 4) Selección de ROIs (uno por pez)
# ------------------------------

# Puedes seleccionar varias ROIs con selectROIs
rois = cv2.selectROIs("Selecciona peces", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Selecciona peces")

rois = list(rois)
print("ROIs seleccionadas:", rois)

# Cada ROI es (x, y, w, h)
rois = list(rois)

if len(rois) == 0:
    raise RuntimeError("No se ha seleccionado ninguna ROI")

# ------------------------------
# 5) Inicializar estructuras para cada pez
# ------------------------------

fish_data = []  # lista de dicts, uno por pez

hsv_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)

for roi in rois:
    x, y, w, h = roi
    # ROI en HSV
    roi_hsv = hsv_first[y:y + h, x:x + w]

    # Máscara para ignorar píxeles muy oscuros o demasiado claros si hace falta
    mask = cv2.inRange(roi_hsv, (0, 30, 30), (180, 255, 255))

    # Histograma de la ROI en el canal H (o H+S) para CamShift
    roi_hist = cv2.calcHist([roi_hsv], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Window inicial para CamShift
    track_window = (x, y, w, h)

    # Guardamos la info del pez
    fish_data.append({
        "track_window": track_window,
        "roi_hist": roi_hist,
        "prev_center": center_of_rect(track_window),
        "velocity_m_s": (0.0, 0.0)  # vx, vy en m/s
    })

# Criterio de parada de CamShift
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# ------------------------------
# 6) Bucle principal de tracking
# ------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for i, fish in enumerate(fish_data):
        track_window = fish["track_window"]
        roi_hist = fish["roi_hist"]
        prev_center = fish["prev_center"]

        # Backprojection respecto al histograma de la ROI inicial
        back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Aplicar CamShift
        ret_camshift, new_window = cv2.CamShift(back_proj, track_window, term_crit)
        fish["track_window"] = new_window

        # ret_camshift = ((cx, cy), (w, h), angle)
        pts = cv2.boxPoints(ret_camshift)
        #pts = np.int0(pts)
        pts = pts.astype(np.int32)

        # Bounding box aproximado como rectángulo axis-aligned
        xs = pts[:, 0]
        ys = pts[:, 1]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w_box = x_max - x_min
        h_box = y_max - y_min
        rect = (x_min, y_min, w_box, h_box)

        # Centro actual
        curr_center = center_of_rect(rect)

        # Dibujar bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_min + w_box, y_min + h_box), (0, 255, 0), 2)

        # ------------------------------
        # Cálculo de velocidad
        # ------------------------------
        # Desplazamiento en píxeles por frame
        dx_pixels = curr_center[0] - prev_center[0]
        dy_pixels = curr_center[1] - prev_center[1]

        # Píxeles/frame -> píxeles/segundo
        dx_pixels_per_s = dx_pixels * fps
        dy_pixels_per_s = dy_pixels * fps

        # Conversión a metros/segundo (suponiendo proporción uniforme)
        vx_m_s = dx_pixels_per_s * meters_per_pixel_x
        vy_m_s = dy_pixels_per_s * meters_per_pixel_y

        fish["velocity_m_s"] = (vx_m_s, vy_m_s)
        fish["prev_center"] = curr_center

        # Módulo de la velocidad
        speed_m_s = math.sqrt(vx_m_s ** 2 + vy_m_s ** 2)

        # ------------------------------
        # Dibujo del vector de velocidad
        # ------------------------------
        # Escalado visual del vector velocidad (solo para que sea visible)
        scale = 0.5  # ajusta este factor para que la flecha no sea demasiado larga
        end_point = (
            curr_center[0] + vx_m_s * scale,
            curr_center[1] + vy_m_s * scale
        )

        # Flecha
        draw_arrow(frame, curr_center, end_point, color=(0, 0, 255), thickness=2)

        # Punto del centro
        cv2.circle(frame, (int(curr_center[0]), int(curr_center[1])), 3, (255, 0, 0), -1)

        # Texto con velocidad (en m/s) encima de la caja
        text = f"{speed_m_s:.2f} m/s"
        cv2.putText(frame, text, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # También puedes mostrar la posición (x, y) del centro si lo necesitas
        # pos_text = f"({int(curr_center[0])}, {int(curr_center[1])})"
        # cv2.putText(frame, pos_text, (x_min, y_min + h_box + 15),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("Tracking peces - CamShift", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # ESC o q para salir
        break

cap.release()
cv2.destroyAllWindows()
