# Adaptive Face Image Quality Checker (MediaPipe)
#
# Novedades frente a la versión básica:
# - Opción --use_face_roi_metrics: calcula brillo/exposición SOLO dentro del recorte de rostro.
# - Modo --adaptive: estima umbrales por percentiles del propio dataset (ej. p20 de nitidez).
# - Modo de decisión "weighted": en vez de AND estricto, usa una puntuación con pesos.
# - Resumen por persona: genera un CSV adicional con % de pass por subcarpeta (persona).
#
# Ejemplos:
#   (1) Relajar umbrales fijos:
#       python quality_adaptive.py --img_dir data/preprocessed/train --min_lap_var 60 --min_face_px 60 --min_face_ratio 0.02 \
#         --min_brightness 40 --max_brightness 220 --max_underexp_frac 0.60 --max_overexp_frac 0.35 --use_face_roi_metrics
#
#   (2) Modo adaptativo por percentiles (nitidez p20, brillo [p10,p95]):
#       python quality_adaptive.py --img_dir data/preprocessed/train --adaptive \
#         --lap_var_pctl 20 --brightness_low_pctl 10 --brightness_high_pctl 95 --use_face_roi_metrics
#
#   (3) Modo "weighted":
#       python quality_adaptive.py --img_dir data/preprocessed/train --decision_mode weighted \
#         --w_detect 0.30 --w_size 0.20 --w_ratio 0.10 --w_sharp 0.25 --w_bright 0.10 --w_expo 0.05 --score_thresh 0.6
#
# Requisitos: mediapipe, opencv-python, (opcional) pandas

import argparse
import csv
import os
import sys
import time
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("ERROR: No se pudo importar mediapipe. Instala con `pip install mediapipe`.") from e


def find_images(root: str, exts=('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')) -> List[str]:
    files = []
    for r, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(exts):
                files.append(os.path.join(r, fn))
    files.sort()
    return files


def clip(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def detect_largest_face_mediapipe(img_bgr: np.ndarray, detector, img_w: int, img_h: int) -> Tuple[bool, Tuple[int, int, int, int]]:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = detector.process(img_rgb)
    if not res.detections:
        return False, (0, 0, 0, 0)
    bboxes = []
    for det in res.detections:
        rbb = det.location_data.relative_bounding_box
        x = int(round(rbb.xmin * img_w))
        y = int(round(rbb.ymin * img_h))
        w = int(round(rbb.width * img_w))
        h = int(round(rbb.height * img_h))
        x = clip(x, 0, img_w - 1)
        y = clip(y, 0, img_h - 1)
        w = clip(w, 0, img_w - x)
        h = clip(h, 0, img_h - y)
        if w > 0 and h > 0:
            bboxes.append((x, y, w, h))
    if not bboxes:
        return False, (0, 0, 0, 0)
    x, y, w, h = max(bboxes, key=lambda b: b[2]*b[3])
    return True, (x, y, w, h)


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_fractions(gray: np.ndarray, underexp_thresh: int = 15, overexp_thresh: int = 240) -> Tuple[float, float, float]:
    mean_brightness = float(gray.mean())
    total = gray.size
    underexp = float((gray < underexp_thresh).sum()) / total
    overexp = float((gray > overexp_thresh).sum()) / total
    return mean_brightness, underexp, overexp


def compute_metrics(img_path: str,
                    mp_detector,
                    underexp_thresh: int,
                    overexp_thresh: int,
                    use_face_roi_metrics: bool) -> Dict[str, Any]:
    img = cv2.imread(img_path)
    if img is None:
        return {
            "image_path": img_path,
            "error": "cannot_read",
            "face_detected": 0,
            "face_area_ratio": 0.0,
            "face_min_side": 0,
            "sharpness_lap_var": 0.0,
            "mean_brightness": 0.0,
            "underexp_frac": 0.0,
            "overexp_frac": 0.0,
            "width": 0,
            "height": 0,
            "face_x": 0, "face_y": 0, "face_w": 0, "face_h": 0
        }

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_found, (x, y, fw, fh) = detect_largest_face_mediapipe(img, mp_detector, w, h)
    face_area_ratio = (fw * fh) / float(w * h) if face_found else 0.0
    face_min_side = int(min(fw, fh)) if face_found else 0

    # Región para métricas de brillo/exposición
    roi = gray[y:y+fh, x:x+fw] if (use_face_roi_metrics and face_found) else gray
    sharpness = variance_of_laplacian(roi if roi.size > 0 else gray)
    mean_brightness, underexp_frac, overexp_frac = exposure_fractions(roi if roi.size > 0 else gray,
                                                                      underexp_thresh, overexp_thresh)

    return {
        "image_path": img_path,
        "error": "",
        "face_detected": int(face_found),
        "face_area_ratio": face_area_ratio,
        "face_min_side": face_min_side,
        "sharpness_lap_var": sharpness,
        "mean_brightness": mean_brightness,
        "underexp_frac": underexp_frac,
        "overexp_frac": overexp_frac,
        "width": w,
        "height": h,
        "face_x": x, "face_y": y, "face_w": fw, "face_h": fh
    }


def hard_pass(row: Dict[str, Any],
              min_face_px: int,
              min_face_ratio: float,
              min_lap_var: float,
              min_brightness: float,
              max_brightness: float,
              max_underexp_frac: float,
              max_overexp_frac: float) -> bool:
    if row["error"]:
        return False
    if row["face_detected"] != 1:
        return False
    if row["face_min_side"] < min_face_px:
        return False
    if row["face_area_ratio"] < min_face_ratio:
        return False
    if row["sharpness_lap_var"] < min_lap_var:
        return False
    if not (min_brightness <= row["mean_brightness"] <= max_brightness):
        return False
    if row["underexp_frac"] > max_underexp_frac:
        return False
    if row["overexp_frac"] > max_overexp_frac:
        return False
    return True


def weighted_score(row: Dict[str, Any],
                   w_detect: float, w_size: float, w_ratio: float, w_sharp: float, w_bright: float, w_expo: float,
                   min_face_px: int, min_face_ratio: float, min_lap_var: float,
                   min_brightness: float, max_brightness: float,
                   max_underexp_frac: float, max_overexp_frac: float) -> float:
    # Normalizaciones simples [0,1] por criterio
    s_detect = 1.0 if row["face_detected"] == 1 else 0.0
    s_size = min(1.0, row["face_min_side"] / max(1e-6, float(min_face_px))) if min_face_px > 0 else 1.0
    s_ratio = min(1.0, row["face_area_ratio"] / max(1e-6, float(min_face_ratio))) if min_face_ratio > 0 else 1.0
    s_sharp = min(1.0, row["sharpness_lap_var"] / max(1e-6, float(min_lap_var))) if min_lap_var > 0 else 1.0

    # Brillo: 1 si está dentro [min, max], decrece linealmente al salir del rango (acotado a 0)
    mb = row["mean_brightness"]
    if mb < min_brightness:
        s_bright = max(0.0, 1.0 - (min_brightness - mb) / max(1.0, min_brightness))
    elif mb > max_brightness:
        s_bright = max(0.0, 1.0 - (mb - max_brightness) / max(1.0, 255 - max_brightness))
    else:
        s_bright = 1.0

    # Exposición: 1 si ambas fracciones por debajo de máximos, penalización lineal si exceden
    ue = row["underexp_frac"]
    oe = row["overexp_frac"]
    pen_ue = max(0.0, (ue - max_underexp_frac) / max(1e-6, 1.0 - max_underexp_frac))
    pen_oe = max(0.0, (oe - max_overexp_frac) / max(1e-6, 1.0 - max_overexp_frac))
    s_expo = max(0.0, 1.0 - 0.5 * (pen_ue + pen_oe))

    score = (w_detect * s_detect + w_size * s_size + w_ratio * s_ratio +
             w_sharp * s_sharp + w_bright * s_bright + w_expo * s_expo)
    return float(score)


def main():
    ap = argparse.ArgumentParser(description="Adaptive/Weighted Face Image Quality Checker (MediaPipe)")
    ap.add_argument("--img_dir", type=str, required=True, help="Directorio raíz con imágenes (recursivo)")
    ap.add_argument("--csv_out", type=str, default="image_quality_report.csv", help="CSV de salida por imagen")
    ap.add_argument("--csv_summary_out", type=str, default="image_quality_summary_per_person.csv", help="CSV resumen por persona")
    # Umbrales fijos (modo hard/weighted)
    ap.add_argument("--min_face_px", type=int, default=100)
    ap.add_argument("--min_face_ratio", type=float, default=0.05)
    ap.add_argument("--min_lap_var", type=float, default=120.0)
    ap.add_argument("--min_brightness", type=float, default=60.0)
    ap.add_argument("--max_brightness", type=float, default=200.0)
    ap.add_argument("--underexp_thresh", type=int, default=15)
    ap.add_argument("--overexp_thresh", type=int, default=240)
    ap.add_argument("--max_underexp_frac", type=float, default=0.40)
    ap.add_argument("--max_overexp_frac", type=float, default=0.25)
    # ROI y modos
    ap.add_argument("--use_face_roi_metrics", action="store_true", help="Calcular nitidez/brillo/exposición solo en el ROI del rostro")
    ap.add_argument("--decision_mode", choices=["hard", "weighted"], default="hard")
    ap.add_argument("--score_thresh", type=float, default=0.6, help="Umbral de aceptación para modo weighted")
    ap.add_argument("--w_detect", type=float, default=0.30)
    ap.add_argument("--w_size", type=float, default=0.20)
    ap.add_argument("--w_ratio", type=float, default=0.10)
    ap.add_argument("--w_sharp", type=float, default=0.25)
    ap.add_argument("--w_bright", type=float, default=0.10)
    ap.add_argument("--w_expo", type=float, default=0.05)
    # Adaptativo por percentiles (sobre el propio dataset)
    ap.add_argument("--adaptive", action="store_true", help="Estimar umbrales a partir de percentiles del dataset")
    ap.add_argument("--lap_var_pctl", type=float, default=20.0, help="Percentil para nitidez (ej. p20)")
    ap.add_argument("--brightness_low_pctl", type=float, default=10.0, help="Percentil bajo de brillo medio")
    ap.add_argument("--brightness_high_pctl", type=float, default=95.0, help="Percentil alto de brillo medio")
    # MediaPipe
    ap.add_argument("--mp_model_selection", type=int, default=1, choices=[0, 1])
    ap.add_argument("--mp_min_conf", type=float, default=0.5)
    args = ap.parse_args()

    if not os.path.isdir(args.img_dir):
        print(f"[ERROR] Directorio no válido: {args.img_dir}", file=sys.stderr)
        sys.exit(2)

    # Detector
    FaceDetection = mp.solutions.face_detection.FaceDetection
    with FaceDetection(model_selection=args.mp_model_selection, min_detection_confidence=args.mp_min_conf) as detector:
        img_paths = find_images(args.img_dir)
        if not img_paths:
            print(f"[WARN] No se encontraron imágenes en: {args.img_dir}", file=sys.stderr)

        rows: List[Dict[str, Any]] = []
        t0 = time.time()
        # Primera pasada: métricas crudas (para adaptativo si aplica)
        for i, p in enumerate(img_paths, 1):
            m = compute_metrics(p, detector, args.underexp_thresh, args.overexp_thresh, args.use_face_roi_metrics)
            rows.append(m)
            if i % 100 == 0:
                elapsed = time.time() - t0
                print(f"[Info] (fase 1) Procesadas {i}/{len(img_paths)} imágenes en {elapsed:.1f}s")

    # Estimar umbrales adaptativos si procede (solo si hay datos)
    if args.adaptive and len(rows) > 0:
        # Usamos solo imágenes con rostro detectado para percentiles de nitidez/brillo
        lap_vals = np.array([r["sharpness_lap_var"] for r in rows if r["face_detected"] == 1], dtype=float)
        bri_vals = np.array([r["mean_brightness"] for r in rows if r["face_detected"] == 1], dtype=float)
        if lap_vals.size > 0:
            args.min_lap_var = float(np.percentile(lap_vals, args.lap_var_pctl))
        if bri_vals.size > 0:
            args.min_brightness = float(np.percentile(bri_vals, args.brightness_low_pctl))
            args.max_brightness = float(np.percentile(bri_vals, args.brightness_high_pctl))
        print(f"[Adaptive] min_lap_var={args.min_lap_var:.2f}, brightness∈[{args.min_brightness:.2f}, {args.max_brightness:.2f}]")

    # Segunda pasada: decisión final y escritura
    total = len(rows)
    passed = 0
    for r in rows:
        if args.decision_mode == "hard":
            ok = hard_pass(r,
                           args.min_face_px, args.min_face_ratio, args.min_lap_var,
                           args.min_brightness, args.max_brightness,
                           args.max_underexp_frac, args.max_overexp_frac)
            r["pass"] = int(ok)
        else:
            score = weighted_score(r,
                                   args.w_detect, args.w_size, args.w_ratio, args.w_sharp, args.w_bright, args.w_expo,
                                   args.min_face_px, args.min_face_ratio, args.min_lap_var,
                                   args.min_brightness, args.max_brightness,
                                   args.max_underexp_frac, args.max_overexp_frac)
            r["score"] = score
            r["pass"] = int(score >= args.score_thresh)
        passed += r["pass"]

    pct = (passed / total * 100.0) if total > 0 else 0.0

    # CSV por imagen
    fieldnames = [
        "image_path","error","pass","face_detected","face_area_ratio","face_min_side",
        "sharpness_lap_var","mean_brightness","underexp_frac","overexp_frac",
        "width","height","face_x","face_y","face_w","face_h"
    ]
    if args.decision_mode == "weighted":
        fieldnames.append("score")

    if pd is not None:
        try:
            df = pd.DataFrame(rows, columns=fieldnames)
            df.to_csv(args.csv_out, index=False, encoding="utf-8")
        except Exception:
            with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    else:
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Resumen por persona (subcarpeta inmediata dentro de img_dir)
    def person_from_path(pth: str, root: str) -> str:
        rel = os.path.relpath(pth, root)
        parts = rel.split(os.sep)
        return parts[0] if len(parts) > 0 else ""

    summary: Dict[str, Dict[str, int]] = {}
    for r in rows:
        person = person_from_path(r["image_path"], args.img_dir)
        if person not in summary:
            summary[person] = {"total": 0, "passed": 0}
        summary[person]["total"] += 1
        summary[person]["passed"] += int(r["pass"])

    sum_rows = []
    for person, agg in summary.items():
        if agg["total"] > 0:
            p = 100.0 * agg["passed"] / agg["total"]
        else:
            p = 0.0
        sum_rows.append({"person": person, "total": agg["total"], "passed": agg["passed"], "pct": p})
    sum_rows.sort(key=lambda x: x["person"])

    if pd is not None:
        try:
            df2 = pd.DataFrame(sum_rows, columns=["person","total","passed","pct"])
            df2.to_csv(args.csv_summary_out, index=False, encoding="utf-8")
        except Exception:
            with open(args.csv_summary_out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["person","total","passed","pct"])
                writer.writeheader()
                writer.writerows(sum_rows)
    else:
        with open(args.csv_summary_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["person","total","passed","pct"])
            writer.writeheader()
            writer.writerows(sum_rows)

    print("\n=== RESUMEN GLOBAL ===")
    print(f"Imágenes analizadas: {total}")
    print(f"Imágenes que cumplen calidad: {passed}")
    print(f"Porcentaje que cumplen calidad: {pct:.2f}%")
    print(f"CSV por imagen: {os.path.abspath(args.csv_out)}")
    print(f"CSV por persona: {os.path.abspath(args.csv_summary_out)}")

    print("\nSugerencia para tu paper:")
    mode_txt = "modo hard" if args.decision_mode == "hard" else f"modo weighted (umbral {args.score_thresh})"
    roi_txt = "ROI facial" if args.use_face_roi_metrics else "imagen completa"
    print(f'Se evaluó la calidad de {total} imágenes con {mode_txt} y métricas en {roi_txt}; el {pct:.2f}% cumplió los criterios.')

if __name__ == "__main__":
    main()

"""python quality_check_faces_mediapipe_adaptive.py --img_dir ../data/preprocessed/test \
  --decision_mode weighted --score_thresh 0.6 \
  --w_detect 0.30 --w_size 0.20 --w_ratio 0.10 --w_sharp 0.25 --w_bright 0.10 --w_expo 0.05 \
  --min_lap_var 60 --min_face_px 60 --min_face_ratio 0.02 \
  --min_brightness 40 --max_brightness 220 \
  --max_underexp_frac 0.60 --max_overexp_frac 0.35 \
  --use_face_roi_metrics """