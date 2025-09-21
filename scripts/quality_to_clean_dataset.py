"""
quality_to_clean_dataset.py

Filtra imágenes por calidad (MediaPipe + métricas) y copia SOLO las que pasan
a un directorio de salida, preservando la estructura de clases (subcarpetas inmediatas).
Opcionalmente: copiar fallidas, recortar a ROI facial, limitar por clase, y generar CSV.

Ejemplo (copiar aprobadas a dataset limpio, con ROI facial y límites relajados):
python quality_to_clean_dataset.py --img_dir data/preprocessed/train \
  --out_dir_clean data/clean/train --use_face_roi_metrics \
  --min_lap_var 60 --min_face_px 60 --min_face_ratio 0.02 \
  --min_brightness 40 --max_brightness 220 \
  --max_underexp_frac 0.60 --max_overexp_frac 0.35

Ejemplo (modo weighted + score 0.55, copiar fallidas para inspección):
python quality_to_clean_dataset.py --img_dir data/preprocessed/train \
  --out_dir_clean data/clean/train --out_dir_failed data/clean_failed/train \
  --decision_mode weighted --score_thresh 0.55 --use_face_roi_metrics

Requisitos: mediapipe, opencv-python, (opcional) pandas
"""

import argparse
import csv
import os
import sys
import time
import shutil
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
            "face_x": 0, "face_y": 0, "face_w": 0, "face_h": 0,
        }

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_found, (x, y, fw, fh) = detect_largest_face_mediapipe(img, mp_detector, w, h)
    face_area_ratio = (fw * fh) / float(w * h) if face_found else 0.0
    face_min_side = int(min(fw, fh)) if face_found else 0

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
        "face_x": x, "face_y": y, "face_w": fw, "face_h": fh,
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
    s_detect = 1.0 if row["face_detected"] == 1 else 0.0
    s_size = min(1.0, row["face_min_side"] / max(1e-6, float(min_face_px))) if min_face_px > 0 else 1.0
    s_ratio = min(1.0, row["face_area_ratio"] / max(1e-6, float(min_face_ratio))) if min_face_ratio > 0 else 1.0
    s_sharp = min(1.0, row["sharpness_lap_var"] / max(1e-6, float(min_lap_var))) if min_lap_var > 0 else 1.0

    mb = row["mean_brightness"]
    if mb < min_brightness:
        s_bright = max(0.0, 1.0 - (min_brightness - mb) / max(1.0, min_brightness))
    elif mb > max_brightness:
        s_bright = max(0.0, 1.0 - (mb - max_brightness) / max(1.0, 255 - max_brightness))
    else:
        s_bright = 1.0

    ue = row["underexp_frac"]
    oe = row["overexp_frac"]
    pen_ue = max(0.0, (ue - max_underexp_frac) / max(1e-6, 1.0 - max_underexp_frac))
    pen_oe = max(0.0, (oe - max_overexp_frac) / max(1e-6, 1.0 - max_overexp_frac))
    s_expo = max(0.0, 1.0 - 0.5 * (pen_ue + pen_oe))

    score = (w_detect * s_detect + w_size * s_size + w_ratio * s_ratio +
             w_sharp * s_sharp + w_bright * s_bright + w_expo * s_expo)
    return float(score)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def person_from_path(img_path: str, root: str) -> str:
    rel = os.path.relpath(img_path, root)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) > 0 else "unknown"


def save_image(src_path: str, dst_path: str, crop: Tuple[int, int, int, int] = None, resize: Tuple[int, int] = None):
    if crop is None and resize is None:
        ensure_dir(os.path.dirname(dst_path))
        shutil.copy2(src_path, dst_path)
        return True

    img = cv2.imread(src_path)
    if img is None:
        return False
    if crop is not None:
        x, y, w, h = crop
        x = max(0, x); y = max(0, y)
        w = max(1, w); h = max(1, h)
        img = img[y:y+h, x:x+w].copy()
    if resize is not None:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
    ensure_dir(os.path.dirname(dst_path))
    ok = cv2.imwrite(dst_path, img)
    return bool(ok)


def main():
    ap = argparse.ArgumentParser(description="Filtra y copia imágenes de calidad a un dataset limpio (MediaPipe).")
    ap.add_argument("--img_dir", type=str, required=True, help="Directorio raíz de entrada (recursivo)")
    ap.add_argument("--out_dir_clean", type=str, required=True, help="Directorio raíz de salida para imágenes aprobadas")
    ap.add_argument("--out_dir_failed", type=str, default="", help="(Opcional) Directorio raíz para imágenes rechazadas")
    ap.add_argument("--csv_out", type=str, default="image_quality_report.csv", help="CSV de métricas por imagen")
    ap.add_argument("--csv_summary_out", type=str, default="image_quality_summary_per_person.csv", help="CSV resumen por persona")
    # Decisión
    ap.add_argument("--decision_mode", choices=["hard", "weighted"], default="hard")
    ap.add_argument("--score_thresh", type=float, default=0.6, help="Umbral para modo weighted")
    ap.add_argument("--w_detect", type=float, default=0.30)
    ap.add_argument("--w_size", type=float, default=0.20)
    ap.add_argument("--w_ratio", type=float, default=0.10)
    ap.add_argument("--w_sharp", type=float, default=0.25)
    ap.add_argument("--w_bright", type=float, default=0.10)
    ap.add_argument("--w_expo", type=float, default=0.05)
    # Umbrales
    ap.add_argument("--min_face_px", type=int, default=100)
    ap.add_argument("--min_face_ratio", type=float, default=0.05)
    ap.add_argument("--min_lap_var", type=float, default=120.0)
    ap.add_argument("--min_brightness", type=float, default=60.0)
    ap.add_argument("--max_brightness", type=float, default=200.0)
    ap.add_argument("--underexp_thresh", type=int, default=15)
    ap.add_argument("--overexp_thresh", type=int, default=240)
    ap.add_argument("--max_underexp_frac", type=float, default=0.40)
    ap.add_argument("--max_overexp_frac", type=float, default=0.25)
    # ROI / salida
    ap.add_argument("--use_face_roi_metrics", action="store_true", help="Calcular métricas en ROI facial")
    ap.add_argument("--save_mode", choices=["original", "crop"], default="original", help="Guardar imagen completa o recorte facial")
    ap.add_argument("--resize_w", type=int, default=0, help="Redimensionar a ancho específico (0 = mantener)")
    ap.add_argument("--resize_h", type=int, default=0, help="Redimensionar a alto específico (0 = mantener)")
    ap.add_argument("--max_per_class", type=int, default=0, help="Límite de imágenes por clase (0 = sin límite)")
    # MediaPipe
    ap.add_argument("--mp_model_selection", type=int, default=1, choices=[0, 1])
    ap.add_argument("--mp_min_conf", type=float, default=0.5)
    args = ap.parse_args()

    if not os.path.isdir(args.img_dir):
        print(f"[ERROR] Directorio no válido: {args.img_dir}", file=sys.stderr)
        sys.exit(2)

    resize_to = None
    if args.resize_w > 0 and args.resize_h > 0:
        resize_to = (args.resize_w, args.resize_h)

    # Detector
    FaceDetection = mp.solutions.face_detection.FaceDetection
    with FaceDetection(model_selection=args.mp_model_selection, min_detection_confidence=args.mp_min_conf) as detector:
        img_paths = find_images(args.img_dir)
        if not img_paths:
            print(f"[WARN] No se encontraron imágenes en: {args.img_dir}", file=sys.stderr)

        rows: List[Dict[str, Any]] = []
        kept_per_class: Dict[str, int] = {}
        t0 = time.time()
        kept_total = 0
        failed_total = 0

        for i, p in enumerate(img_paths, 1):
            m = compute_metrics(p, detector, args.underexp_thresh, args.overexp_thresh, args.use_face_roi_metrics)

            # Decisión
            if args.decision_mode == "hard":
                ok = hard_pass(m,
                               args.min_face_px, args.min_face_ratio, args.min_lap_var,
                               args.min_brightness, args.max_brightness,
                               args.max_underexp_frac, args.max_overexp_frac)
                m["pass"] = int(ok)
            else:
                score = weighted_score(m,
                                       args.w_detect, args.w_size, args.w_ratio, args.w_sharp, args.w_bright, args.w_expo,
                                       args.min_face_px, args.min_face_ratio, args.min_lap_var,
                                       args.min_brightness, args.max_brightness,
                                       args.max_underexp_frac, args.max_overexp_frac)
                m["score"] = score
                m["pass"] = int(score >= args.score_thresh)

            rows.append(m)

            # Dónde guardar según clase
            person = person_from_path(p, args.img_dir)
            if person not in kept_per_class:
                kept_per_class[person] = 0

            # Guardar aprobadas (respeta límite por clase)
            if m["pass"] == 1:
                if args.max_per_class == 0 or kept_per_class[person] < args.max_per_class:
                    rel_name = os.path.basename(p)
                    dst_dir = os.path.join(args.out_dir_clean, person)
                    dst_path = os.path.join(dst_dir, rel_name)

                    if args.save_mode == "crop" and m["face_detected"] == 1:
                        crop = (m["face_x"], m["face_y"], m["face_w"], m["face_h"])
                    else:
                        crop = None

                    if save_image(p, dst_path, crop=crop, resize=resize_to):
                        kept_per_class[person] += 1
                        kept_total += 1

            # (Opcional) Guardar rechazadas para inspección
            elif args.out_dir_failed:
                rel_name = os.path.basename(p)
                dst_dir = os.path.join(args.out_dir_failed, person)
                dst_path = os.path.join(dst_dir, rel_name)
                if save_image(p, dst_path, crop=None, resize=resize_to):
                    failed_total += 1

            if i % 100 == 0:
                elapsed = time.time() - t0
                print(f"[Info] Procesadas {i}/{len(img_paths)} imágenes en {elapsed:.1f}s | kept={kept_total}, failed_saved={failed_total}")

    # CSV
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

    # Resumen por persona
    summary = []
    for person, kept in sorted(kept_per_class.items()):
        summary.append({"person": person, "kept": kept})
    if pd is not None:
        try:
            df2 = pd.DataFrame(summary, columns=["person","kept"])
            df2.to_csv(args.csv_summary_out, index=False, encoding="utf-8")
        except Exception:
            with open(args.csv_summary_out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["person","kept"])
                writer.writeheader()
                writer.writerows(summary)
    else:
        with open(args.csv_summary_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["person","kept"])
            writer.writeheader()
            writer.writerows(summary)

    print("\n=== RESUMEN ===")
    print(f"Total aprobadas copiadas: {sum(k['kept'] for k in [{'kept': v} for v in kept_per_class.values()])}")
    print(f"Clases afectadas: {len(kept_per_class)}")
    if args.out_dir_failed:
        print(f"Total rechazadas copiadas: {failed_total}")
    print(f"CSV métricas: {os.path.abspath(args.csv_out)}")
    print(f"CSV resumen por persona: {os.path.abspath(args.csv_summary_out)}")
    print(f"Salida (aprobadas): {os.path.abspath(args.out_dir_clean)}")
    if args.out_dir_failed:
        print(f"Salida (rechazadas): {os.path.abspath(args.out_dir_failed)}")


if __name__ == "__main__":
    main()

"""python quality_to_clean_dataset.py \
  --img_dir ../data/preprocessed/test \
  --out_dir_clean ../data/clean/test \
  --use_face_roi_metrics \
  --decision_mode weighted --score_thresh 0.55 \
  --min_lap_var 60 --min_face_px 60 --min_face_ratio 0.02 \
  --min_brightness 40 --max_brightness 220 \
  --max_underexp_frac 0.60 --max_overexp_frac 0.35 """