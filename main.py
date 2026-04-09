import os
import io
import pickle
import base64
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace

# ─── Config ────────────────────────────────────────────────────────────────
CATALOG_DIR     = Path("catalog")
CACHE_FILE      = Path("cache/descriptors.pkl")
MODEL_NAME      = "ArcFace"       # Opciones: ArcFace, Facenet, VGG-Face
DETECTOR        = "retinaface"    # Opciones: retinaface, mtcnn, opencv
DISTANCE_METRIC = "cosine"
MATCH_THRESHOLD = 0.40            # Menor = más estricto

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tribuna")

# ─── App ───────────────────────────────────────────────────────────────────
app = FastAPI(title="TribunaFinder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── State ─────────────────────────────────────────────────────────────────
catalog_data = []   # [{photo, face_index, box, embedding}]


# ─── Helpers ───────────────────────────────────────────────────────────────

def box_to_dict(region: dict) -> dict:
    return {
        "x": int(region.get("x", 0)),
        "y": int(region.get("y", 0)),
        "w": int(region.get("w", 0)),
        "h": int(region.get("h", 0)),
    }


def build_catalog():
    """Escanea todas las fotos del catálogo y guarda embeddings."""
    global catalog_data
    CACHE_FILE.parent.mkdir(exist_ok=True, parents=True)

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    photos = [p for p in sorted(CATALOG_DIR.iterdir()) if p.suffix.lower() in extensions and p.is_file()]

    if not photos:
        log.warning("No se encontraron fotos en /catalog")
        return

    if CACHE_FILE.exists():
        log.info("Cache encontrado — cargando descriptores...")
        with open(CACHE_FILE, "rb") as f:
            catalog_data = pickle.load(f)
        log.info(f"  {len(catalog_data)} caras cargadas desde cache")
        return

    log.info(f"Procesando {len(photos)} fotos del catálogo...")
    catalog_data = []

    for photo_path in photos:
        log.info(f"  Escaneando {photo_path.name}...")
        try:
            results = DeepFace.represent(
                img_path=str(photo_path),
                model_name=MODEL_NAME,
                detector_backend=DETECTOR,
                enforce_detection=False,
            )
            for i, r in enumerate(results):
                catalog_data.append({
                    "photo":      photo_path.name,
                    "face_index": i,
                    "box":        box_to_dict(r.get("facial_area", {})),
                    "embedding":  np.array(r["embedding"]),
                })
            log.info(f"    {len(results)} caras encontradas")
        except Exception as e:
            log.error(f"    Error procesando {photo_path.name}: {e}")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(catalog_data, f)

    log.info(f"Cache guardado — {len(catalog_data)} caras en total")


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(1 - np.dot(a, b))


def image_to_base64(photo_name: str, box: dict) -> str:
    """Devuelve la foto con la cara resaltada en base64."""
    path = CATALOG_DIR / photo_name
    img  = cv2.imread(str(path))
    if img is None:
        return ""

    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 135), 3)

    # Fondo semitransparente sobre el recuadro
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 135), -1)
    img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 135), 3)

    # Label
    label = "Sos vos!"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img, (x, y - 28), (x + 110, y), (0, 255, 135), -1)
    cv2.putText(img, label, (x + 4, y - 8), font, 0.7, (0, 0, 0), 2)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode("utf-8")


# ─── Startup ───────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    if not CATALOG_DIR.exists():
        CATALOG_DIR.mkdir(parents=True)
    build_catalog()


# ─── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path("static/index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>TribunaFinder</h1><p>Frontend no encontrado en /static/index.html</p>")


@app.get("/status")
async def status():
    return {
        "faces_in_catalog": len(catalog_data),
        "photos": list({d["photo"] for d in catalog_data}),
        "model": MODEL_NAME,
        "ready": len(catalog_data) > 0,
    }


@app.post("/find")
async def find_face(file: UploadFile = File(...)):
    """Recibe una selfie y busca la mejor coincidencia en el catálogo."""

    if not catalog_data:
        raise HTTPException(503, "Catálogo vacío — agregá fotos a /catalog y reiniciá el servidor")

    # Leer imagen subida
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, "No se pudo leer la imagen")

    # Extraer embedding de la selfie
    try:
        results = DeepFace.represent(
            img_path=img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True,
        )
    except ValueError:
        raise HTTPException(422, "No se detectó ninguna cara en la foto. Intentá con una imagen más clara y de frente.")
    except Exception as e:
        raise HTTPException(500, f"Error procesando la imagen: {str(e)}")

    if not results:
        raise HTTPException(422, "No se detectó ninguna cara en la foto.")

    selfie_embedding = np.array(results[0]["embedding"])

    # Comparar contra catálogo
    best_dist  = float("inf")
    best_match = None

    for entry in catalog_data:
        dist = cosine_distance(selfie_embedding, entry["embedding"])
        if dist < best_dist:
            best_dist  = dist
            best_match = entry

    confidence = round(max(0, min(1, 1 - best_dist / 0.8)) * 100, 1)
    found      = best_dist < MATCH_THRESHOLD

    if found and best_match:
        img_b64 = image_to_base64(best_match["photo"], best_match["box"])
        return JSONResponse({
            "found":      True,
            "photo":      best_match["photo"],
            "box":        best_match["box"],
            "distance":   round(float(best_dist), 4),
            "confidence": confidence,
            "image_b64":  img_b64,
        })
    else:
        return JSONResponse({
            "found":      False,
            "confidence": confidence,
            "distance":   round(float(best_dist), 4),
        })


@app.post("/rebuild-cache")
async def rebuild_cache():
    """Elimina el cache y vuelve a procesar el catálogo."""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    build_catalog()
    return {"ok": True, "faces": len(catalog_data)}


# ─── Static files ──────────────────────────────────────────────────────────
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
