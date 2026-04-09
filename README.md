# TribunaFinder 🎯

Encontrá tu cara en una tribuna usando reconocimiento facial con ArcFace + FastAPI.

---

## Estructura del proyecto

```
tribuna-finder/
├── main.py              ← servidor FastAPI
├── requirements.txt     ← dependencias
├── render.yaml          ← configuración de deploy en Render.com
├── static/
│   └── index.html       ← frontend web
├── catalog/             ← ACÁ van tus fotos de tribuna
│   ├── foto1.jpg
│   ├── foto2.jpg
│   └── ...
└── cache/               ← se genera automáticamente (no subir a git)
```

---

## Cómo usarlo localmente

### 1. Clonar e instalar
```bash
git clone https://github.com/TU_USUARIO/tribuna-finder.git
cd tribuna-finder
pip install -r requirements.txt
```

### 2. Agregar tus fotos de tribuna
Copiá tus fotos (JPG o PNG) a la carpeta `catalog/`:
```
catalog/
  partido1.jpg
  partido2.jpg
  ...
```

### 3. Correr el servidor
```bash
uvicorn main:app --reload
```

Abrí el navegador en: **http://localhost:8000**

Al iniciar, el servidor escanea automáticamente las fotos y crea un cache en `cache/descriptors.pkl`.
La próxima vez que inicies, carga el cache directamente (mucho más rápido).

---

## Deploy en Render.com (gratis)

1. Subí el proyecto a GitHub (sin la carpeta `cache/`, ya está en `.gitignore`)
2. Entrá a [render.com](https://render.com) y creá una cuenta gratuita
3. **New → Web Service → conectá tu repo de GitHub**
4. Render detecta el `render.yaml` automáticamente
5. Hacé click en **Deploy**

> ⚠️ En el plan gratuito de Render, el servidor se "duerme" tras 15 min de inactividad.
> La primera visita puede tardar ~30 segundos en despertar.

### ¿Cómo subo las fotos del catálogo a Render?
En el plan gratuito, las fotos van **dentro del repo en la carpeta `catalog/`**.
Si las fotos son pesadas (>50MB en total), considerá el plan pagado con disco persistente.

---

## API Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Frontend web |
| GET | `/status` | Estado del servidor y cantidad de caras |
| POST | `/find` | Busca una cara (body: `file` = imagen) |
| POST | `/rebuild-cache` | Regenera el cache del catálogo |

---

## Configuración avanzada (en main.py)

```python
MODEL_NAME      = "ArcFace"    # ArcFace (mejor), Facenet, VGG-Face
DETECTOR        = "retinaface" # retinaface (mejor), mtcnn, opencv (más rápido)
MATCH_THRESHOLD = 0.40         # Bajar = más estricto, subir = más permisivo
```

---

## Tips para mejores resultados

- **Fotos de tribuna:** encuadre cerrado, que las caras tengan al menos 40-50px de alto
- **Selfie del usuario:** buena luz, de frente, sin anteojos de sol ni gorras
- Si hay muchos falsos positivos → bajá `MATCH_THRESHOLD` (ej: 0.35)
- Si no encuentra caras que deberían estar → subí `MATCH_THRESHOLD` (ej: 0.45)
