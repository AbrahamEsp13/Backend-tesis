# QuizAI: backend

API en FastAPI que genera evaluaciones de opción múltiple a partir de un PDF con RAG
(ChromaDB + Gemini), y gestiona usuarios, clases, publicación de exámenes y la
calificación de los intentos de los estudiantes.

El frontend está en el repositorio `Frontend-tesis`.

## Requisitos

- Python 3.12
- Una base de datos PostgreSQL (en producción se usa Neon)
- Una API key de Gemini

## Puesta en marcha

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
uvicorn main:app --reload      # http://localhost:8000  (documentación: /docs)
```

Al arrancar se crean las tablas que falten y se agregan las columnas nuevas a las
tablas existentes (ver `agregar_columnas_faltantes` en `database.py`).

## Pruebas

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Las pruebas (`tests/`) usan una base SQLite temporal y simulan el RAG y Gemini, así que
no necesitan `.env`, conexión a Neon ni API key, y nunca modifican datos reales.
Cubren autenticación, CORS, cuenta, clases, intentos y calificación, generación
(complejidad, créditos, validación de la salida del modelo) y la migración de columnas.

## Variables de entorno (`.env`)

| Variable | Obligatoria | Descripción |
|---|---|---|
| `DATABASE_URL` | Sí | Cadena de conexión de PostgreSQL (`postgresql://...`). |
| `JWT_SECRET` | Sí | Clave para firmar los tokens, mínimo 32 caracteres. Generar con `python -c "import secrets; print(secrets.token_urlsafe(48))"`. |
| `GEMINI_API_KEY` | Sí | API key de Google Gemini. |
| `CORS_ORIGINS` | En producción | Orígenes del frontend permitidos, separados por comas (p. ej. `https://mi-app.vercel.app`). Sin ella solo se permite `http://localhost:5173`. |

El archivo `.env` está en `.gitignore`: no se sube al repositorio.

## Estructura

| Archivo | Contenido |
|---|---|
| `main.py` | Aplicación FastAPI: autenticación, cuestionarios, generación, intentos, comunidad y cuenta. |
| `clases.py` | Endpoints de clases (`/api/clases`) y reglas de visibilidad para estudiantes. |
| `reportes.py` | Reportes de preguntas problemáticas hechos por los estudiantes. |
| `resumen_docente.py` | Resumen para el inicio del docente: entregas recientes, pendientes y promedio por clase. |
| `valoraciones.py` | Calidad del generador: el docente valora cada pregunta generada; estadísticas por complejidad. |
| `auth.py` | Tokens JWT y dependencias `usuario_actual` / `docente_actual`. |
| `database.py` | Modelos de SQLAlchemy y migración de columnas al arrancar. |
| `rag.py` | Recuperación: extrae el texto del PDF, selecciona fragmentos por cobertura temática y registra sus páginas. |
| `generacion.py` | Generación con Gemini (salida JSON con esquema), niveles de complejidad, validación de preguntas y cita al documento (`fuente`). |
| `calificacion.py` | Calificación de intentos en el servidor y ocultamiento de la clave para estudiantes. |
| `exportar_valoraciones.py` | Exporta la calidad del generador de todos los docentes (resumen y CSV) para la tesis. |
| `prueba_rag.py`, `ver_modelos.py` | Scripts de prueba manual (no forman parte de la API). |

## Reglas principales

- **Autenticación:** el login devuelve un JWT (7 días) que el frontend envía en
  `Authorization: Bearer <token>`. El usuario siempre se obtiene del token, nunca de un
  id enviado por el cliente.
- **Créditos:** cada docente tiene 5 generaciones por día; se restablecen al cambiar la fecha
  y solo se descuentan si la generación termina bien.
- **Visibilidad:** un estudiante solo ve (y solo puede entregar) los cuestionarios
  **publicados** y **asignados a una clase** a la que pertenece. Nunca recibe
  `respuesta_correcta` ni la justificación antes de entregar.
- **Calificación:** se hace en el servidor al entregar (`calificacion.py`); cada reintento
  se guarda como un intento nuevo.

## Endpoints

La referencia completa e interactiva está en `/docs` con el servidor en marcha. Resumen:

| Área | Rutas |
|---|---|
| Autenticación | `POST /api/auth/register`, `POST /api/auth/login` |
| Cuenta | `PUT /api/usuarios/{id}/perfil`, `PUT .../password`, `PUT .../preferencias`, `GET .../creditos`, `DELETE /api/usuarios/{id}` |
| Generación | `POST /api/generar-cuestionario` (PDF, número de preguntas, complejidad) |
| Cuestionarios | `GET /api/cuestionarios`, `PUT/DELETE /api/cuestionarios/{id}`, `PUT .../publicar` (con `clases` opcional: publica y asigna), `GET .../clases`, `PUT .../despublicar`, `.../compartir`, `POST .../clonar` |
| Intentos | `POST /api/cuestionarios/{id}/intentos` (estudiante; devuelve clave y retroalimentación), `GET /api/cuestionarios/{id}/intentos` (docente), `GET /api/intentos/mios` (estudiante), `GET /api/intentos/{id}` (revisión: el estudiante autor o el docente dueño) |
| Reportes | `POST /api/intentos/{id}/reportes` (estudiante), `DELETE /api/cuestionarios/{id}/reportes/{rid}` (docente); el docente los recibe en `GET .../intentos` |
| Calidad del generador | `PUT /api/cuestionarios/{id}/preguntas/{i}/valoracion`, `GET /api/estadisticas/generador` |
| Clases | `GET/POST /api/clases`, `GET/PUT/DELETE /api/clases/{id}`, `PUT .../codigo`, `POST/DELETE .../cuestionarios/{cid}`, `DELETE .../estudiantes/{eid}`, `POST /api/clases/unirse`, `DELETE .../salir` |
| Comunidad | `GET /api/comunidad/cuestionarios` |
| Inicio del docente | `GET /api/docente/resumen` |
