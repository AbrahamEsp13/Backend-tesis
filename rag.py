"""Recuperación (la "R" de RAG): del PDF al contexto que recibe el modelo.

Flujo:
  1. Extraer el texto del PDF (en memoria, sin archivos temporales).
  2. Fragmentar en trozos de ~TAMANO_FRAGMENTO caracteres con SOLAPAMIENTO, sin cortar palabras.
  3. Indexar los fragmentos en una colección de ChromaDB exclusiva de este documento
     (se elimina al terminar: no se mezclan PDFs de distintos usuarios ni crece la memoria).
  4. Seleccionar por COBERTURA: muestreo del punto más lejano sobre los embeddings.
     Se empieza por el fragmento más representativo (el más cercano al centroide) y se agrega,
     uno a uno, el fragmento más distinto de los ya elegidos, hasta llenar el presupuesto.
     Así el contexto abarca los distintos temas del documento en lugar de solo una parte.
  5. Ordenar los fragmentos elegidos como aparecen en el documento y unirlos en el contexto.
"""
import io
import re
import uuid

import chromadb
import numpy as np
from pypdf import PdfReader

TAMANO_FRAGMENTO = 1000          # caracteres por fragmento
SOLAPAMIENTO = 200               # caracteres compartidos entre fragmentos consecutivos
MAX_CARACTERES_CONTEXTO = 24000  # presupuesto de contexto para el modelo (~6k tokens)
MAX_FRAGMENTOS = MAX_CARACTERES_CONTEXTO // TAMANO_FRAGMENTO

# Cliente en memoria. Cada documento usa su propia colección temporal.
_chroma = chromadb.Client()


class DocumentoSinTexto(ValueError):
    """El PDF no tiene texto extraíble (p. ej. un escaneo sin OCR)."""


def extraer_texto(contenido_pdf: bytes) -> str:
    lector = PdfReader(io.BytesIO(contenido_pdf))
    paginas = [(pagina.extract_text() or "") for pagina in lector.pages]
    texto = "\n".join(paginas)
    # Normaliza espacios: los PDF suelen traer saltos y espacios de más
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", texto)).strip()


def fragmentar(texto: str, tamano: int = TAMANO_FRAGMENTO, solapamiento: int = SOLAPAMIENTO) -> list[str]:
    """Divide el texto en fragmentos de ~`tamano` caracteres que se solapan, cortando en espacios."""
    fragmentos = []
    inicio = 0
    while inicio < len(texto):
        fin = min(inicio + tamano, len(texto))
        if fin < len(texto):
            espacio = texto.rfind(" ", inicio + tamano // 2, fin)
            if espacio != -1:
                fin = espacio
        fragmento = texto[inicio:fin].strip()
        if fragmento:
            fragmentos.append(fragmento)
        if fin >= len(texto):
            break
        siguiente = fin - solapamiento
        espacio = texto.find(" ", siguiente, fin)
        inicio = espacio + 1 if espacio != -1 else max(fin, inicio + 1)
    return fragmentos


def seleccionar_por_cobertura(embeddings: np.ndarray, k: int) -> list[int]:
    """Índices de k fragmentos que cubren el espacio semántico (muestreo del punto más lejano).

    Usa distancia coseno. Devuelve los índices ordenados por posición en el documento.
    """
    n = len(embeddings)
    if k >= n:
        return list(range(n))

    normalizados = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    centroide = normalizados.mean(axis=0)
    primero = int(np.argmax(normalizados @ centroide))  # el más representativo del documento

    elegidos = [primero]
    # distancia de cada fragmento al elegido más cercano
    distancia_minima = 1.0 - normalizados @ normalizados[primero]
    distancia_minima[primero] = -np.inf
    while len(elegidos) < k:
        siguiente = int(np.argmax(distancia_minima))
        elegidos.append(siguiente)
        distancia_minima = np.minimum(distancia_minima, 1.0 - normalizados @ normalizados[siguiente])
        distancia_minima[elegidos] = -np.inf
    return sorted(elegidos)


def construir_contexto(contenido_pdf: bytes) -> dict:
    """Devuelve {"contexto", "fragmentos_totales", "fragmentos_usados"} para un PDF."""
    texto = extraer_texto(contenido_pdf)
    if len(texto) < 50:
        raise DocumentoSinTexto("El PDF no contiene texto que se pueda leer (¿es un escaneo o una imagen?).")

    fragmentos = fragmentar(texto)
    nombre = f"doc_{uuid.uuid4().hex}"
    coleccion = _chroma.create_collection(name=nombre, metadata={"hnsw:space": "cosine"})
    try:
        coleccion.add(
            ids=[f"f{i}" for i in range(len(fragmentos))],
            documents=fragmentos,
            metadatas=[{"posicion": i} for i in range(len(fragmentos))],
        )
        datos = coleccion.get(include=["embeddings", "metadatas"])
        # Chroma no garantiza el orden: reordenamos por posición en el documento
        orden = np.argsort([m["posicion"] for m in datos["metadatas"]])
        embeddings = np.asarray(datos["embeddings"])[orden]
    finally:
        _chroma.delete_collection(nombre)

    indices = seleccionar_por_cobertura(embeddings, MAX_FRAGMENTOS)
    contexto = "\n\n".join(f"[Fragmento {n} de {len(fragmentos)}]\n{fragmentos[i]}" for n, i in enumerate(indices, 1))
    return {
        "contexto": contexto,
        "fragmentos_totales": len(fragmentos),
        "fragmentos_usados": len(indices),
    }
