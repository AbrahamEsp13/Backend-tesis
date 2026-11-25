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
  5. Ordenar los fragmentos elegidos como aparecen en el documento y unirlos en el contexto,
     numerados ([Fragmento 1], [Fragmento 2], ...). Cada fragmento conserva sus páginas para
     que cada pregunta pueda citar de dónde salió (ver generacion.agregar_fuentes).
"""
import bisect
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


def _normalizar(texto: str) -> str:
    # Los PDF suelen traer saltos y espacios de más
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", texto)).strip()


def extraer_paginas(contenido_pdf: bytes) -> list[str]:
    """Texto normalizado de cada página (la página N está en la posición N-1)."""
    lector = PdfReader(io.BytesIO(contenido_pdf))
    return [_normalizar(pagina.extract_text() or "") for pagina in lector.pages]


def unir_paginas(paginas: list[str]) -> tuple[str, list[int]]:
    """El texto completo y la posición (carácter) donde empieza cada página."""
    inicios, posicion = [], 0
    for pagina in paginas:
        inicios.append(posicion)
        posicion += len(pagina) + 1  # +1 por el salto de línea que las une
    return "\n".join(paginas), inicios


def extraer_texto(contenido_pdf: bytes) -> str:
    return unir_paginas(extraer_paginas(contenido_pdf))[0].strip()


def fragmentar(texto: str, tamano: int = TAMANO_FRAGMENTO, solapamiento: int = SOLAPAMIENTO) -> list[str]:
    """Divide el texto en fragmentos de ~`tamano` caracteres que se solapan, cortando en espacios."""
    return [fragmento for _, _, fragmento in fragmentar_con_posiciones(texto, tamano, solapamiento)]


def fragmentar_con_posiciones(texto: str, tamano: int = TAMANO_FRAGMENTO, solapamiento: int = SOLAPAMIENTO) -> list[tuple[int, int, str]]:
    """Como `fragmentar`, pero cada fragmento va con su posición en el texto: (inicio, fin, texto)."""
    fragmentos = []
    inicio = 0
    while inicio < len(texto):
        fin = min(inicio + tamano, len(texto))
        if fin < len(texto):
            espacio = texto.rfind(" ", inicio + tamano // 2, fin)
            if espacio != -1:
                fin = espacio
        crudo = texto[inicio:fin]
        fragmento = crudo.strip()
        if fragmento:
            desde = inicio + len(crudo) - len(crudo.lstrip())
            fragmentos.append((desde, desde + len(fragmento), fragmento))
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


def pagina_de(posicion: int, inicios: list[int]) -> int:
    """Número de página (desde 1) que contiene el carácter `posicion` del texto unido."""
    return max(1, bisect.bisect_right(inicios, posicion))


def construir_contexto(contenido_pdf: bytes) -> dict:
    """
    Devuelve, para un PDF:
      - "contexto": los fragmentos elegidos, numerados como [Fragmento N], para el prompt;
      - "fragmentos": [{"numero", "texto", "pagina", "pagina_fin"}] de esos mismos fragmentos;
      - "fragmentos_totales" y "fragmentos_usados".
    """
    texto, inicios = unir_paginas(extraer_paginas(contenido_pdf))
    if len(texto.strip()) < 50:
        raise DocumentoSinTexto("El PDF no contiene texto que se pueda leer (¿es un escaneo o una imagen?).")

    con_posiciones = fragmentar_con_posiciones(texto)
    fragmentos = [fragmento for _, _, fragmento in con_posiciones]
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
    elegidos = [
        {
            "numero": n,
            "texto": fragmentos[i],
            "pagina": pagina_de(con_posiciones[i][0], inicios),
            "pagina_fin": pagina_de(con_posiciones[i][1] - 1, inicios),
        }
        for n, i in enumerate(indices, 1)
    ]
    contexto = "\n\n".join(f"[Fragmento {f['numero']}]\n{f['texto']}" for f in elegidos)
    return {
        "contexto": contexto,
        "fragmentos": elegidos,
        "fragmentos_totales": len(fragmentos),
        "fragmentos_usados": len(indices),
    }
