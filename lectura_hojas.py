"""4.10 Calificar con foto: lectura de la hoja de respuestas impresa (visión por computadora, OpenCV).

La hoja (components/impresion/DocumentoExamen.jsx, utils/hojaRespuestas.js) tiene un recuadro con
cuatro marcas negras cuadradas en las esquinas; la de arriba a la izquierda es más grande y sirve para
saber la orientación. Dentro del recuadro, cada pregunta es un renglón de círculos en posiciones fijas
en milímetros. Como la foto se endereza con las marcas, la escala de impresión no importa.

Pasos:
  1. Buscar las cuatro marcas: contornos cuadrados, rellenos y oscuros. Si hay más candidatos (logos),
     se elige el cuadrilátero cuya proporción se parece más a la del recuadro esperado.
  2. Enderezar (transformación de perspectiva) a un rectángulo de ESCALA px por mm.
  3. Medir qué tan relleno está el interior de cada círculo (sin el contorno impreso; las letras van en
     gris claro y quedan como fondo).
  4. Decidir por pregunta: una opción, vacía, más de una (multiple) o dudosa. El docente revisa todo
     antes de registrar: la lectura nunca se guarda sola.

IMPORTANTE: la disposición (constantes de abajo) debe coincidir con utils/hojaRespuestas.js.
"""
import base64
import itertools
import math

import cv2
import numpy as np

# ---------- Disposición de la hoja (mm) — igual que utils/hojaRespuestas.js ----------
ANCHO = 180.0           # ancho del recuadro
FILAS_POR_COLUMNA = 20
MAX_COLUMNAS = 4
MAX_OPCIONES = 6
MARCA_GRANDE = 9.0      # marca de arriba a la izquierda (orientación)
MARCA = 7.0             # las otras tres
MARGEN_ARRIBA = 14.0    # de la orilla del recuadro al primer renglón
MARGEN_ABAJO = 14.0
ALTO_FILA = 8.0
X_PRIMER_CIRCULO = 12.0  # dentro de cada columna (antes va el número de la pregunta)
PASO_CIRCULO = 7.0
DIAMETRO = 5.0

ESCALA = 6.0            # px por mm de la imagen enderezada

# ---------- Decisión ----------
UMBRAL_MARCADA = 0.45   # fracción oscura del interior del círculo
UMBRAL_DUDOSA = 0.22


class HojaIlegible(ValueError):
    """No se pudo leer la hoja (mensaje para el docente)."""


def disposicion(opciones_por_pregunta: list[int]) -> dict | None:
    """
    Geometría del recuadro para un examen: {columnas, filas, alto, ancho_columna, centros}.
    centros[i][j] = (x, y) en mm del círculo j de la pregunta i. None si no cabe (muchas preguntas u opciones).
    """
    n = len(opciones_por_pregunta)
    if n == 0 or max(opciones_por_pregunta) > MAX_OPCIONES:
        return None
    columnas = math.ceil(n / FILAS_POR_COLUMNA)
    if columnas > MAX_COLUMNAS:
        return None
    ancho_columna = ANCHO / columnas
    if X_PRIMER_CIRCULO + max(opciones_por_pregunta) * PASO_CIRCULO > ancho_columna:
        return None
    filas = min(n, FILAS_POR_COLUMNA)
    centros = []
    for i, opciones in enumerate(opciones_por_pregunta):
        columna, fila = divmod(i, FILAS_POR_COLUMNA)
        x0 = columna * ancho_columna + X_PRIMER_CIRCULO + DIAMETRO / 2
        y = MARGEN_ARRIBA + fila * ALTO_FILA + ALTO_FILA / 2
        centros.append([(x0 + j * PASO_CIRCULO, y) for j in range(opciones)])
    return {
        "columnas": columnas,
        "filas": filas,
        "alto": MARGEN_ARRIBA + filas * ALTO_FILA + MARGEN_ABAJO,
        "ancho_columna": ancho_columna,
        "centros": centros,
    }


def centros_de_marcas(alto: float) -> np.ndarray:
    """Centros (mm) de las marcas: arriba-izq, arriba-der, abajo-der, abajo-izq."""
    g, m = MARCA_GRANDE / 2, MARCA / 2
    return np.float32([[g, g], [ANCHO - m, m], [ANCHO - m, alto - m], [m, alto - m]])


# ---------- 1. Marcas ----------

def _candidatos(binaria: np.ndarray) -> list[dict]:
    """Cuadrados rellenos y oscuros (posibles marcas)."""
    area_imagen = binaria.shape[0] * binaria.shape[1]
    contornos, _ = cv2.findContours(binaria, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if not area_imagen * 0.00015 < area < area_imagen * 0.02:
            continue
        aproximado = cv2.approxPolyDP(contorno, 0.05 * cv2.arcLength(contorno, True), True)
        if len(aproximado) != 4 or not cv2.isContourConvex(aproximado):
            continue
        x, y, w, h = cv2.boundingRect(aproximado)
        if not 0.7 < w / h < 1.4 or area / (w * h) < 0.75:
            continue
        mascara = np.zeros_like(binaria)
        cv2.drawContours(mascara, [contorno], -1, 255, -1)
        if cv2.mean(binaria, mask=mascara)[0] < 0.85 * 255:
            continue  # contorno de un cuadro vacío, no una marca rellena
        momento = cv2.moments(contorno)
        candidatos.append({"centro": (momento["m10"] / momento["m00"], momento["m01"] / momento["m00"]), "area": area})
    # Si hay muchos (logos, texturas), las marcas suelen estar entre los más grandes
    return sorted(candidatos, key=lambda c: -c["area"])[:14]


def _ordenar_esquinas(puntos: list[dict]) -> list[dict]:
    """Ordena 4 puntos alrededor de su centro (sentido horario en la imagen) empezando por arriba-izq."""
    cx = sum(p["centro"][0] for p in puntos) / 4
    cy = sum(p["centro"][1] for p in puntos) / 4
    ordenados = sorted(puntos, key=lambda p: math.atan2(p["centro"][1] - cy, p["centro"][0] - cx))
    inicio = min(range(4), key=lambda i: ordenados[i]["centro"][0] + ordenados[i]["centro"][1])
    return ordenados[inicio:] + ordenados[:inicio]


def _distancia(a, b) -> float:
    return math.dist(a["centro"], b["centro"])


def buscar_marcas(binaria: np.ndarray, proporcion: float) -> np.ndarray:
    """
    Las cuatro marcas en orden arriba-izq, arriba-der, abajo-der, abajo-izq de la HOJA (no de la foto):
    la marca más grande es la de arriba a la izquierda, así una foto girada se lee igual.
    `proporcion` = ancho / alto esperado entre centros de marcas.
    """
    candidatos = _candidatos(binaria)
    if len(candidatos) < 4:
        raise HojaIlegible(
            "No se encontraron las cuatro marcas negras de las esquinas. Toma la foto de frente, con buena luz "
            "y con toda la hoja de respuestas a la vista."
        )
    mejor, mejor_error = None, None
    for combinacion in itertools.combinations(candidatos, 4):
        esquinas = _ordenar_esquinas(list(combinacion))
        # Girar para que la marca más grande quede primero (arriba a la izquierda de la hoja)
        grande = max(range(4), key=lambda i: esquinas[i]["area"])
        esquinas = esquinas[grande:] + esquinas[:grande]
        areas = sorted(e["area"] for e in esquinas)
        if areas[-1] < 1.2 * areas[-2]:
            continue  # ninguna marca es claramente la grande
        if max(areas[:3]) > 2.0 * min(areas[:3]):
            continue  # las tres pequeñas deben parecerse
        arriba, derecha = _distancia(esquinas[0], esquinas[1]), _distancia(esquinas[1], esquinas[2])
        abajo, izquierda = _distancia(esquinas[3], esquinas[2]), _distancia(esquinas[0], esquinas[3])
        if min(arriba, derecha, abajo, izquierda) < 50:
            continue
        error = abs(math.log(((arriba + abajo) / (derecha + izquierda)) / proporcion))
        error += abs(math.log(arriba / abajo)) + abs(math.log(izquierda / derecha))  # paralelismo aproximado
        if mejor_error is None or error < mejor_error:
            mejor, mejor_error = esquinas, error
    if mejor is None or mejor_error > 0.45:
        raise HojaIlegible(
            "Las marcas de la foto no corresponden a la hoja de respuestas de este examen. Revisa que sea la hoja "
            "correcta (el número de preguntas debe coincidir) y que se vean las cuatro esquinas."
        )
    return np.float32([e["centro"] for e in mejor])


# ---------- 2 a 4. Enderezar, medir y decidir ----------

def _abrir(imagen_bytes: bytes) -> np.ndarray:
    datos = np.frombuffer(imagen_bytes, np.uint8)
    imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR)  # respeta la orientación EXIF del celular
    if imagen is None:
        raise HojaIlegible("No se pudo abrir la imagen. Usa una foto JPG o PNG.")
    lado = max(imagen.shape[:2])
    if lado > 2000:
        imagen = cv2.resize(imagen, None, fx=2000 / lado, fy=2000 / lado, interpolation=cv2.INTER_AREA)
    if min(imagen.shape[:2]) < 400:
        raise HojaIlegible("La foto tiene muy poca resolución. Tómala más de cerca.")
    return imagen


def _binarizar(gris: np.ndarray) -> np.ndarray:
    """Tinta = 255. Umbral adaptativo con ventana grande: tolera sombras suaves sin vaciar las marcas."""
    bloque = (max(gris.shape) // 8) | 1
    return cv2.adaptiveThreshold(cv2.GaussianBlur(gris, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, bloque, 15)


def _relleno(binaria: np.ndarray, x_mm: float, y_mm: float) -> float:
    """Fracción oscura del interior del círculo (radio 70%: deja fuera el contorno impreso)."""
    radio = int(DIAMETRO / 2 * 0.7 * ESCALA)
    cx, cy = int(round(x_mm * ESCALA)), int(round(y_mm * ESCALA))
    mascara = np.zeros_like(binaria)
    cv2.circle(mascara, (cx, cy), radio, 255, -1)
    return cv2.mean(binaria, mask=mascara)[0] / 255


def decidir(rellenos: list[float]) -> tuple[int | None, str]:
    """(índice elegido o None, estado): 'ok', 'vacia', 'multiple' o 'dudosa'."""
    if not rellenos:
        return None, "vacia"
    orden = sorted(range(len(rellenos)), key=lambda j: -rellenos[j])
    primero = orden[0]
    segundo = rellenos[orden[1]] if len(orden) > 1 else 0.0
    if rellenos[primero] >= UMBRAL_MARCADA:
        if segundo >= UMBRAL_MARCADA:
            return None, "multiple"
        if segundo >= UMBRAL_DUDOSA and rellenos[primero] - segundo < 0.25:
            return primero, "dudosa"
        return primero, "ok"
    if rellenos[primero] >= UMBRAL_DUDOSA:
        return primero, "dudosa"
    return None, "vacia"


def _vista_de_control(gris: np.ndarray, centros: list, lectura: list[dict]) -> str:
    """La hoja enderezada con lo que se leyó, como data URL JPEG (para que el docente la revise)."""
    imagen = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    radio = int(DIAMETRO / 2 * ESCALA) + 2
    colores = {"ok": (60, 160, 40), "dudosa": (0, 160, 230), "multiple": (40, 40, 220)}
    for circulos, resultado in zip(centros, lectura):
        for j, (x, y) in enumerate(circulos):
            centro = (int(x * ESCALA), int(y * ESCALA))
            if resultado["estado"] == "multiple" and resultado["rellenos"][j] >= UMBRAL_MARCADA:
                cv2.circle(imagen, centro, radio, colores["multiple"], 3)
            elif j == resultado["indice"]:
                cv2.circle(imagen, centro, radio, colores[resultado["estado"]], 3)
    ancho = imagen.shape[1]
    if ancho > 900:
        imagen = cv2.resize(imagen, None, fx=900 / ancho, fy=900 / ancho, interpolation=cv2.INTER_AREA)
    _, jpg = cv2.imencode(".jpg", imagen, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()


def leer_hoja(imagen_bytes: bytes, opciones_por_pregunta: list[int], con_vista: bool = True) -> dict:
    """
    Lee una foto de la hoja de respuestas.
    Devuelve {lectura: [{indice, estado, rellenos}], vista: data URL | None}. Lanza HojaIlegible.
    """
    geometria = disposicion(opciones_por_pregunta)
    if geometria is None:
        raise HojaIlegible("Este examen tiene demasiadas preguntas u opciones para calificarse con foto.")
    imagen = _abrir(imagen_bytes)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    destino = centros_de_marcas(geometria["alto"])
    proporcion = (destino[1][0] - destino[0][0]) / (destino[3][1] - destino[0][1])
    origen = buscar_marcas(_binarizar(gris), proporcion)

    transformacion = cv2.getPerspectiveTransform(origen, destino * ESCALA)
    tamano = (int(ANCHO * ESCALA), int(geometria["alto"] * ESCALA))
    enderezada = cv2.warpPerspective(gris, transformacion, tamano, flags=cv2.INTER_LINEAR, borderValue=255)
    # Otsu sobre la hoja enderezada: tinta oscura contra papel; el gris claro de las letras queda como papel
    _, binaria = cv2.threshold(cv2.GaussianBlur(enderezada, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    lectura = []
    for circulos in geometria["centros"]:
        rellenos = [round(_relleno(binaria, x, y), 3) for x, y in circulos]
        indice, estado = decidir(rellenos)
        lectura.append({"indice": indice, "estado": estado, "rellenos": rellenos})
    return {"lectura": lectura, "vista": _vista_de_control(enderezada, geometria["centros"], lectura) if con_vista else None}
