"""4.10 Lectura de la hoja de respuestas con hojas sintéticas (dibujadas con OpenCV) y fotos simuladas."""
import math

import cv2
import numpy as np
import pytest

import lectura_hojas as lh

PX_MM = 4.0  # resolución de la "impresión" sintética


def dibujar_hoja(opciones_por_pregunta, marcadas, *, logo=True, lapiz=False, multiple=(), tenue=()):
    """Una hoja carta con encabezado (y un logo cuadrado como distractor) y el recuadro de respuestas."""
    geo = lh.disposicion(opciones_por_pregunta)
    hoja = np.full((int(279 * PX_MM), int(216 * PX_MM)), 255, np.uint8)
    ox, oy = 18.0, 70.0  # posición del recuadro en la hoja (mm)
    mm = lambda x, y: (int((ox + x) * PX_MM), int((oy + y) * PX_MM))  # noqa: E731

    if logo:  # logo cuadrado y negro en el encabezado: no debe confundirse con una marca
        cv2.rectangle(hoja, (int(20 * PX_MM), int(15 * PX_MM)), (int(40 * PX_MM), int(35 * PX_MM)), 0, -1)
    cv2.putText(hoja, "ESCUELA SECUNDARIA 12", (int(60 * PX_MM), int(30 * PX_MM)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)

    alto = geo["alto"]
    for (x, y), lado in zip([(0, 0), (lh.ANCHO - lh.MARCA, 0), (lh.ANCHO - lh.MARCA, alto - lh.MARCA), (0, alto - lh.MARCA)],
                            [lh.MARCA_GRANDE, lh.MARCA, lh.MARCA, lh.MARCA]):
        cv2.rectangle(hoja, mm(x, y), mm(x + lado, y + lado), 0, -1)

    radio = int(lh.DIAMETRO / 2 * PX_MM)
    color_relleno = 90 if lapiz else 0
    for i, circulos in enumerate(geo["centros"]):
        cx, cy = mm(circulos[0][0] - 9, circulos[0][1] + 1.5)
        cv2.putText(hoja, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        for j, (x, y) in enumerate(circulos):
            centro = mm(x, y)
            cv2.circle(hoja, centro, radio, 0, 1)
            cv2.putText(hoja, "ABCDEF"[j], (centro[0] - 4, centro[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 190, 1)
            if marcadas.get(i) == j or (i, j) in multiple:
                cv2.circle(hoja, centro, radio - 1, color_relleno, -1)
            elif (i, j) in tenue:  # una raya ligera, no un relleno
                cv2.line(hoja, (centro[0] - 4, centro[1]), (centro[0] + 4, centro[1]), 0, 3)
    return hoja


def fotografiar(hoja, *, angulo=0.0, perspectiva=0.0, fondo=90, ruido=6, semilla=0):
    """Simula una foto: hoja sobre una mesa, girada, con perspectiva, ruido y luz desigual."""
    rng = np.random.default_rng(semilla)
    h, w = hoja.shape
    lienzo = np.full((int(h * 1.5), int(w * 1.5)), fondo, np.uint8)
    dx, dy = (lienzo.shape[1] - w) // 2, (lienzo.shape[0] - h) // 2
    lienzo[dy:dy + h, dx:dx + w] = hoja
    centro = (lienzo.shape[1] / 2, lienzo.shape[0] / 2)
    giro = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    lienzo = cv2.warpAffine(lienzo, giro, (lienzo.shape[1], lienzo.shape[0]), borderValue=fondo)
    alto, ancho = lienzo.shape
    p = perspectiva * ancho
    origen = np.float32([[0, 0], [ancho, 0], [ancho, alto], [0, alto]])
    destino = np.float32([[p, 0], [ancho - p, p * 0.5], [ancho, alto], [0, alto - p * 0.3]])
    lienzo = cv2.warpPerspective(lienzo, cv2.getPerspectiveTransform(origen, destino), (ancho, alto), borderValue=fondo)
    degradado = np.tile(np.linspace(0.8, 1.0, ancho), (alto, 1))  # luz desigual
    foto = np.clip(lienzo * degradado + rng.normal(0, ruido, lienzo.shape), 0, 255).astype(np.uint8)
    _, jpg = cv2.imencode(".jpg", cv2.cvtColor(foto, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpg.tobytes()


def indices(resultado):
    return [r["indice"] for r in resultado["lectura"]]


def test_disposicion():
    geo = lh.disposicion([4] * 25)
    assert geo["columnas"] == 2 and geo["filas"] == 20
    assert geo["alto"] == lh.MARGEN_ARRIBA + 20 * lh.ALTO_FILA + lh.MARGEN_ABAJO
    assert geo["centros"][20][0][0] > geo["ancho_columna"]  # la pregunta 21 va en la segunda columna
    assert lh.disposicion([4] * 81) is None  # más de 4 columnas
    assert lh.disposicion([7]) is None
    assert lh.disposicion([]) is None


@pytest.mark.parametrize("angulo,perspectiva", [(0, 0), (4, 0.03), (-7, 0.05), (90, 0.02), (180, 0), (-90, 0.04)])
def test_lee_fotos_giradas_y_en_perspectiva(angulo, perspectiva):
    opciones = [4] * 12
    marcadas = {0: 0, 1: 3, 2: 1, 3: 2, 5: 0, 6: 1, 7: 3, 8: 2, 9: 1, 10: 0, 11: 3}  # la 5 (índice 4) en blanco
    foto = fotografiar(dibujar_hoja(opciones, marcadas), angulo=angulo, perspectiva=perspectiva)
    resultado = lh.leer_hoja(foto, opciones)
    assert indices(resultado) == [marcadas.get(i) for i in range(12)]
    assert resultado["lectura"][4]["estado"] == "vacia"
    assert resultado["vista"].startswith("data:image/jpeg;base64,")


def test_varias_columnas_lapiz_y_opciones_distintas():
    opciones = [4] * 30 + [3, 5]
    marcadas = {i: i % o for i, o in enumerate(opciones)}
    foto = fotografiar(dibujar_hoja(opciones, marcadas, lapiz=True), angulo=3, perspectiva=0.02, semilla=2)
    assert indices(lh.leer_hoja(foto, opciones, con_vista=False)) == [marcadas[i] for i in range(len(opciones))]


def test_multiple_y_tenue():
    opciones = [4] * 5
    foto = fotografiar(dibujar_hoja(opciones, {0: 1, 2: 0, 3: 3}, multiple={(1, 0), (1, 2)}, tenue={(4, 2)}))
    lectura = lh.leer_hoja(foto, opciones)["lectura"]
    assert lectura[1]["estado"] == "multiple" and lectura[1]["indice"] is None
    assert lectura[4]["estado"] in ("vacia", "dudosa")
    assert [r["estado"] for r in (lectura[0], lectura[2], lectura[3])] == ["ok", "ok", "ok"]


def test_hoja_de_otro_examen_o_sin_marcas():
    foto = fotografiar(dibujar_hoja([4] * 5, {0: 0}))
    with pytest.raises(lh.HojaIlegible, match="no corresponden"):
        lh.leer_hoja(foto, [4] * 20)  # proporción del recuadro muy distinta
    sin_marcas = fotografiar(np.full((900, 700), 255, np.uint8))
    with pytest.raises(lh.HojaIlegible, match="marcas"):
        lh.leer_hoja(sin_marcas, [4] * 5)
    with pytest.raises(lh.HojaIlegible, match="abrir"):
        lh.leer_hoja(b"no es una imagen", [4] * 5)


def test_decidir():
    assert lh.decidir([0.05, 0.9, 0.1, 0.0]) == (1, "ok")
    assert lh.decidir([0.8, 0.85, 0.1, 0.0]) == (None, "multiple")
    assert lh.decidir([0.05, 0.3, 0.1, 0.0]) == (1, "dudosa")
    assert lh.decidir([0.5, 0.35, 0.0, 0.0]) == (0, "dudosa")
    assert lh.decidir([0.02, 0.05, 0.1, 0.0]) == (None, "vacia")
    assert math.isclose(lh.UMBRAL_MARCADA, 0.45)
