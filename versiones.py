"""2.6 Versiones del examen por estudiante: preguntas y opciones en otro orden en cada intento.

- El docente lo activa en la configuración del examen (`barajar_preguntas`, `barajar_opciones`).
- El orden se decide en el servidor al iniciar y se guarda en el InicioIntento (si recarga la página,
  ve el mismo) y luego en el Intento (la revisión lo muestra igual que lo vio).
  orden = {"preguntas": [índices originales en el orden mostrado],
           "opciones": {"<índice original>": [posiciones originales de las opciones en el orden mostrado]}}
- Todo lo demás (calificación por id de opción, estadísticas, reportes, práctica) sigue usando el índice
  original: cada pregunta que recibe el estudiante lleva su `indice_original` y el navegador entrega con él.
- Si el docente edita el examen después de iniciar, `normalizar_orden` descarta lo que ya no aplica.
"""
import random

from calificacion import OPCION_MULTIPLE, ocultar_respuestas, tipo_de


def generar_orden(preguntas: list, barajar_preguntas: bool, barajar_opciones: bool, rng: random.Random | None = None) -> dict | None:
    """Un orden nuevo, o None si el examen no baraja nada."""
    if not (barajar_preguntas or barajar_opciones):
        return None
    rng = rng or random.SystemRandom()
    indices = list(range(len(preguntas)))
    if barajar_preguntas:
        rng.shuffle(indices)
    opciones = {}
    if barajar_opciones:
        for i, pregunta in enumerate(preguntas):
            if tipo_de(pregunta) != OPCION_MULTIPLE:
                continue  # 4.2: "Verdadero / Falso" no se baraja; las cortas no tienen opciones
            posiciones = list(range(len(pregunta.get("opciones") or [])))
            rng.shuffle(posiciones)
            opciones[str(i)] = posiciones
    # Ids por posición original: si el docente reordena o borra preguntas, el orden se puede reacomodar
    return {"preguntas": indices, "opciones": opciones, "ids": [p.get("id") for p in preguntas]}


def _reacomodar_por_ids(orden: dict, preguntas: list) -> dict:
    """Traduce las posiciones guardadas a las actuales usando los ids de pregunta (si el orden los tiene)."""
    ids = orden.get("ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        return orden  # orden antiguo o generado antes de que las preguntas tuvieran id
    actuales = {p.get("id"): i for i, p in enumerate(preguntas) if p.get("id")}
    nueva = {}
    for i, id_pregunta in enumerate(ids):
        if id_pregunta in actuales:
            nueva[i] = actuales[id_pregunta]
    return {
        "preguntas": [nueva[i] for i in orden.get("preguntas") or [] if isinstance(i, int) and i in nueva],
        "opciones": {str(nueva[int(k)]): v for k, v in (orden.get("opciones") or {}).items()
                     if str(k).isdigit() and int(k) in nueva},
    }


def normalizar_orden(orden: dict | None, preguntas: list) -> dict:
    """
    Un orden válido para las preguntas actuales: quita índices que ya no existen, agrega al final las
    preguntas nuevas y usa el orden original de las opciones si su número cambió.
    """
    n = len(preguntas)
    orden = _reacomodar_por_ids(orden or {}, preguntas)
    vistos, indices = set(), []
    for i in (orden or {}).get("preguntas") or []:
        if isinstance(i, int) and 0 <= i < n and i not in vistos:
            vistos.add(i)
            indices.append(i)
    indices += [i for i in range(n) if i not in vistos]
    opciones = {}
    for clave, posiciones in ((orden or {}).get("opciones") or {}).items():
        if not str(clave).isdigit() or int(clave) >= n:
            continue
        total = len(preguntas[int(clave)].get("opciones") or [])
        if isinstance(posiciones, list) and sorted(posiciones) == list(range(total)):
            opciones[str(clave)] = posiciones
    return {"preguntas": indices, "opciones": opciones}


def preguntas_en_orden(preguntas: list, orden: dict | None) -> list:
    """
    La copia del examen que ve el estudiante en este intento: sin la clave, en el orden del intento,
    cada pregunta con su `indice_original` y, si se barajaron, las opciones (y sus ids) reordenadas con
    `opciones_barajadas: true` (la interfaz oculta entonces la letra "a) ..." que puso la IA).
    """
    orden = normalizar_orden(orden, preguntas)
    sin_clave = ocultar_respuestas(preguntas)
    resultado = []
    for i in orden["preguntas"]:
        pregunta = {**sin_clave[i], "indice_original": i}
        posiciones = orden["opciones"].get(str(i))
        if posiciones:
            pregunta["opciones"] = [pregunta["opciones"][j] for j in posiciones]
            if isinstance(pregunta.get("ids_opciones"), list) and len(pregunta["ids_opciones"]) == len(posiciones):
                pregunta["ids_opciones"] = [pregunta["ids_opciones"][j] for j in posiciones]
            pregunta["opciones_barajadas"] = True
        resultado.append(pregunta)
    return resultado
