"""4.6 Clave por identificador estable.

Cada opción tiene un id (`ids_opciones`, en paralelo a `opciones`) y cada pregunta guarda el id de
su opción correcta (`id_correcta`, privado para el estudiante). Los intentos guardan ids, no textos:
si el docente corrige el texto de una opción o borra una pregunta después de que hubo entregas,
los intentos anteriores siguen apuntando a la opción que el estudiante eligió.

- `respuesta_correcta` sigue siendo el texto (lo usa la interfaz); `id_correcta` se calcula a partir
  de él cada vez que se guardan preguntas (`normalizar_preguntas`).
- Al devolver un intento, `respuestas_para_mostrar` traduce los ids al índice y al texto ACTUALES de
  cada pregunta. Así el frontend sigue comparando textos sin saber de ids.
- Los ids son únicos dentro de un examen (se buscan en todo el examen para seguir a una pregunta
  que cambió de posición). Una respuesta a una pregunta que ya no existe deja de mostrarse.
- Cada pregunta también tiene un id estable (`id`, "pr_..."). Lo usan lo que se guarda por pregunta
  y no por opción: reportes (se reacomodan al guardar el examen), valoraciones, respuestas cortas
  (`pregunta_id` en cada calificación) y el orden de las versiones (2.6).
"""
import copy
import secrets

from sqlalchemy.orm import Session

PREFIJO = "op_"
PREFIJO_PREGUNTA = "pr_"


def nuevo_id(prefijo: str = PREFIJO) -> str:
    return prefijo + secrets.token_hex(4)


def _texto(valor) -> str:
    return valor if isinstance(valor, str) else ""


def normalizar_preguntas(preguntas: list | None) -> list:
    """
    Copia de las preguntas con ids completos y únicos, e `id_correcta` según `respuesta_correcta`.
    Conserva los ids existentes por posición: editar el texto de una opción no cambia su id.
    El `id_correcta` que venga del cliente se ignora (se recalcula siempre).
    """
    usados = set()
    ids_preguntas = set()
    resultado = []
    for original in preguntas or []:
        pregunta = dict(original)
        # Id de la pregunta: se conserva si es válido y no se repite en el examen
        id_pregunta = pregunta.get("id")
        if not isinstance(id_pregunta, str) or not id_pregunta.startswith(PREFIJO_PREGUNTA) or id_pregunta in ids_preguntas:
            id_pregunta = nuevo_id(PREFIJO_PREGUNTA)
            while id_pregunta in ids_preguntas:
                id_pregunta = nuevo_id(PREFIJO_PREGUNTA)
        ids_preguntas.add(id_pregunta)
        pregunta["id"] = id_pregunta
        opciones = pregunta.get("opciones") or []
        previos = pregunta.get("ids_opciones") if isinstance(pregunta.get("ids_opciones"), list) else []
        ids = []
        for posicion in range(len(opciones)):
            candidato = previos[posicion] if posicion < len(previos) else None
            if not isinstance(candidato, str) or not candidato.startswith(PREFIJO) or candidato in usados:
                candidato = nuevo_id()
                while candidato in usados:
                    candidato = nuevo_id()
            usados.add(candidato)
            ids.append(candidato)
        pregunta["ids_opciones"] = ids
        correcta = pregunta.get("respuesta_correcta")
        pregunta["id_correcta"] = ids[opciones.index(correcta)] if correcta in opciones else None
        resultado.append(pregunta)
    return resultado


def tiene_ids(preguntas: list | None) -> bool:
    return all(
        isinstance(p.get("ids_opciones"), list) and len(p["ids_opciones"]) == len(p.get("opciones") or []) and "id_correcta" in p
        and str(p.get("id") or "").startswith(PREFIJO_PREGUNTA)
        for p in preguntas or []
    )


def indices_por_id(preguntas: list | None) -> dict[str, int]:
    """{ id de pregunta: posición actual }."""
    return {p["id"]: i for i, p in enumerate(preguntas or []) if isinstance(p.get("id"), str)}


def indice_de_pregunta(preguntas: list, referencia: str) -> int | None:
    """La posición de una pregunta por su id ("pr_...") o, por compatibilidad, por su posición ("3")."""
    if str(referencia).startswith(PREFIJO_PREGUNTA):
        return indices_por_id(preguntas).get(referencia)
    if str(referencia).isdigit() and int(referencia) < len(preguntas or []):
        return int(referencia)
    return None


def cortas_actuales(preguntas: list, cortas: dict | None) -> dict:
    """
    Las respuestas cortas de un intento con la posición ACTUAL de su pregunta (por `pregunta_id`).
    Se omiten las de preguntas borradas; las antiguas sin `pregunta_id` conservan su posición.
    """
    posiciones = indices_por_id(preguntas)
    resultado = {}
    for clave, valor in (cortas or {}).items():
        id_pregunta = (valor or {}).get("pregunta_id")
        if id_pregunta:
            if id_pregunta in posiciones:
                resultado[str(posiciones[id_pregunta])] = valor
        elif str(clave).isdigit() and int(clave) < len(preguntas or []):
            resultado[str(clave)] = valor
    return resultado


def respuestas_por_indice_actual(preguntas: list, respuestas: dict | None, cortas: dict | None = None) -> dict:
    """
    Como respuestas_para_mostrar, pero con los ids (para volver a calificar): { posición actual: id de opción }.
    Los textos de las respuestas cortas se mueven con su pregunta (según `cortas`).
    """
    ubicacion = {}
    for i, pregunta in enumerate(preguntas or []):
        for id_opcion in pregunta.get("ids_opciones") or []:
            ubicacion[id_opcion] = i
    movimientos = _movimientos_de_cortas(preguntas, cortas)
    resultado = {}
    for clave, valor in (respuestas or {}).items():
        if valor in ubicacion:
            resultado[str(ubicacion[valor])] = valor
        elif str(clave) in movimientos:
            if movimientos[str(clave)] is not None:
                resultado[movimientos[str(clave)]] = valor
        elif not (isinstance(valor, str) and valor.startswith(PREFIJO)):
            resultado[str(clave)] = valor
    return resultado


def _movimientos_de_cortas(preguntas: list, cortas: dict | None) -> dict:
    """{ posición al entregar: posición actual (o None si la pregunta se borró) } de las respuestas cortas."""
    posiciones = indices_por_id(preguntas)
    movimientos = {}
    for clave, valor in (cortas or {}).items():
        id_pregunta = (valor or {}).get("pregunta_id")
        if id_pregunta:
            movimientos[str(clave)] = str(posiciones[id_pregunta]) if id_pregunta in posiciones else None
    return movimientos


def id_de_respuesta(pregunta: dict, valor) -> str | None:
    """El id de la opción que indica `valor` (un id o, por compatibilidad, el texto de la opción)."""
    ids = pregunta.get("ids_opciones") or []
    if valor in ids:
        return valor
    opciones = pregunta.get("opciones") or []
    if valor in opciones and opciones.index(valor) < len(ids):
        return ids[opciones.index(valor)]
    return None


def respuestas_a_ids(preguntas: list, respuestas: dict) -> dict:
    """
    { "<índice>": id } a partir de lo que envió el estudiante (ids o textos).
    Un valor que no corresponde a ninguna opción de esa pregunta se conserva tal cual (cuenta como error).
    """
    resultado = {}
    for clave, valor in (respuestas or {}).items():
        if not str(clave).isdigit() or int(clave) >= len(preguntas):
            continue
        resultado[str(clave)] = id_de_respuesta(preguntas[int(clave)], valor) or valor
    return resultado


def es_correcta(pregunta: dict, valor) -> bool:
    """Con id_correcta, compara ids; en preguntas sin ids (datos antiguos), compara textos."""
    if pregunta.get("id_correcta"):
        return valor is not None and valor == pregunta["id_correcta"]
    return valor is not None and valor == pregunta.get("respuesta_correcta")


def respuestas_para_mostrar(preguntas: list, respuestas: dict | None, seguridad: dict | None = None, cortas: dict | None = None) -> tuple[dict, dict]:
    """
    Traduce un intento guardado a índices y textos actuales: ({ "<índice>": texto }, { "<índice>": bool }).
    - id encontrado en el examen: el índice y el texto actuales de esa opción;
    - id que ya no existe (se borró la pregunta): la respuesta se omite;
    - valor que no es un id (intento antiguo sin migrar): se deja como estaba.
    """
    ubicacion = {}
    for i, pregunta in enumerate(preguntas or []):
        for j, id_opcion in enumerate(pregunta.get("ids_opciones") or []):
            if j < len(pregunta.get("opciones") or []):
                ubicacion[id_opcion] = (i, j)
    seguridad = seguridad or {}
    movimientos = _movimientos_de_cortas(preguntas, cortas)  # textos de respuestas cortas (siguen a su pregunta)
    textos, seguridad_actual = {}, {}
    for clave, valor in (respuestas or {}).items():
        if valor in ubicacion:
            i, j = ubicacion[valor]
            nueva_clave = str(i)
            textos[nueva_clave] = preguntas[i]["opciones"][j]
        elif str(clave) in movimientos:
            if movimientos[str(clave)] is None:
                continue  # la pregunta se borró
            nueva_clave = movimientos[str(clave)]
            textos[nueva_clave] = valor
        elif isinstance(valor, str) and valor.startswith(PREFIJO):
            continue
        else:
            nueva_clave = str(clave)
            textos[nueva_clave] = valor
        if clave in seguridad:
            seguridad_actual[nueva_clave] = seguridad[clave]
    return textos, seguridad_actual


def migrar(db: Session) -> dict:
    """
    Una vez por arranque (idempotente): asigna ids a los exámenes que no los tienen y convierte a ids
    las respuestas de texto de los intentos. Devuelve cuántos registros cambió.
    """
    from database import Cuestionario, Intento  # aquí para no crear un ciclo al importar

    cambios = {"cuestionarios": 0, "intentos": 0}
    examenes = {}
    for cuestionario in db.query(Cuestionario).all():
        preguntas = cuestionario.preguntas_json or []
        if not tiene_ids(preguntas):
            cuestionario.preguntas_json = normalizar_preguntas(preguntas)
            cambios["cuestionarios"] += 1
        examenes[cuestionario.id] = cuestionario.preguntas_json or []

    for intento in db.query(Intento).all():
        preguntas = examenes.get(intento.cuestionario_id)
        respuestas = intento.respuestas or {}
        if preguntas is None or all(isinstance(v, str) and v.startswith(PREFIJO) for v in respuestas.values()):
            continue
        # Solo se convierte lo que se puede; lo demás se conserva tal cual
        convertidas = copy.deepcopy(respuestas)
        for clave, valor in respuestas.items():
            if str(clave).isdigit() and int(clave) < len(preguntas):
                convertidas[clave] = id_de_respuesta(preguntas[int(clave)], valor) or valor
        if convertidas != respuestas:
            intento.respuestas = convertidas
            cambios["intentos"] += 1

    if cambios["cuestionarios"] or cambios["intentos"]:
        db.commit()
    return cambios
