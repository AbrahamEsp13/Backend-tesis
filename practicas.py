"""2.3 Práctica adaptativa: "Practicar lo que fallé".

Después de entregar, el estudiante pide una práctica sin calificación generada SOLO con los
fragmentos del documento de las preguntas que falló (la `fuente` que guardó la 2.1).

- Contexto: por cada pregunta fallada, el fragmento completo (`fuente.fragmento`), o la cita
  si la pregunta es anterior a ese campo; si no tiene fuente (p. ej. pregunta manual), la
  pregunta con su clave y su justificación. Los fragmentos repetidos se usan una sola vez.
- Límite propio del estudiante: PRACTICAS_DIARIAS por día (no gasta los créditos del docente).
- La práctica trae la clave: el frontend corrige cada pregunta al instante. Al terminar se
  guardan las respuestas y los aciertos.
- El docente no ve las prácticas, solo cuántas terminó cada estudiante (en Resultados).
"""
import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from auth import usuario_actual
from database import Cuestionario, Intento, Practica, Usuario, ahora_utc, get_db
from generacion import generar_practica, agregar_fuentes, RespuestaInvalida
from identificadores import respuestas_para_mostrar
import configuracion

router = APIRouter(tags=["practicas"])

PRACTICAS_DIARIAS = 3
PREGUNTAS_POR_ERROR = 2
MIN_PREGUNTAS = 3
MAX_PREGUNTAS = 8


class RespuestasPractica(BaseModel):
    respuestas: dict[str, str]


# ---------- Lógica (sin FastAPI, para probarla directo) ----------

def preguntas_falladas(preguntas: list, respuestas: dict) -> list[int]:
    """Índices de las preguntas mal respondidas o sin responder."""
    return [i for i, p in enumerate(preguntas) if respuestas.get(str(i)) != p.get("respuesta_correcta")]


def numero_de_preguntas(n_falladas: int) -> int:
    return max(MIN_PREGUNTAS, min(MAX_PREGUNTAS, n_falladas * PREGUNTAS_POR_ERROR))


def fragmentos_de_errores(preguntas: list, indices: list[int]) -> list[dict]:
    """
    Fragmentos numerados para el prompt, como los del RAG: [{numero, texto, pagina, pagina_fin}].
    `pagina` es None cuando la pregunta no tenía fuente (el texto sale de la pregunta misma).
    """
    fragmentos, vistos = [], set()
    for i in indices:
        pregunta = preguntas[i]
        fuente = pregunta.get("fuente") or {}
        texto = (fuente.get("fragmento") or fuente.get("texto") or "").strip()
        if texto:
            pagina, pagina_fin = fuente.get("pagina"), fuente.get("pagina_fin")
        else:
            texto = "\n".join(filter(None, [
                pregunta.get("pregunta"),
                f"Respuesta correcta: {pregunta.get('respuesta_correcta')}",
                pregunta.get("justificacion_pedagogica"),
            ]))
            pagina = pagina_fin = None
        if texto in vistos:
            continue
        vistos.add(texto)
        fragmentos.append({"numero": len(fragmentos) + 1, "texto": texto, "pagina": pagina, "pagina_fin": pagina_fin})
    return fragmentos


def construir_contexto_practica(fragmentos: list[dict]) -> str:
    return "\n\n".join(f"[Fragmento {f['numero']}]\n{f['texto']}" for f in fragmentos)


def errores_para_prompt(preguntas: list, indices: list[int], respuestas: dict) -> list[dict]:
    return [
        {"pregunta": preguntas[i].get("pregunta"), "respuesta_correcta": preguntas[i].get("respuesta_correcta"),
         "elegida": respuestas.get(str(i))}
        for i in indices
    ]


def con_fuentes_validas(preguntas: list[dict], fragmentos: list[dict]) -> list[dict]:
    """Como agregar_fuentes, pero sin fuente cuando el fragmento no viene del PDF (no tiene página)."""
    resultado = []
    for pregunta in agregar_fuentes(preguntas, fragmentos):
        if pregunta.get("fuente") and pregunta["fuente"].get("pagina") is None:
            pregunta.pop("fuente")
        resultado.append(pregunta)
    return resultado


def inicio_de_hoy() -> datetime.datetime:
    return datetime.datetime.combine(ahora_utc().date(), datetime.time.min)


def practicas_restantes(estudiante_id: int, db: Session) -> int:
    hechas_hoy = db.query(func.count(Practica.id)).filter(
        Practica.estudiante_id == estudiante_id, Practica.fecha >= inicio_de_hoy()
    ).scalar()
    return max(0, PRACTICAS_DIARIAS - hechas_hoy)


def practica_a_dict(practica: Practica) -> dict:
    return {
        "id": practica.id,
        "intento_id": practica.intento_id,
        "cuestionario_id": practica.cuestionario_id,
        "preguntas_json": practica.preguntas_json,
        "preguntas_origen": practica.preguntas_origen,
        "respuestas": practica.respuestas,
        "aciertos": practica.aciertos,
        "fecha": practica.fecha,
        "fecha_terminada": practica.fecha_terminada,
    }


def practicas_terminadas_por_estudiante(cuestionario_id: int, db: Session) -> dict[str, int]:
    """Para el docente: { "<estudiante_id>": prácticas terminadas de este examen }."""
    filas = db.query(Practica.estudiante_id, func.count(Practica.id)).filter(
        Practica.cuestionario_id == cuestionario_id, Practica.fecha_terminada.isnot(None)
    ).group_by(Practica.estudiante_id).all()
    return {str(estudiante): total for estudiante, total in filas}


def borrar_practicas(db: Session, *, cuestionario_ids: list[int] | None = None, estudiante_id: int | None = None):
    consulta = db.query(Practica)
    if cuestionario_ids is not None:
        consulta = consulta.filter(Practica.cuestionario_id.in_(cuestionario_ids))
    if estudiante_id is not None:
        consulta = consulta.filter(Practica.estudiante_id == estudiante_id)
    consulta.delete(synchronize_session=False)


def _solo_estudiante(usuario: Usuario):
    if usuario.rol != "estudiante":
        raise HTTPException(status_code=403, detail="Las prácticas son solo para estudiantes.")


def _practica_propia(practica_id: int, usuario: Usuario, db: Session) -> Practica:
    practica = db.query(Practica).filter(Practica.id == practica_id).first()
    if practica is None or practica.estudiante_id != usuario.id:
        raise HTTPException(status_code=404, detail="Práctica no encontrada.")
    return practica


# ---------- Endpoints ----------

@router.get("/api/practicas/cupo")
def cupo(usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    _solo_estudiante(usuario)
    return {"restantes": practicas_restantes(usuario.id, db), "limite": PRACTICAS_DIARIAS}


# "def" y no "async def": la llamada a Gemini tarda segundos y no debe bloquear a los demás.
@router.post("/api/intentos/{intento_id}/practica")
def crear_practica(intento_id: int, usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    _solo_estudiante(usuario)
    intento = db.query(Intento).filter(Intento.id == intento_id).first()
    if intento is None or intento.estudiante_id != usuario.id:
        raise HTTPException(status_code=404, detail="Intento no encontrado.")
    cuestionario = db.query(Cuestionario).filter(Cuestionario.id == intento.cuestionario_id).first()
    if cuestionario is None:
        raise HTTPException(status_code=404, detail="Esta evaluación ya no existe.")

    if not configuracion.respuestas_visibles(cuestionario, ahora_utc()):
        raise HTTPException(status_code=403, detail="Podrás practicar cuando tu docente muestre las respuestas de esta evaluación.")

    preguntas = cuestionario.preguntas_json or []
    # Los ids guardados, traducidos a los textos actuales (4.6): se comparan con respuesta_correcta
    respuestas, _ = respuestas_para_mostrar(preguntas, intento.respuestas)
    falladas = preguntas_falladas(preguntas, respuestas)
    if not falladas:
        raise HTTPException(status_code=400, detail="No fallaste ninguna pregunta en este intento: no hay nada que practicar.")
    if practicas_restantes(usuario.id, db) <= 0:
        raise HTTPException(
            status_code=429,
            detail=f"Ya hiciste tus {PRACTICAS_DIARIAS} prácticas de hoy. Mañana podrás generar más.",
        )

    fragmentos = fragmentos_de_errores(preguntas, falladas)
    try:
        nuevas = generar_practica(
            construir_contexto_practica(fragmentos),
            numero_de_preguntas(len(falladas)),
            errores_para_prompt(preguntas, falladas, respuestas),
        )
    except RespuestaInvalida as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        print(f"ERROR al generar la práctica: {e}")
        raise HTTPException(status_code=502, detail="El servicio de IA no respondió correctamente. Intenta de nuevo en unos minutos.")
    nuevas = con_fuentes_validas(nuevas, fragmentos)

    practica = Practica(
        estudiante_id=usuario.id,
        intento_id=intento.id,
        cuestionario_id=cuestionario.id,
        preguntas_json=nuevas,
        preguntas_origen=falladas,
    )
    db.add(practica)
    db.commit()
    db.refresh(practica)
    print(f"INFO práctica: {len(nuevas)} preguntas a partir de {len(falladas)} errores ({len(fragmentos)} fragmentos)")
    return {"practica": practica_a_dict(practica), "restantes": practicas_restantes(usuario.id, db)}


@router.get("/api/practicas/{practica_id}")
def obtener_practica(practica_id: int, usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    practica = _practica_propia(practica_id, usuario, db)
    cuestionario = db.query(Cuestionario).filter(Cuestionario.id == practica.cuestionario_id).first()
    return {
        "practica": practica_a_dict(practica),
        "nombre_examen": cuestionario and (cuestionario.nombre_examen or cuestionario.nombre_documento),
        "restantes": practicas_restantes(usuario.id, db),
    }


@router.put("/api/practicas/{practica_id}/respuestas")
def terminar_practica(practica_id: int, datos: RespuestasPractica, usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    """Guarda las respuestas al terminar. Una práctica terminada ya no cambia."""
    practica = _practica_propia(practica_id, usuario, db)
    if practica.fecha_terminada is None:
        preguntas = practica.preguntas_json or []
        respuestas = {k: v for k, v in datos.respuestas.items() if k.isdigit() and int(k) < len(preguntas)}
        practica.respuestas = respuestas
        practica.aciertos = len(preguntas) - len(preguntas_falladas(preguntas, respuestas))
        practica.fecha_terminada = ahora_utc()
        db.commit()
        db.refresh(practica)
    return {"practica": practica_a_dict(practica)}
