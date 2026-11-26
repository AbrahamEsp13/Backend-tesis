"""4.2 Respuestas cortas: calificación con la IA y corrección del docente.

- Al entregar, todas las respuestas cortas del intento se califican en UNA llamada a la IA, contra la
  respuesta modelo y el criterio (la justificación). No gasta créditos del docente.
- Resultado por pregunta, en `Intento.respuestas_cortas`:
    { "<índice>": {"texto", "puntaje": 0|0.5|1, "comentario", "estado"} }
  estado: "ia" (la calificó la IA), "docente" (la calificó o corrigió el docente), "vacia" (sin respuesta)
  o "pendiente" (la IA no respondió: vale 0 hasta que el docente la califique).
- El docente puede cambiar el puntaje desde la revisión del intento; la calificación se recalcula.
- El texto escrito también queda en `Intento.respuestas` (como valor que no es id de opción).
"""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth import docente_actual
from calificacion import calificar, es_respuesta_corta
from database import Cuestionario, Intento, Usuario, get_db

router = APIRouter(tags=["respuestas_cortas"])

MAX_TEXTO = 1000


class CorreccionDocente(BaseModel):
    puntaje: Literal[0, 0.5, 1]
    comentario: str | None = Field(default=None, max_length=600)


def textos_cortos(preguntas: list, respuestas: dict) -> dict:
    """Las respuestas escritas (recortadas) de las preguntas de respuesta corta."""
    return {
        str(i): str(respuestas.get(str(i)) or "").strip()[:MAX_TEXTO]
        for i, pregunta in enumerate(preguntas) if es_respuesta_corta(pregunta)
    }


def corregir(preguntas: list, respuestas: dict) -> dict:
    """Califica con la IA las respuestas cortas del intento (si hay). Nunca lanza: si falla, quedan pendientes."""
    import generacion  # aquí: así las pruebas pueden reemplazar la función de la IA
    textos = textos_cortos(preguntas, respuestas)
    resultado, por_calificar = {}, []
    for clave, texto in textos.items():
        if not texto:
            resultado[clave] = {"texto": "", "puntaje": 0, "comentario": "No respondiste esta pregunta.", "estado": "vacia"}
        else:
            pregunta = preguntas[int(clave)]
            por_calificar.append({
                "numero": int(clave) + 1, "pregunta": pregunta.get("pregunta"),
                "referencia": pregunta.get("respuesta_correcta"), "criterio": pregunta.get("justificacion_pedagogica"),
                "respuesta": texto,
            })
    if por_calificar:
        try:
            calificadas = generacion.calificar_respuestas_cortas(por_calificar)
        except Exception as e:
            print(f"ERROR al calificar respuestas cortas: {e}")
            calificadas = {}
        for item in por_calificar:
            clave = str(item["numero"] - 1)
            calificacion = calificadas.get(item["numero"])
            if calificacion:
                resultado[clave] = {"texto": item["respuesta"], **calificacion, "estado": "ia"}
            else:
                resultado[clave] = {"texto": item["respuesta"], "puntaje": 0, "comentario": "", "estado": "pendiente"}
        pendientes = sum(r["estado"] == "pendiente" for r in resultado.values())
        print(f"INFO respuestas cortas: {len(por_calificar)} calificadas con IA, {pendientes} pendientes")
    return resultado


def sin_correccion(cortas: dict | None) -> dict:
    """Para el estudiante cuando el docente no muestra las respuestas: solo lo que escribió."""
    return {clave: {"texto": valor.get("texto", ""), "estado": "oculta"} for clave, valor in (cortas or {}).items()}


@router.put("/api/intentos/{intento_id}/respuestas-cortas/{indice}")
def corregir_docente(intento_id: int, indice: int, datos: CorreccionDocente,
                     usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    """El docente califica o corrige una respuesta corta; se recalcula la calificación del intento."""
    from main import intento_a_dict  # aquí: main importa este módulo
    intento = db.query(Intento).filter(Intento.id == intento_id).first()
    cuestionario = intento and db.query(Cuestionario).filter(Cuestionario.id == intento.cuestionario_id).first()
    if intento is None or cuestionario is None or cuestionario.usuario_id != usuario.id:
        raise HTTPException(status_code=404, detail="Intento no encontrado.")
    preguntas = cuestionario.preguntas_json or []
    if not 0 <= indice < len(preguntas) or not es_respuesta_corta(preguntas[indice]):
        raise HTTPException(status_code=400, detail="Esa pregunta no es de respuesta corta.")

    cortas = dict(intento.respuestas_cortas or {})
    anterior = cortas.get(str(indice)) or {"texto": str((intento.respuestas or {}).get(str(indice)) or "")}
    cortas[str(indice)] = {
        **anterior,
        "puntaje": datos.puntaje,
        "comentario": (datos.comentario or "").strip() if datos.comentario is not None else anterior.get("comentario", ""),
        "estado": "docente",
    }
    resultado = calificar(preguntas, intento.respuestas or {}, cortas)
    intento.respuestas_cortas = cortas  # dict nuevo: SQLAlchemy detecta el cambio del JSON
    intento.puntos_obtenidos = resultado["puntos_obtenidos"]
    intento.puntos_totales = resultado["puntos_totales"]
    intento.calificacion = resultado["calificacion"]
    db.commit()
    db.refresh(intento)
    return {"intento": intento_a_dict(intento, preguntas)}
