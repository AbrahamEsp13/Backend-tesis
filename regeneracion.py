"""4.3 Regenerar una sola pregunta desde el editor.

- El docente pide otra versión de una pregunta (opcionalmente con un motivo y una indicación).
- El contexto es el fragmento del PDF del que salió (`fuente.fragmento`, o la cita si es una
  pregunta anterior a ese campo); si no tiene fuente, la pregunta misma con su clave y justificación.
- No guarda nada: devuelve la pregunta nueva y el editor la propone ("Usar esta" / "Descartar").
  Así no se mezcla con cambios sin guardar ni depende de la posición guardada de la pregunta.
- Cuesta menos que una generación completa: no gasta créditos; tiene un cupo diario propio
  (REGENERACIONES_DIARIAS) que se reinicia junto con los créditos.
- La pregunta nueva queda con `origen: "regenerada"` (cuenta como generada en la calidad del
  generador) y conserva el puntaje de la original.
"""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth import docente_actual, cuestionario_propio
from database import Usuario, get_db
from generacion import regenerar_pregunta, agregar_fuentes, RespuestaInvalida, COMPLEJIDAD_PREDETERMINADA

router = APIRouter(tags=["regeneracion"])

REGENERACIONES_DIARIAS = 10

Motivo = Literal["ambigua", "demasiado_facil", "demasiado_dificil", "fuera_del_tema", "clave_incorrecta", "otro"]

INDICACIONES_MOTIVO = {
    "ambigua": "La pregunta anterior era ambigua: redacta un enunciado preciso con una sola respuesta defendible.",
    "demasiado_facil": "La pregunta anterior era demasiado fácil: exige comprender o aplicar, con distractores plausibles.",
    "demasiado_dificil": "La pregunta anterior era demasiado difícil: hazla más directa y clara, sin trampas.",
    "fuera_del_tema": "La pregunta anterior se salía del tema: cíñete estrictamente a lo que dice el fragmento.",
    "clave_incorrecta": "La respuesta marcada como correcta en la pregunta anterior era dudosa: asegúrate de que la clave sea indiscutible según el texto.",
    "otro": "",
}


class PreguntaEditor(BaseModel):
    """La pregunta tal como está en el editor (puede tener cambios sin guardar)."""
    tipo: str | None = None  # 4.2: la nueva es del mismo tipo
    pregunta: str = Field(max_length=4000)
    opciones: list[str] = Field(default_factory=list, max_length=10)
    respuesta_correcta: str | None = None
    justificacion_pedagogica: str | None = Field(default=None, max_length=4000)
    puntuacion: float | None = None
    fuente: dict | None = None


class Regeneracion(BaseModel):
    pregunta: PreguntaEditor
    motivo: Motivo | None = None
    indicaciones: str | None = Field(default=None, max_length=300)


def fragmento_de(pregunta: PreguntaEditor) -> dict:
    """El fragmento numerado (como los del RAG) que sirve de contexto."""
    fuente = pregunta.fuente or {}
    texto = str(fuente.get("fragmento") or fuente.get("texto") or "").strip()[:6000]
    if texto:
        return {"numero": 1, "texto": texto, "pagina": fuente.get("pagina"), "pagina_fin": fuente.get("pagina_fin")}
    texto = "\n".join(filter(None, [
        pregunta.pregunta,
        f"Respuesta correcta: {pregunta.respuesta_correcta}" if pregunta.respuesta_correcta else None,
        pregunta.justificacion_pedagogica,
    ]))
    return {"numero": 1, "texto": texto, "pagina": None, "pagina_fin": None}


def regeneraciones_restantes(usuario: Usuario) -> int:
    """Supone que el cupo del día ya se reinició (main.regenerar_creditos)."""
    return max(0, REGENERACIONES_DIARIAS - (usuario.regeneraciones_usadas or 0))


# "def" y no "async def": la llamada a Gemini tarda segundos y no debe bloquear a los demás.
@router.post("/api/cuestionarios/{cuestionario_id}/regenerar-pregunta")
def regenerar(cuestionario_id: int, datos: Regeneracion, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    from main import regenerar_creditos  # aquí: main importa este módulo
    cuestionario = cuestionario_propio(cuestionario_id, usuario, db)
    regenerar_creditos(usuario, db)  # reinicia créditos y regeneraciones si cambió el día
    if regeneraciones_restantes(usuario) <= 0:
        raise HTTPException(status_code=429, detail=f"Ya usaste tus {REGENERACIONES_DIARIAS} regeneraciones de hoy. Mañana podrás regenerar más.")

    fragmento = fragmento_de(datos.pregunta)
    indicaciones = " ".join(filter(None, [
        INDICACIONES_MOTIVO.get(datos.motivo or "otro", ""),
        f"Indicación del docente: {datos.indicaciones.strip()}" if datos.indicaciones and datos.indicaciones.strip() else None,
    ]))
    original = {
        "tipo": datos.pregunta.tipo,
        "pregunta": datos.pregunta.pregunta,
        "opciones": datos.pregunta.opciones,
        "respuesta_correcta": datos.pregunta.respuesta_correcta,
    }
    try:
        nuevas = regenerar_pregunta(
            f"[Fragmento 1]\n{fragmento['texto']}",
            original,
            cuestionario.complejidad or COMPLEJIDAD_PREDETERMINADA,
            indicaciones,
        )
    except RespuestaInvalida as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        print(f"ERROR al regenerar la pregunta: {e}")
        raise HTTPException(status_code=502, detail="El servicio de IA no respondió correctamente. Intenta de nuevo en unos minutos.")

    nueva = agregar_fuentes(nuevas[:1], [fragmento])[0]
    if nueva.get("fuente") and nueva["fuente"].get("pagina") is None:
        nueva.pop("fuente")  # el contexto no venía del PDF
    nueva["puntuacion"] = datos.pregunta.puntuacion or 1
    nueva["origen"] = "regenerada"

    usuario.regeneraciones_usadas = (usuario.regeneraciones_usadas or 0) + 1
    db.commit()
    print(f"INFO regeneración: motivo={datos.motivo or '-'}, con fuente={'fuente' in nueva}")
    return {"pregunta": nueva, "regeneraciones_restantes": regeneraciones_restantes(usuario)}
