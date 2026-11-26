"""4.1 Configuración del examen: fecha límite, tiempo límite, intentos máximos y cuándo ver las respuestas.

- Fecha límite: después de ella no se puede iniciar. Si se entrega tarde, el intento queda marcado.
- Tiempo límite: se mide en el servidor desde el InicioIntento, no con el reloj del navegador.
  Si el estudiante recarga la página, se reutiliza su inicio en curso (el reloj no se reinicia).
- Intentos máximos: cada inicio consume un intento (si contara solo al entregar, bastaría con ver
  el examen, abandonarlo y volver a empezar con el tiempo completo). Los intentos anteriores a
  esta versión, que no tienen inicio, también cuentan.
- Entregas tardías: se aceptan, con un margen de GRACIA_SEGUNDOS, pero quedan con
  `fuera_de_tiempo = True` para que el docente decida (un corte de red no hace perder el trabajo).
- Borrador: mientras responde, el navegador guarda sus respuestas en el inicio
  (PUT /api/inicios/{id}/borrador). Al retomar, POST .../inicio las devuelve traducidas a las
  preguntas actuales, así recargar la página o cambiar de dispositivo no las pierde.
- Entrega automática: si el tiempo (o la fecha límite) vence y el navegador no llegó a entregar
  (p. ej. sin conexión), el servidor entrega el borrador como intento con `entrega_automatica`.
  No hay tarea programada: `entregar_vencidos` corre cuando se consultan los intentos afectados.
  Solo se entregan inicios con respuestas guardadas y con un límite real; sin límite, un inicio
  abandonado no se entrega (el estudiante pudo no querer hacerlo).
- Mostrar respuestas: 'al_entregar' (predeterminado), 'despues_del_cierre' (requiere fecha límite;
  evita que las claves circulen mientras el examen sigue abierto) o 'nunca'. Cuando no se muestran,
  el estudiante ve su calificación pero no la clave, la retroalimentación ni la cita, y no puede
  generar la práctica adaptativa.
"""
import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

import clases
from auth import usuario_actual, docente_actual, cuestionario_propio
from database import Cuestionario, InicioIntento, Intento, Usuario, ahora_utc, get_db
from calificacion import calificar
import identificadores
import versiones
from identificadores import respuestas_a_ids, respuestas_para_mostrar

router = APIRouter(tags=["configuracion"])

GRACIA_SEGUNDOS = 60
VIGENCIA_SIN_LIMITE = datetime.timedelta(hours=3)  # un inicio sin tiempo límite se reutiliza hasta entonces


class Borrador(BaseModel):
    respuestas: dict[str, str] = Field(default_factory=dict)   # { "<índice>": id de opción (o texto) }
    seguridad: dict[str, bool] = Field(default_factory=dict)


class Configuracion(BaseModel):
    fecha_limite: datetime.datetime | None = None
    tiempo_limite: int | None = Field(default=None, ge=1, le=600)        # minutos
    intentos_maximos: int | None = Field(default=None, ge=1, le=50)
    mostrar_respuestas: Literal["al_entregar", "despues_del_cierre", "nunca"] = "al_entregar"
    barajar_preguntas: bool = False
    barajar_opciones: bool = False


def a_utc_sin_zona(fecha: datetime.datetime | None) -> datetime.datetime | None:
    """Fechas con zona -> UTC sin zona (como guarda la base). Sin zona se toman como UTC."""
    if fecha is None or fecha.tzinfo is None:
        return fecha
    return fecha.astimezone(datetime.timezone.utc).replace(tzinfo=None)


# ---------- Consultas ----------

def configuracion_a_dict(cuestionario: Cuestionario) -> dict:
    return {
        "fecha_limite": cuestionario.fecha_limite,
        "tiempo_limite": cuestionario.tiempo_limite,
        "intentos_maximos": cuestionario.intentos_maximos,
        "mostrar_respuestas": cuestionario.mostrar_respuestas or "al_entregar",
        "barajar_preguntas": bool(cuestionario.barajar_preguntas),
        "barajar_opciones": bool(cuestionario.barajar_opciones),
    }


def cerrado(cuestionario: Cuestionario, ahora: datetime.datetime) -> bool:
    return cuestionario.fecha_limite is not None and ahora > cuestionario.fecha_limite


def respuestas_visibles(cuestionario: Cuestionario, ahora: datetime.datetime) -> bool:
    modo = cuestionario.mostrar_respuestas or "al_entregar"
    if modo == "nunca":
        return False
    if modo == "despues_del_cierre":
        return cerrado(cuestionario, ahora)
    return True


def intentos_usados(estudiante_id: int, cuestionario_id: int, db: Session) -> int:
    inicios = db.query(func.count(InicioIntento.id)).filter(
        InicioIntento.estudiante_id == estudiante_id, InicioIntento.cuestionario_id == cuestionario_id
    ).scalar()
    antiguos = db.query(func.count(Intento.id)).filter(
        Intento.estudiante_id == estudiante_id, Intento.cuestionario_id == cuestionario_id, Intento.inicio_id.is_(None)
    ).scalar()
    return inicios + antiguos


def fin_del_inicio(cuestionario: Cuestionario, inicio: InicioIntento) -> datetime.datetime | None:
    """Hasta cuándo puede responder sin quedar fuera de tiempo (None = sin límite)."""
    limites = []
    if cuestionario.tiempo_limite:
        limites.append(inicio.fecha_inicio + datetime.timedelta(minutes=cuestionario.tiempo_limite))
    if cuestionario.fecha_limite:
        limites.append(cuestionario.fecha_limite)
    return min(limites) if limites else None


def segundos_restantes(cuestionario: Cuestionario, inicio: InicioIntento, ahora: datetime.datetime) -> int | None:
    fin = fin_del_inicio(cuestionario, inicio)
    return None if fin is None else max(0, int((fin - ahora).total_seconds()))


def inicio_en_curso(cuestionario: Cuestionario, estudiante_id: int, db: Session, ahora: datetime.datetime) -> InicioIntento | None:
    """El último inicio sin entregar que todavía se puede usar (para reanudar tras recargar la página)."""
    inicio = db.query(InicioIntento).filter(
        InicioIntento.cuestionario_id == cuestionario.id,
        InicioIntento.estudiante_id == estudiante_id,
        InicioIntento.entregado == False,  # noqa: E712
    ).order_by(InicioIntento.fecha_inicio.desc()).first()
    if inicio is None:
        return None
    fin = fin_del_inicio(cuestionario, inicio)
    vigente_hasta = fin if fin is not None else inicio.fecha_inicio + VIGENCIA_SIN_LIMITE
    return inicio if ahora <= vigente_hasta else None


def estado_estudiante(cuestionario: Cuestionario, estudiante_id: int, db: Session, ahora: datetime.datetime) -> dict:
    """Lo que el panel del estudiante necesita para mostrar el examen y habilitar "Iniciar"."""
    usados = intentos_usados(estudiante_id, cuestionario.id, db)
    restantes = None if cuestionario.intentos_maximos is None else max(0, cuestionario.intentos_maximos - usados)
    en_curso = inicio_en_curso(cuestionario, estudiante_id, db, ahora)
    motivo = None
    if en_curso is None:
        if cerrado(cuestionario, ahora):
            motivo = "cerrado"
        elif restantes == 0:
            motivo = "sin_intentos"
    return {
        "intentos_usados": usados,
        "intentos_restantes": restantes,
        "en_curso": en_curso is not None,
        "puede_iniciar": motivo is None,
        "motivo": motivo,
        "respuestas_visibles": respuestas_visibles(cuestionario, ahora),
    }


def evaluar_entrega(cuestionario: Cuestionario, inicio: InicioIntento | None, ahora: datetime.datetime) -> tuple[int | None, bool]:
    """(segundos que tardó medidos en el servidor o None, si se entregó fuera de tiempo)."""
    gracia = datetime.timedelta(seconds=GRACIA_SEGUNDOS)
    if inicio is None:
        tarde = cuestionario.fecha_limite is not None and ahora > cuestionario.fecha_limite + gracia
        return None, tarde
    fin = fin_del_inicio(cuestionario, inicio)
    return int((ahora - inicio.fecha_inicio).total_seconds()), fin is not None and ahora > fin + gracia


def requiere_inicio(cuestionario: Cuestionario) -> bool:
    """Con tiempo límite o intentos máximos, entregar sin un inicio registrado no se permite."""
    return bool(cuestionario.tiempo_limite or cuestionario.intentos_maximos)


def borrador_a_dict(cuestionario: Cuestionario, inicio: InicioIntento) -> dict | None:
    """Las respuestas guardadas, con los índices y textos actuales de las preguntas (ver 4.6)."""
    if not inicio.borrador_respuestas and not inicio.borrador_seguridad:
        return None
    respuestas, seguridad = respuestas_para_mostrar(
        cuestionario.preguntas_json or [], inicio.borrador_respuestas or {}, inicio.borrador_seguridad or {}
    )
    return {"respuestas": respuestas, "seguridad": seguridad, "fecha": inicio.fecha_borrador}


def entregar_vencidos(db: Session, *, cuestionario_ids: list[int] | None = None, estudiante_id: int | None = None) -> int:
    """
    Convierte en intentos los inicios vencidos que tienen respuestas guardadas y no se entregaron.
    Se califican con las preguntas actuales; la fecha y el tiempo son los del vencimiento.
    Devuelve cuántos entregó (hace commit si entregó alguno).
    """
    if cuestionario_ids is not None and not cuestionario_ids:
        return 0
    consulta = db.query(InicioIntento).filter(
        InicioIntento.entregado == False,  # noqa: E712
        InicioIntento.borrador_respuestas.isnot(None),
    )
    if cuestionario_ids is not None:
        consulta = consulta.filter(InicioIntento.cuestionario_id.in_(cuestionario_ids))
    if estudiante_id is not None:
        consulta = consulta.filter(InicioIntento.estudiante_id == estudiante_id)

    ahora = ahora_utc()
    gracia = datetime.timedelta(seconds=GRACIA_SEGUNDOS)
    examenes: dict[int, Cuestionario | None] = {}
    entregados = 0
    for inicio in consulta.all():
        if not inicio.borrador_respuestas:
            continue
        if inicio.cuestionario_id not in examenes:
            examenes[inicio.cuestionario_id] = db.query(Cuestionario).filter(Cuestionario.id == inicio.cuestionario_id).first()
        cuestionario = examenes[inicio.cuestionario_id]
        fin = cuestionario and fin_del_inicio(cuestionario, inicio)
        if fin is None or ahora <= fin + gracia:
            continue  # sin límite, o el navegador todavía puede entregar
        preguntas = cuestionario.preguntas_json or []
        respuestas = respuestas_a_ids(preguntas, inicio.borrador_respuestas)  # textos antiguos -> ids
        import respuestas_cortas  # aquí: evita un ciclo al importar
        cortas = respuestas_cortas.corregir(preguntas, respuestas)  # 4.2
        resultado = calificar(preguntas, respuestas, cortas)
        db.add(Intento(
            cuestionario_id=cuestionario.id,
            estudiante_id=inicio.estudiante_id,
            respuestas=respuestas,
            puntos_obtenidos=resultado["puntos_obtenidos"],
            puntos_totales=resultado["puntos_totales"],
            calificacion=resultado["calificacion"],
            tiempo_segundos=max(0, int((fin - inicio.fecha_inicio).total_seconds())),
            fecha=fin,
            seguridad={k: v for k, v in (inicio.borrador_seguridad or {}).items() if k in respuestas},
            respuestas_cortas=cortas,
            inicio_id=inicio.id,
            orden=inicio.orden,
            fuera_de_tiempo=False,  # el borrador solo se acepta mientras hay tiempo
            entrega_automatica=True,
        ))
        inicio.entregado = True
        entregados += 1
    if entregados:
        db.commit()
        print(f"INFO entrega automática: {entregados} intento(s) vencido(s) con respuestas guardadas")
    return entregados


def borrar_inicios(db: Session, *, cuestionario_ids: list[int] | None = None, estudiante_id: int | None = None):
    consulta = db.query(InicioIntento)
    if cuestionario_ids is not None:
        consulta = consulta.filter(InicioIntento.cuestionario_id.in_(cuestionario_ids))
    if estudiante_id is not None:
        consulta = consulta.filter(InicioIntento.estudiante_id == estudiante_id)
    consulta.delete(synchronize_session=False)


# ---------- Endpoints ----------

@router.put("/api/cuestionarios/{cuestionario_id}/configuracion")
def configurar(cuestionario_id: int, datos: Configuracion, usuario: Usuario = Depends(docente_actual), db: Session = Depends(get_db)):
    cuestionario = cuestionario_propio(cuestionario_id, usuario, db)
    if datos.mostrar_respuestas == "despues_del_cierre" and datos.fecha_limite is None:
        raise HTTPException(status_code=400, detail="Para mostrar las respuestas después del cierre, el examen necesita una fecha límite.")
    cuestionario.fecha_limite = a_utc_sin_zona(datos.fecha_limite)
    cuestionario.tiempo_limite = datos.tiempo_limite
    cuestionario.intentos_maximos = datos.intentos_maximos
    cuestionario.mostrar_respuestas = datos.mostrar_respuestas
    cuestionario.barajar_preguntas = datos.barajar_preguntas
    cuestionario.barajar_opciones = datos.barajar_opciones
    db.commit()
    return {"configuracion": configuracion_a_dict(cuestionario)}


@router.post("/api/cuestionarios/{cuestionario_id}/inicio")
def iniciar(cuestionario_id: int, usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    """
    El estudiante empieza (o retoma) la evaluación. Devuelve el inicio y los segundos que le quedan
    (None si no hay límite). Se rechaza si el examen cerró o no le quedan intentos.
    """
    if usuario.rol != "estudiante":
        raise HTTPException(status_code=403, detail="Solo los estudiantes pueden iniciar evaluaciones.")
    cuestionario = db.query(Cuestionario).filter(Cuestionario.id == cuestionario_id).first()
    if cuestionario is None or not cuestionario.publicado or not clases.estudiante_puede_ver(usuario.id, cuestionario.id, db):
        raise HTTPException(status_code=404, detail="Esta evaluación ya no está disponible.")

    entregar_vencidos(db, cuestionario_ids=[cuestionario.id], estudiante_id=usuario.id)
    ahora = ahora_utc()
    inicio = inicio_en_curso(cuestionario, usuario.id, db, ahora)
    reanudado = inicio is not None
    if inicio is None:
        estado = estado_estudiante(cuestionario, usuario.id, db, ahora)
        if estado["motivo"] == "cerrado":
            raise HTTPException(status_code=403, detail="Esta evaluación ya cerró: la fecha límite pasó.")
        if estado["motivo"] == "sin_intentos":
            raise HTTPException(status_code=403, detail="Ya usaste todos los intentos de esta evaluación.")
        # 2.6: el orden de este intento (None si el examen no baraja)
        orden = versiones.generar_orden(cuestionario.preguntas_json or [], bool(cuestionario.barajar_preguntas), bool(cuestionario.barajar_opciones))
        inicio = InicioIntento(cuestionario_id=cuestionario.id, estudiante_id=usuario.id, fecha_inicio=ahora, orden=orden)
        db.add(inicio)
        db.commit()
        db.refresh(inicio)

    # Las preguntas necesitan ids para entregar por opción (4.6); los exámenes antiguos se completan aquí
    if not identificadores.tiene_ids(cuestionario.preguntas_json):
        cuestionario.preguntas_json = identificadores.normalizar_preguntas(cuestionario.preguntas_json)
        db.commit()

    return {
        "inicio_id": inicio.id,
        "fecha_inicio": inicio.fecha_inicio,
        "reanudado": reanudado,
        "segundos_transcurridos": int((ahora - inicio.fecha_inicio).total_seconds()),
        "segundos_restantes": segundos_restantes(cuestionario, inicio, ahora),
        "configuracion": configuracion_a_dict(cuestionario),
        # Al retomar: lo que ya había respondido (índices originales)
        "borrador": borrador_a_dict(cuestionario, inicio) if reanudado else None,
        # El examen tal como lo verá en este intento: sin clave y, si se baraja, en su orden (2.6)
        "preguntas_json": versiones.preguntas_en_orden(cuestionario.preguntas_json or [], inicio.orden),
    }


@router.put("/api/inicios/{inicio_id}/borrador")
def guardar_borrador(inicio_id: int, datos: Borrador, usuario: Usuario = Depends(usuario_actual), db: Session = Depends(get_db)):
    """
    Guarda las respuestas en curso (el navegador lo hace solo, unos segundos después de cada cambio).
    Se rechaza si el intento ya se entregó o si ya no se puede retomar (se acabó el tiempo).
    """
    inicio = db.query(InicioIntento).filter(InicioIntento.id == inicio_id).first()
    if inicio is None or inicio.estudiante_id != usuario.id:
        raise HTTPException(status_code=404, detail="No se encontró este intento.")
    if inicio.entregado:
        raise HTTPException(status_code=409, detail="Este intento ya se entregó.")
    cuestionario = db.query(Cuestionario).filter(Cuestionario.id == inicio.cuestionario_id).first()
    ahora = ahora_utc()
    if cuestionario is None or inicio_en_curso(cuestionario, usuario.id, db, ahora) is not inicio:
        raise HTTPException(status_code=409, detail="El tiempo de este intento terminó.")

    preguntas = cuestionario.preguntas_json or []
    respuestas = respuestas_a_ids(preguntas, datos.respuestas)
    inicio.borrador_respuestas = respuestas
    inicio.borrador_seguridad = {k: v for k, v in datos.seguridad.items() if k in respuestas}
    inicio.fecha_borrador = ahora
    db.commit()
    return {"fecha": inicio.fecha_borrador}
