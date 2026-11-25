"""3.2 Inicio del docente: resumen de entregas, borradores, clases y reportes."""


def resumen(cliente, encabezados):
    return cliente.get("/api/docente/resumen", headers=encabezados)


def entregar(cliente, encabezados, examen_id, respuestas):
    return cliente.post(f"/api/cuestionarios/{examen_id}/intentos", headers=encabezados, json={"respuestas": respuestas})


def test_docente_sin_nada(cliente, registrar):
    docente, _ = registrar("docente", "docente")
    datos = resumen(cliente, docente).json()
    assert datos == {"entregas_recientes": [], "borradores": [], "publicados_sin_clase": [],
                     "clases": [], "reportes": [], "total_examenes": 0}


def test_resumen_completo(cliente, registrar, crear_cuestionario, clase_con_examen):
    d = clase_con_examen
    docente_id = d["usuario_docente"]["id"]
    borrador = crear_cuestionario(docente_id, nombre="Borrador", publicado=False)
    huerfano = crear_cuestionario(docente_id, nombre="Sin clase", publicado=True)
    vacia = cliente.post("/api/clases", headers=d["docente"], json={"nombre": "Vacía"}).json()["clase"]

    # ana: mejor intento 10 (p1=a vale 1, p2=b vale 3 -> 4/4); beto: 0
    entregar(cliente, d["estudiante"], d["examen_id"], {"0": "b", "1": "a"})
    intento = entregar(cliente, d["estudiante"], d["examen_id"], {"0": "a", "1": "b"}).json()["intento"]
    beto, _ = registrar("beto")
    cliente.post("/api/clases/unirse", headers=beto, json={"codigo": d["clase"]["codigo"]})
    entregar(cliente, beto, d["examen_id"], {"0": "b", "1": "a"})
    cliente.post(f"/api/intentos/{intento['id']}/reportes", headers=d["estudiante"], json={"pregunta": 0, "motivo": "ambigua"})

    datos = resumen(cliente, d["docente"]).json()
    assert len(datos["entregas_recientes"]) == 3
    assert datos["entregas_recientes"][0]["nombre_estudiante"] == "beto"
    assert [b["id"] for b in datos["borradores"]] == [borrador]
    assert [p["id"] for p in datos["publicados_sin_clase"]] == [huerfano]
    clases = {c["nombre"]: c for c in datos["clases"]}
    assert clases["3A"]["num_estudiantes"] == 2 and clases["3A"]["entregas"] == 2
    assert clases["3A"]["promedio"] == 5.0  # (10 + 0) / 2 con el mejor intento de cada uno
    assert clases["Vacía"] == {"id": vacia["id"], "nombre": "Vacía", "codigo": vacia["codigo"],
                               "num_estudiantes": 0, "num_cuestionarios": 0, "entregas": 0, "promedio": None}
    assert datos["reportes"] == [{"cuestionario_id": d["examen_id"], "nombre_examen": "Examen", "reportes": 1}]
    assert datos["total_examenes"] == 3


def test_solo_datos_propios(cliente, registrar, clase_con_examen):
    d = clase_con_examen
    entregar(cliente, d["estudiante"], d["examen_id"], {"0": "a"})
    otro, _ = registrar("otro", "docente")
    datos = resumen(cliente, otro).json()
    assert datos["entregas_recientes"] == [] and datos["clases"] == []
    assert resumen(cliente, d["estudiante"]).status_code == 403
