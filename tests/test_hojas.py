"""4.10 Calificar con foto: endpoints de lectura y registro."""
from test_lectura_hojas import dibujar_hoja, fotografiar


def ruta(datos, parte):
    return f"/api/cuestionarios/{datos['examen_id']}/hojas/{parte}"


def test_estudiantes_de_las_clases_del_examen(cliente, clase_con_examen, registrar):
    registrar("fuera", "estudiante")  # no está en la clase
    datos = cliente.get(ruta(clase_con_examen, "estudiantes"), headers=clase_con_examen["docente"]).json()
    assert [e["nombre"] for e in datos["estudiantes"]] == ["ana"]
    assert datos["estudiantes"][0]["clases"] == ["3A"] and datos["estudiantes"][0]["intentos_en_papel"] == 0
    assert datos["se_puede_leer"] is True


def test_leer_una_foto(cliente, clase_con_examen):
    # El examen de prueba tiene 2 preguntas de 2 opciones; la clave es "a" y "b"
    foto = fotografiar(dibujar_hoja([2, 2], {0: 0, 1: 0}), angulo=5, perspectiva=0.03)
    respuesta = cliente.post(ruta(clase_con_examen, "leer"), headers=clase_con_examen["docente"],
                             files={"foto": ("hoja.jpg", foto, "image/jpeg")})
    assert respuesta.status_code == 200, respuesta.text
    datos = respuesta.json()
    assert [(r["pregunta"], r["indice"], r["opcion"], r["estado"]) for r in datos["lectura"]] == [(0, 0, "a", "ok"), (1, 0, "a", "ok")]
    assert datos["por_revisar"] == 0 and datos["vista"].startswith("data:image/jpeg")


def test_foto_ilegible(cliente, clase_con_examen):
    respuesta = cliente.post(ruta(clase_con_examen, "leer"), headers=clase_con_examen["docente"],
                             files={"foto": ("hoja.jpg", b"no es imagen", "image/jpeg")})
    assert respuesta.status_code == 422 and "imagen" in respuesta.json()["detail"]


def test_registrar_califica_y_lo_ve_el_estudiante(cliente, clase_con_examen):
    alumna = clase_con_examen["usuario_estudiante"]["id"]
    respuesta = cliente.post(ruta(clase_con_examen, "registrar"), headers=clase_con_examen["docente"],
                             json={"estudiante_id": alumna, "respuestas": {"0": 0, "1": 0}})
    assert respuesta.status_code == 200, respuesta.text
    intento = respuesta.json()["intento"]
    assert intento["en_papel"] is True and intento["puntos_obtenidos"] == 1 and intento["puntos_totales"] == 4
    assert intento["respuestas"] == {"0": "a", "1": "a"}

    mios = cliente.get("/api/intentos/mios", headers=clase_con_examen["estudiante"]).json()["data"]
    assert len(mios) == 1 and mios[0]["en_papel"] is True
    estudiantes = cliente.get(ruta(clase_con_examen, "estudiantes"), headers=clase_con_examen["docente"]).json()["estudiantes"]
    assert estudiantes[0]["intentos_en_papel"] == 1


def test_registrar_en_blanco_y_validaciones(cliente, clase_con_examen, registrar):
    docente = clase_con_examen["docente"]
    alumna = clase_con_examen["usuario_estudiante"]["id"]
    en_blanco = cliente.post(ruta(clase_con_examen, "registrar"), headers=docente,
                             json={"estudiante_id": alumna, "respuestas": {"0": None, "1": 1}}).json()["intento"]
    assert en_blanco["puntos_obtenidos"] == 3 and en_blanco["respuestas"] == {"1": "b"}

    _, externo = registrar("externo", "estudiante")
    assert cliente.post(ruta(clase_con_examen, "registrar"), headers=docente,
                        json={"estudiante_id": externo["id"], "respuestas": {}}).status_code == 400
    assert cliente.post(ruta(clase_con_examen, "registrar"), headers=docente,
                        json={"estudiante_id": alumna, "respuestas": {"0": 5}}).status_code == 400


def test_solo_el_docente_dueno(cliente, clase_con_examen, registrar):
    otro, _ = registrar("otro", "docente")
    assert cliente.get(ruta(clase_con_examen, "estudiantes"), headers=otro).status_code in (403, 404)
    assert cliente.get(ruta(clase_con_examen, "estudiantes"), headers=clase_con_examen["estudiante"]).status_code == 403
