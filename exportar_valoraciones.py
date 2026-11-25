"""Exporta la calidad del generador de TODOS los docentes (datos para el capítulo de resultados).

Uso (con la DATABASE_URL del .env):
    python exportar_valoraciones.py                  # resumen por complejidad en la consola
    python exportar_valoraciones.py preguntas.csv    # además, una fila por pregunta generada

El CSV no incluye nombres ni correos: solo ids de examen, complejidad y datos de cada pregunta.
"""
import csv
import sys

from database import Cuestionario, SessionLocal
from valoraciones import SIN_COMPLEJIDAD, es_generada, estadisticas


def porcentaje(parte: int, total: int) -> str:
    return f"{parte / total * 100:.1f}%" if total else "-"


def imprimir(nombre: str, d: dict):
    print(f"\n[{nombre}]")
    print(f"  Preguntas generadas: {d['generadas']}")
    print(f"  Con cita al documento: {d['con_fuente']} ({porcentaje(d['con_fuente'], d['generadas'])}),"
          f" literal: {d['con_cita_literal']} ({porcentaje(d['con_cita_literal'], d['generadas'])})")
    print(f"  Valoradas: {d['valoradas']} ({porcentaje(d['valoradas'], d['generadas'])})")
    print(f"  Útiles: {d['utiles']} ({porcentaje(d['utiles'], d['valoradas'])} de las valoradas)")
    print(f"  No útiles: {d['no_utiles']}")
    for motivo, veces in d["motivos"].items():
        if veces:
            print(f"    - {motivo}: {veces}")


def exportar_csv(cuestionarios, ruta: str):
    with open(ruta, "w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["cuestionario_id", "complejidad", "pregunta", "con_fuente", "cita_literal", "util", "motivo"])
        for c in cuestionarios:
            for indice, p in enumerate(c.preguntas_json or []):
                if not isinstance(p, dict) or not es_generada(p):
                    continue
                fuente = p.get("fuente") or {}
                valoracion = p.get("valoracion") or {}
                util = valoracion.get("util")
                escritor.writerow([
                    c.id, c.complejidad or SIN_COMPLEJIDAD, indice, int(bool(fuente)), int(bool(fuente.get("literal"))),
                    "" if util is None else int(util), valoracion.get("motivo") or "",
                ])
    print(f"\nCSV guardado en {ruta}")


def main():
    sesion = SessionLocal()
    try:
        cuestionarios = sesion.query(Cuestionario).all()
        resumen = estadisticas(cuestionarios)
        imprimir("TOTAL", resumen["total"])
        for complejidad, datos in sorted(resumen["por_complejidad"].items()):
            imprimir(complejidad, datos)
        if len(sys.argv) > 1:
            exportar_csv(cuestionarios, sys.argv[1])
    finally:
        sesion.close()


if __name__ == "__main__":
    main()
