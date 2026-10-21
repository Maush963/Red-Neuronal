"""Red neuronal 2-4-8-1 implementada desde cero y exportable a Excel."""

import csv
import math
import random


ARQUITECTURA = [2, 4, 8, 1]
EPOCAS = 5000
TASA_APRENDIZAJE = 0.08
SEMILLA = 20


def sigmoid(valor):
    if valor < -700:
        return 0.0
    if valor > 700:
        return 1.0
    return 1.0 / (1.0 + math.exp(-valor))


def derivada_sigmoid(salida):
    return salida * (1.0 - salida)


def relu(valor):
    return valor if valor > 0.0 else 0.0


def derivada_relu(valor):
    return 1.0 if valor > 0.0 else 0.0


def crear_datos():
    """Crea 20 puntos para clasificar el XOR."""
    datos = []
    valores = (-1.0, 1.0)
    for x1 in valores:
        for x2 in valores:
            etiqueta = 1.0 if x1 * x2 < 0 else 0.0
            for _ in range(5):
                datos.append(([x1, x2], [etiqueta]))
    return datos


def crear_matriz(filas, columnas, generador):
    matriz = []
    for _ in range(filas):
        fila = []
        for _ in range(columnas):
            fila.append(generador())
        matriz.append(fila)
    return matriz


def crear_red(arquitectura, generador):
    pesos = []
    sesgos = []
    for capa in range(len(arquitectura) - 1):
        anteriores = arquitectura[capa]
        actuales = arquitectura[capa + 1]
        escala = math.sqrt(2.0 / anteriores)
        pesos.append(
            crear_matriz(
                anteriores,
                actuales,
                lambda: generador.uniform(-escala, escala),
            )
        )
        sesgos.append([0.0 for _ in range(actuales)])
    return pesos, sesgos


def avanzar(entrada, pesos, sesgos):
    activaciones = [entrada[:]]
    sumas = []
    actual = entrada[:]

    for numero_capa in range(len(pesos)):
        siguiente_suma = []
        siguiente = []
        for neurona in range(len(pesos[numero_capa][0])):
            suma = sesgos[numero_capa][neurona]
            for anterior in range(len(actual)):
                suma += actual[anterior] * pesos[numero_capa][anterior][neurona]
            siguiente_suma.append(suma)
            if numero_capa == len(pesos) - 1:
                siguiente.append(sigmoid(suma))
            else:
                siguiente.append(relu(suma))
        sumas.append(siguiente_suma)
        activaciones.append(siguiente)
        actual = siguiente

    return activaciones, sumas


def entrenar(datos, arquitectura, epocas, tasa_aprendizaje, semilla):
    generador = random.Random(semilla)
    pesos, sesgos = crear_red(arquitectura, generador)
    pesos_iniciales = copiar_red(pesos, sesgos)

    for epoca in range(epocas):
        error_total = 0.0
        orden = list(range(len(datos)))
        generador.shuffle(orden)

        for indice in orden:
            entrada, esperado = datos[indice]
            activaciones, sumas = avanzar(entrada, pesos, sesgos)
            prediccion = activaciones[-1][0]
            error_total += (esperado[0] - prediccion) ** 2

            deltas = [
                [0.0 for _ in range(len(matriz[0]))] for matriz in pesos
            ]
            ultima = len(pesos) - 1
            deltas[ultima][0] = (
                (prediccion - esperado[0]) * derivada_sigmoid(prediccion)
            )

            for capa in range(ultima - 1, -1, -1):
                for neurona in range(len(pesos[capa][0])):
                    influencia = 0.0
                    for siguiente in range(len(pesos[capa + 1][0])):
                        influencia += (
                            pesos[capa + 1][neurona][siguiente]
                            * deltas[capa + 1][siguiente]
                        )
                    deltas[capa][neurona] = (
                        influencia * derivada_relu(sumas[capa][neurona])
                    )

            for capa in range(len(pesos)):
                for anterior in range(len(pesos[capa])):
                    for neurona in range(len(pesos[capa][anterior])):
                        pesos[capa][anterior][neurona] -= (
                            tasa_aprendizaje
                            * activaciones[capa][anterior]
                            * deltas[capa][neurona]
                        )
                for neurona in range(len(sesgos[capa])):
                    sesgos[capa][neurona] -= (
                        tasa_aprendizaje * deltas[capa][neurona]
                    )

    return pesos_iniciales, (pesos, sesgos)


def copiar_red(pesos, sesgos):
    pesos_copia = []
    for matriz in pesos:
        pesos_copia.append([fila[:] for fila in matriz])
    return pesos_copia, [capa[:] for capa in sesgos]


def exportar_pesos(ruta, pesos, sesgos, nombre_estado):
    with open(ruta, "a", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        for capa, matriz in enumerate(pesos):
            for anterior in range(len(matriz)):
                for neurona in range(len(matriz[anterior])):
                    escritor.writerow(
                        [
                            nombre_estado,
                            capa + 1,
                            "peso",
                            anterior + 1,
                            neurona + 1,
                            format(matriz[anterior][neurona], ".10f"),
                        ]
                    )
        for capa, sesgo in enumerate(sesgos):
            for neurona, valor in enumerate(sesgo):
                escritor.writerow(
                    [
                        nombre_estado,
                        capa + 1,
                        "bias",
                        "-",
                        neurona + 1,
                        format(valor, ".10f"),
                    ]
                )

def main():
    datos = crear_datos()
    iniciales, finales = entrenar(
        datos, ARQUITECTURA, EPOCAS, TASA_APRENDIZAJE, SEMILLA
    )
    pesos_iniciales, sesgos_iniciales = iniciales
    pesos_finales, sesgos_finales = finales

    with open("pesos_generados.csv", "w", newline="", encoding="utf-8") as archivo:
        csv.writer(archivo).writerow(
            ["estado", "capa", "tipo", "neurona_anterior", "neurona_actual", "valor"]
        )
    exportar_pesos("pesos_generados.csv", pesos_iniciales, sesgos_iniciales, "epoch_0")
    exportar_pesos(
        "pesos_generados.csv", pesos_finales, sesgos_finales, "epoch_final"
    )

    print("Red neuronal entrenada: " + " -> ".join(str(n) for n in ARQUITECTURA))
    print("Muestras: " + str(len(datos)))
    print("Archivo creado: pesos_generados.csv")


if __name__ == "__main__":
    main()
