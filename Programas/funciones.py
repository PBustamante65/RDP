# Funciones

# Las funciones son un conjunto de instrucciones que se pueden reutilizar
# Las funciones pueden recibir parametros y devolver valores
# Son la base de la programacion orientada a objetos


def saludar(nombre):  # Definir la funcion saludar, este recibe un parametro nombre
    print(f"Hola, {nombre}\n")  # La funcion imprime un saludo con el nombre recibido


print("Ejemplo de una funcion:")
saludar("Patricio")  # Llamar a la funcion saludar con el argumento "Patricio"

# Las funciones pueden reutilizarse con distintos parametros
nombre1 = "Juan"
nombre2 = "Pedro"

print("Ejemplo de una funcion con distintos parametros:")
saludar(nombre1)  # Llamar a la funcion saludar con el argumento nombre1
saludar(nombre2)  # Llamar a la funcion saludar con el argumento nombre2


# Las funciones pueden ser utilizadas para hacer operaciones matematicas


def syr(a, b):
    suma = a + b
    resta = a - b
    return (
        suma,
        resta,
    )  # Una funcion puede devolver mas de un valor, el posicionamiento de los valores es importante, ya que se devuelve en orden de aparicion


suma, resta = syr(2, 3)
print(f"Suma: {suma}, Resta: {resta}\n")


# Diccionarios
# Los diccionarios son una coleccion de elementos, pueden ser de distintos tipos de datos
# Los diccionarios son como una lista, pero con claves y valores, es decir, cada elemento de un diccionario, contiene subinformacion, es decir, un valor asociado a una clave


# El diccionario estudiantes tiene 3 estudiantes, cada estudiante tiene 3 materias diferentes, cada materia tiene una calificacion
# De este modo, el diccionario estudiantes es una coleccion de estudiantes, cada estudiante tiene sus materias y calificaciones
estudiantes = {
    "Alice": {"Matematicas": 85, "Ciencia": 82, "Ingles": 78},
    "Bob": {"Matematicas": 90, "Ciencia": 85, "Ingles": 88},
    "Carlos": {"Matematicas": 78, "Ciencia": 83, "Ingles": 91},
}

##### Crear un metodo de cada estudiante y el promedio general de todos los estudiantes


def promediosgeneral(estudiantes):  # Definir la funcion promediosgeneral
    promedios = {}  # Se crea un diccionario para guardar los promedios
    total_suma = (
        0  # Se crea una variable para guardar la suma de todas las calificaciones
    )
    total_count = 0  # Se crea una variable para guardar la cantidad de calificaciones
    for (
        estudiante,
        calificacion,
    ) in estudiantes.items():  # Se itera sobre cada estudiante y sus calificaciones
        suma = sum(
            calificacion.values()
        )  # Se suma las calificaciones individuales de cada estudiante
        count = len(calificacion)  # Se cuenta la cantidad de calificaciones
        promedio = suma / count  # Se calcula el promedio
        promedios[estudiante] = promedio  # Se guarda el promedio en el diccionario
        total_suma += suma  # Se suma la suma de todas las calificaciones
        total_count += count  # Se suma la cantidad de calificaciones
    promedio_general = total_suma / total_count  # Se calcula el promedio general

    return (
        promedios,
        promedio_general,
    )  # Se devuelve el diccionario de promedios y el promedio general


promedio, promedio_general = promediosgeneral(estudiantes)
print(promedio)
print(promedio_general)
