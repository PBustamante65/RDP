# Variables y condiciones

# Variables

# Pueden ser de distintos tipos: int, float, str, bool
# int: numeros enteros
# float: numeros decimales
# str: cadenas de caracteres
# bool: booleanos (True/False)

x = 5
y = 5

# Condiciones en python

# == : igual a
# != : diferente a
# > : mayor que
# < : menor que
# =>: mayor o igual que
# <=: menor o igual que
# !=: diferente que

# Condicion if
# If responde a una condicion es decir, si una condicion se cumple, se ejecuta el codigo

if x >= y:  # Si x es mayor o igual a y
    print("Hola")  # imprimir "Hola"
else:  # De lo contrario
    print("Adios")  # Imprimir "Adios"


# Problema: Dada una calificacion, indicar si se aprueba o reprueba y convertirlo al sistema de calificaciones de letras
# para pedir un dato se usa la funcion input, esta se pueden usar con int, float, str, bool
calificacion = int(input("Ingrese una calificacion del 0-100:"))


# Para convertir la calificacion a letras se hacen las preguntas de las siguientes condiciones

if calificacion >= 90:  # Si la variable calificacion es mayor o igual a 90
    letra = "A"  # Crear la variable letra con el valor A
# elif es un else if, si no se cumple la primera condicion se evalua la siguiente
elif calificacion >= 80:  # Si la calificacion es mayor o igual a 80
    letra = "B"  # Crear la variable letra con el valor B
elif calificacion >= 70:
    letra = "C"
elif calificacion >= 60:
    letra = "D"
# else se ejecuta si no se cumple ninguna de las condiciones anteriores, denota el final de las condiciones
else:  # Si no se cumple ninguna de las condiciones anteriores
    letra = "F"  # Crear la variable letra con el valor F


# Para saber si la calificacion es aprobada o reprobada se hace la siguiente condicion
if calificacion >= 60:  # Si la calificacion es mayor o igual a 60
    resultado = "Aprobado"  # Crear la variable resultado con el valor Aprobado
else:
    resultado = "Reprobado"

print(letra)  # Al final de todo se imprime la variable letra

##Print
# Print es un comando que nos permite imprimir en la consola
# Print tiene varios argumentos, se puede imprimir una variable, una cadena de caracteres, un numero, etc
# Esto se puede hacer con una coma o con un +
# Para generar una nueva linea se puede usar \n, esta puede ir al inicio o al final
print("La calificacion en sistema americano es: ", letra)
print("\n La calificacion en sistema americano es: " + letra)


# Print tambien puede usar el operador f, este se usa con llaves, estas se llaman f-string, util para formatear cadenas de caracteres
print(f"Tu calificacion de {calificacion} es : {letra}" + ", " + resultado)
