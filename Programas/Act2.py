# Listas y diccionarios

##Listas
# Las listas son una coleccion de elementos, pueden ser de distintos tipos de datos
# Las listas se pueden crear con corchetes, tambien se puede crear con la funcion list

lista = [1, 2, 3, 4, 5]
print(f"Ejemplo de una lista: {lista}")

# Una lista puede tener distintos tipos de datos
lista = [1, "hola", 3.14, True]
print(f"\n Una lista con distintos tipos de datos: {lista}")

# Para agregar datos a la lista se usa el metodo append
# Este metodo agrega el dato al final de la lista
lista.append(6)
print("\n El metodo append, agrega un dato a la lista al final:")
print(f"lista.append(6): {lista}\n")


# Para imprimir un elemento en especifico se usa el indice
# Python usa el indice 0 como el primer elemento
print("Para imprimir un elemento en especifico se usa el indice:")
print(f"lista[0]: {lista[0]}\n")
print(f"print(lista[0], lista[1]): {lista[0], lista[1]}\n")

# Esto se puede simplificar con 'slices', los cuales son un rango de elementos
print("Para imprimir un rango de elementos se usa 'slices':")
print(f"print(lista[0:3]): {lista[0:3]}\n")

# Para insertar un dato en una posicion especifica se usa el metodo insert
# Esto agrega el dato en la posicion indicada
lista.insert(2, "insertar")
print("Para insertar un dato en una posicion especifica se usa el metodo insert:")
print(f"lista.insert(2, 'insertar'): {lista}\n")

# El metod pop elimina el ultimo elemento de la lista
# Se puede usar con un indice especifico
lista.pop()
print("El metodo pop elimina el ultimo elemento de la lista:")
print(f"lista.pop(): {lista}\n")

lista.pop(2)
print("El metodo pop tambien elimina el elemento en la posicion indicada:")
print(f"lista.pop(2): {lista}\n")


# Para eliminar un elemento en especifico se usa el metodo remove
lista.remove("hola")
print("Para eliminar un elemento en especifico se usa el metodo remove:")
print(f"lista.remove('hola'): {lista}\n")


# Operaciones tipicas con listas
# Los elementos de una lista se pueden sumar, restar, multiplicar, etc

lista = [1, 2, 3, 4, 5]

# Se puede imprimir la longitud de la lista con el metodo len
print(f"Lista: {lista}")
print("La longitud de la lista es:")
print(f"len(lista): {len(lista)}\n")

# Si todos los elementos de la lista son del mismo tipo se puede sumar
print("Suma de todos los elementos de la lista:")
print(f"sum(lista): {sum(lista)}\n")

# Estos tambien se pueden ordenar con la funcion sort
# Se puede usar el parametro reverse para ordenar de manera inversa

# Orden inverso
lista.sort(reverse=True)

print("Ordenar de manera Descendente:")
print(f"lista.sort(reverse=False): {lista}\n")


# Todos los valores de una lista pueden ser iterados, utilizando el ciclo for


lista.sort(reverse=False)

# Ciclo for
# El ciclo for es una estructura de control que se usa para iterar sobre una secuencia de elementos
# En este caso la secuencia es una lista

print("Iterar sobre una lista con el ciclo for:")
print("for i in lista:")
print("    print(i)")

for i in lista:  # Para cada i en la lista
    print(i)  # Imprimir i
