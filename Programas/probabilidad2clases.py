# Clasificacion probabilistica de 2 clases

# Se importan las librerias de numpy para calculos numericos y matplotlib para graficar
import numpy as np
import matplotlib.pyplot as plt


# Clases
# Las clases son un conjunto de instrucciones que se pueden reutilizar en el codigo
# Son la base de la programacion orientada a objetos


class classifier:  # Se define la clase classifier

    def __init__(
        self, mu, sigma
    ):  # Generalmente las clases se inicializan con el metodo __init__, el cual recibe parametros para su uso global dentro de la clase
        self.mu = mu  # Se define la variable mu
        self.sigma = sigma  # Se define la variable sigma

    def normal(self, x):  # Se define el metodo normal el cual recibe el valor de x
        P = (-1 / 2) * ((x - self.mu) ** 2) / (self.sigma**2)
        a = (1 / (np.sqrt(2 * np.pi) * self.sigma)) * np.exp(P)
        return a  # El metodo calcula la distribucion normal para el valor de x y retorna el resultado

    def region(self, P, priori):  # Este metodo calcula la region de decision
        a = np.log(P)  # Se calcula el logaritmo de la distribucion
        disc = a + np.log(priori)  # Se calcula la region de decision
        return disc  # Retorna la region de decision

    def final(
        self, x_test, priori
    ):  # El metodo final se usa para calcular la region de decision de un punto de prueba utilizando una probabilidad a priori

        P = self.normal(
            x_test
        )  # Se calcula la distribucion normal para el punto de prueba
        disc = self.region(P, priori)  # Se calcula la region de decision
        return disc  # Retorna la region de decision


# Clasificacion de 2 clases

# Creamos 1000 puntos, que van desde -5 a 5
x = np.linspace(-5, 5, 1000)

mu1 = float(
    input("Ingresa mu1: ")
)  # Pedimos al usuario que ingrese los valores de mu y sigma para las dos clases
sigma1 = float(input("Ingresa sigma1: "))
mu2 = float(input("Ingresa mu2: "))
sigma2 = float(input("Ingresa sigma2: "))


# Con los valores de mu y sigma, calculamos la distribucion normal para cada clase

classificacion1 = classifier(
    mu1, sigma1
)  # Para cada clase, creamos un objeto de la clase classifier
classificacion2 = classifier(mu2, sigma2)

# Classificacion1 y 2 ahora contienen los metodos y atributos de la clase classifier
# Podemos utilizar los metodos y atributos de la clase para calcular la distribucion normal y la region de decision

# Para calcular la probabilidad de los valores de x, usamos la distribucion normal, normal(x)
P1 = classificacion1.normal(x)
P2 = classificacion2.normal(x)

# Podemos graficar las distribuciones normales para cada clase usando matplotlib (plt)

plt.figure(1, figsize=(10, 5))  # Creamos la figura 1, de un tamaño de 10x5
plt.title("Distribucion normal para cada clase")  # Incluimos un titulo a la figura
plt.plot(
    x, P1, label="Clase 1"
)  # Graficamos la distribucion normal para la clase 1, en el eje x se grafican los valores de x, en el eje y se grafican las probabilidades
plt.plot(
    x, P2, label="Clase 2"
)  # Agregamos el parametro label para identificar la distribucion normal de cada clase
plt.xlabel("x")  # El eje x le damos el nombre x
plt.ylabel("P(x)")  # El eje y le damos el nombre P(x)
plt.legend()  # Usamos el metodo legend para agregar una leyenda a la figura
plt.show()  # Mostramos la figura


# Para clasificar un punto de prueba, pedimos al usuario que ingrese el valor de x
# Es tambien necesario ingresar la probabilidad a priori de cada clase

x_test = float(input("Ingrese x: "))
priori1 = 0.5
priori2 = 0.7

# Calculamos la region de decision para el punto de prueba, utilizando el metodo 'final'
disc1 = classificacion1.final(x_test, priori1)
disc2 = classificacion2.final(x_test, priori2)

# Graficamos la region de decision para cada clase y la posicion del punto de prueba
plt.figure(2, figsize=(10, 5))
plt.title("Distribucion normal")
plt.plot(x, P1)
plt.plot(x, P2)
plt.plot([x_test, x_test], [0, 1], "r--")
plt.xlabel("X")
plt.ylabel("P(x)")
plt.show()

# Para finalizar, mostramos la region de decision para cada clase
print(f"Region de decision para la clase 1: {disc1}")
print(f"Region de decision para la clase 2: {disc2}")

# La region de decision es un numero que indica la probabilidad de que el punto de prueba pertenezca a la clase 1 o la clase 2
# Si una region de decision es mayor que otra, entonces la probabilidad de que el punto de prueba pertenezca a esa clase es mayor

if disc1 > disc2:
    print(f"{x_test} esta en la clase 1")
else:
    print(f"{x_test} esta en la clase 2")
