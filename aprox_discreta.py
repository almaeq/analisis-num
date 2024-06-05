import numpy as np
import matplotlib.pyplot as plt

# Función para solicitar los datos al usuario
def solicitar_datos():
    x = input("Introduce los valores de x separados por comas: ")
    y = input("Introduce los valores de y separados por comas: ")

    # Convertir las cadenas de entrada en listas de números
    x = list(map(float, x.split(',')))
    y = list(map(float, y.split(',')))

    return np.array(x), np.array(y)

# Función para solicitar las funciones base al usuario
def solicitar_bases():
    base_str = input("Introduce las funciones base del subespacio (por ejemplo, '1,x,2*x**2-1'): ")
    base_funcs = base_str.split(',')
    return base_funcs

# Función para evaluar las funciones base
def evaluar_bases(bases, x):
    A = np.zeros((len(x), len(bases)))
    for i, base in enumerate(bases):
        base_func = lambda x: eval(base, {'x': x, 'np': np})
        A[:, i] = base_func(x)
    return A

# Función para construir la función ajustada con los coeficientes
def construir_funcion(base_funcs, coeficientes):
    funcion = ""
    for coef, base in zip(coeficientes, base_funcs):
        funcion += f"{coef}*({base}) + "
    return funcion.rstrip(" + ")

# Solicitar los datos al usuario
x, y = solicitar_datos()

# Verificar que las longitudes de x e y sean iguales
if len(x) != len(y):
    raise ValueError("Los vectores x e y deben tener la misma longitud.")

# Solicitar las funciones base del subespacio al usuario
bases = solicitar_bases()

# Evaluar las funciones base en los puntos x
A = evaluar_bases(bases, x)

# Verificar las dimensiones de A y y
if A.shape[0] != y.shape[0]:
    raise ValueError("El número de filas en la matriz A debe ser igual a la longitud del vector y.")

# Resolver el sistema de ecuaciones normales A.T * A * p = A.T * y para los parámetros p
p = np.linalg.lstsq(A, y, rcond=None)[0]

# Imprimir los coeficientes uno debajo del otro
print("Coeficientes:")
for i, coef in enumerate(p):
    print(f"Coeficiente {i + 1}: {coef}")

# Construir la función ajustada
funcion_ajustada = construir_funcion(bases, p)
print(f"Función ajustada: f(x) = {funcion_ajustada}")

# Generar valores predichos usando los parámetros ajustados
y_pred = A @ p

# Visualización de los datos y la función ajustada
plt.scatter(x, y, color='red', label='Datos')
plt.plot(np.sort(x), y_pred[np.argsort(x)], label='Ajuste en el subespacio dado')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
#plt.savefig('grafico.png')