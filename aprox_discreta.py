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

# Solicitar los datos al usuario
x, y = solicitar_datos()

# Solicitar las funciones base del subespacio al usuario
bases = solicitar_bases()

# Evaluar las funciones base en los puntos x
A = evaluar_bases(bases, x)

# Resolver el sistema de ecuaciones normales A.T * A * p = A.T * y
# para los parámetros p
p = np.linalg.lstsq(A, y, rcond=None)[0]

# Imprimir los coeficientes uno debajo del otro
print("Coeficientes:")
for i, coef in enumerate(p):
    print(f"Coeficiente {i + 1}: {coef}")

# Generar valores predichos usando los parámetros ajustados
y_pred = A @ p

# Visualización de los datos y la función ajustada
plt.scatter(x, y, color='red', label='Datos')
plt.plot(np.sort(x), y_pred[np.argsort(x)], label='Ajuste en el subespacio dado')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
#plt.savefig('grafico.png')
