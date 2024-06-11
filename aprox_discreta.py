import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

# Función para realizar el ajuste lineal
def ajuste_lineal(x, y):
    bases = solicitar_bases()
    A = evaluar_bases(bases, x)
    if A.shape[0] != y.shape[0]:
        raise ValueError("El número de filas en la matriz A debe ser igual a la longitud del vector y.")
    p = np.linalg.lstsq(A, y, rcond=None)[0]
    return p, bases

# Función para realizar el ajuste exponencial simple
def ajuste_exponencial(x, y):
    if np.any(y <= 0):
        raise ValueError("Todos los valores de y deben ser positivos para aplicar el logaritmo natural.")
    Y = np.log(y)
    A = np.vstack([x, np.ones(len(x))]).T
    p = np.linalg.lstsq(A, Y, rcond=None)[0]
    b = p[0]
    A_log = p[1]
    a = np.exp(A_log)
    return a, b

# Función para realizar el ajuste exponencial más complejo
def ajuste_exponencial_complejo(x, y):
    # Definir la función objetivo
    def func(x, c, a, b):
        return c * np.exp(a * x - b * x**2)
    
    # Usar curve_fit para ajustar la función a los datos
    popt, pcov = curve_fit(func, x, y, p0=(1, 1, 1))
    
    return popt

# Función para realizar el ajuste de la forma a * (x / (b + x))
def ajuste_fraccion(x, y):
    def func(x, a, b):
        return a * (x / (b + x))
    
    popt, pcov = curve_fit(func, x, y, p0=(1, 1))
    
    return popt

# Solicitar los datos al usuario
x, y = solicitar_datos()

# Verificar que las longitudes de x e y sean iguales
if len(x) != len(y):
    raise ValueError("Los vectores x e y deben tener la misma longitud.")

# Solicitar el tipo de ajuste al usuario
tipo_ajuste = input("Introduce el tipo de ajuste ('lineal', 'exponencial', 'exponencial_complejo', 'fraccion'): ").strip().lower()

coeficientes = []
if tipo_ajuste == 'lineal':
    # Ajuste lineal
    p, bases = ajuste_lineal(x, y)
    funcion_ajustada = construir_funcion(bases, p)
    coeficientes = p
    A = evaluar_bases(bases, x)
    y_pred = A @ p

elif tipo_ajuste == 'exponencial':
    # Ajuste exponencial
    a, b = ajuste_exponencial(x, y)
    funcion_ajustada = f"{a} * e^({b} * x)"
    coeficientes = [a, b]
    y_pred = a * np.exp(b * x)

elif tipo_ajuste == 'exponencial_complejo':
    # Ajuste exponencial complejo
    c, a, b = ajuste_exponencial_complejo(x, y)
    funcion_ajustada = f"{c} * e^({a} * x - {b} * x**2)"
    coeficientes = [c, a, b]
    y_pred = c * np.exp(a * x - b * x**2)

elif tipo_ajuste == 'fraccion':
    # Ajuste de la forma a * (x / (b + x))
    a, b = ajuste_fraccion(x, y)
    funcion_ajustada = f"{a} * (x / ({b} + x))"
    coeficientes = [a, b]
    y_pred = a * (x / (b + x))

else:
    raise ValueError("Tipo de ajuste no reconocido. Por favor, introduce 'lineal', 'exponencial', 'exponencial_complejo' o 'fraccion'.")

# Imprimir los coeficientes
print("Coeficientes:")
for i, coef in enumerate(coeficientes):
    print(f"Coeficiente {i + 1}: {coef}")

# Imprimir la función ajustada
print(f"Función ajustada: f(x) = {funcion_ajustada}")

# Visualización de los datos y la función ajustada
plt.scatter(x, y, color='red', label='Datos')
plt.plot(np.sort(x), y_pred[np.argsort(x)], label=f'Ajuste {tipo_ajuste}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
#plt.savefig('grafico.png')