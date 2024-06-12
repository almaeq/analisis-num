from aprox_discreta import *
def main():
    x = list(map(float, input("Ingrese los valores de x separados por comas: ").split(',')))
    y = list(map(float, input("Ingrese los valores de y separados por comas: ").split(',')))
    x = np.array(x)
    y = np.array(y)
    
    model = input("Ingrese el tipo de modelo ('polinomica', 'exponencial', 'fraccion'): ").strip().lower()
    
    if model == 'polinomica':
        bases = input("Ingrese las bases polinómicas separadas por comas (ej. 2*x**2 - 1, x, 1): ").split(',')
        coeffs, y_fit = fit_and_plot(x, y, model=model, bases=bases)
        print(f'Coeficientes del ajuste polinómico con bases personalizadas:')
        print(coeffs)
        # Crear una cadena que represente la expresión polinómica ajustada
        polynomial_expr = ' + '.join([f'{coeff:.5f}*{base.strip()}' for coeff, base in zip(coeffs, bases)])
        print(f'Expresión polinómica ajustada: y = {polynomial_expr}')
        # Reemplazar en la función polinómica
        def polynomial_function(x, coeffs=coeffs, bases=bases):
            return eval(polynomial_expr)
        
    elif model == 'exponencial':
        exp_form = input("Ingrese la forma exponencial ('a*e^(b/x)', 'a*e^(a*x - b*x**2)', o 'a*e^(b*x)'): ").strip().lower()
        coeffs, y_fit = fit_and_plot(x, y, model=model, exp_form=exp_form)
        print('Coeficientes del ajuste exponencial:')
        print(coeffs)
        if exp_form == 'a*e^(b/x)':
            print("Función ajustada:")
            print(f"y = {coeffs[0]:.5f} * e^({coeffs[1]:.5f} / x)")
        elif exp_form == 'a*e^(a*x - b*x**2)':
            print("Función ajustada:")
            print(f"y = {np.exp(coeffs[2]):.5f} * e^({coeffs[0]:.5f} * x - {coeffs[1]:.5f} * x^2)")
        elif exp_form == 'a*e^(b*x)':
            print("Función ajustada:")
            print(f"y = {coeffs[0]:.5f} * e^({coeffs[1]:.5f} * x)")

    elif model == 'fraccion':
        custom_form = input("Ingrese la forma personalizada ('a*(x/(b+x))'): ").strip().lower()
        coeffs, y_fit = fit_and_plot(x, y, model=model, custom_form=custom_form)
        print('Coeficientes del ajuste personalizado:')
        print(coeffs)
        print("Función ajustada:")  
        print(f"y = {coeffs[0]:.5f} * (x / ({coeffs[1]:.5f} + x))")

    else:
        print("Modelo no válido. Por favor ingrese 'polinomica', 'exponencial' o 'fracción'.")

if __name__ == "__main__":
    main()