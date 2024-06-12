from aprox_discreta import *
from colorama import Fore, Style, init

init(autoreset=True)

def main():
    x = list(map(float, input("Ingrese los valores de x separados por comas: ").split(',')))
    y = list(map(float, input("Ingrese los valores de y separados por comas: ").split(',')))
    x = np.array(x)
    y = np.array(y)
    
    model = input("Ingrese el tipo de modelo ('polinomica', 'exponencial', 'fraccion'): ").strip().lower()
    
    if model == 'polinomica':
        bases = input("Ingrese las bases polinómicas separadas por comas (ej. 2*x**2 - 1, x, 1): ").split(',')
        coeffs, y_fit = fit_and_plot(x, y, model=model, bases=bases)
        print(Fore.MAGENTA + f'Coeficientes del ajuste polinómico con bases personalizadas:')
        print(Fore.CYAN + str(coeffs))
        polynomial_expr = ' + '.join([f'{coeff:.5f}*{base.strip()}' for coeff, base in zip(coeffs, bases)])
        print(Fore.MAGENTA + f'Expresión polinómica ajustada: y = {polynomial_expr}')
        
    elif model == 'exponencial':
        exp_form = input("Ingrese la forma exponencial ('a*e^(b/x)', 'a*e^(a*x - b*x**2)', o 'a*e^(b*x)'): ").strip().lower()
        coeffs, y_fit = fit_and_plot(x, y, model=model, exp_form=exp_form)
        print(Fore.MAGENTA + 'Coeficientes del ajuste exponencial:')
        print(Fore.CYAN + str(coeffs))
        if exp_form == 'a*e^(b/x)':
            print(Fore.RED + "Función ajustada:")
            print(Fore.YELLOW + f"y = {coeffs[0]:.5f} * e^({coeffs[1]:.5f} / x)")
        elif exp_form == 'a*e^(a*x - b*x**2)':
            print(Fore.RED + "Función ajustada:")
            print(Fore.YELLOW + f"y = {np.exp(coeffs[2]):.5f} * e^({coeffs[0]:.5f} * x - {coeffs[1]:.5f} * x^2)")
        elif exp_form == 'a*e^(b*x)':
            print(Fore.RED + "Función ajustada:")
            print(Fore.YELLOW + f"y = {coeffs[0]:.5f} * e^({coeffs[1]:.5f} * x)")

    elif model == 'fraccion':
        custom_form = input("Ingrese la forma personalizada ('a*(x/(b+x))'): ").strip().lower()
        coeffs, y_fit = fit_and_plot(x, y, model=model, custom_form=custom_form)
        print(Fore.MAGENTA + 'Coeficientes del ajuste personalizado:')
        print(Fore.CYAN + str(coeffs))
        print(Fore.RED + "Función ajustada:")  
        print(Fore.YELLOW + f"y = {coeffs[0]:.5f} * (x / ({coeffs[1]:.5f} + x))")

    else:
        print(Fore.RED + "Modelo no válido. Por favor ingrese 'polinomica', 'exponencial' o 'fracción'.")

if __name__ == "__main__":
    main()