# Definición de la función
equation = lambda x: -1*(-(x[0] - 6)**2 - (x[1] - 5)**2 + 82.81)

# Solución a evaluar
solution = [14.095, 0.84296]

# Evaluar la función
result = equation(solution)

# Imprimir el resultado
print("El resultado de la evaluación es:", result)
