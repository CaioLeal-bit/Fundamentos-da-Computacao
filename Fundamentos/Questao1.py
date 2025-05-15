def lagrange_interpolation(x, x_vals, y_vals):
    n = len(x_vals)
    resultado = 0.0
    for i in range(n):
        termo = y_vals[i]
        for j in range(n):
            if i != j:
                termo *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
        resultado += termo
    return resultado

# Valores fornecidos
x_vals = [1.2, 1.5, 2.0, 1.3]
y_vals = [1.928, 3.875, 9.0, 2.497]

x_interp = 1.8

resultado = lagrange_interpolation(x_interp, x_vals, y_vals)

print(f"L3({x_interp:.1f}) = {resultado:.6f}")