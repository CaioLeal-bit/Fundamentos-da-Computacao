import numpy as np
import matplotlib.pyplot as plt

# Dados
x = np.array([-1, 0, 1, 2, 3, 4, 5, 6])
y = np.array([10, 9, 7, 5, 4, 3, 0, -1])
n = len(x)
e = np.ones(n)

# Passos do Algoritmo 5.1
xmean = (x @ e) / n
ymean = (y @ e) / n
sumx2 = ((x - xmean) @ (x - xmean))
sumxy = ((x - xmean) @ (y - ymean))
a = sumxy / sumx2
b = ymean - a * xmean

# Impressão dos coeficientes
print(f"Coeficientes da reta: a = {a:.4f}, b = {b:.4f}")
print(f"Equação da reta: y = {a:.2f}x + {b:.2f}")

# Definindo intervalo muito maior para a reta
x_reta = np.linspace(min(x) - 10, max(x) + 10, 1000)  # 10 unidades a mais pra cada lado
y_reta = a * x_reta + b

# Gráfico
plt.figure(figsize=(10, 6))  # Deixa o gráfico maior
plt.scatter(x, y, color='blue', label='Pontos dados')
plt.plot(x_reta, y_reta, color='red', label='Reta dos mínimos quadrados')

# Ajustando limites dos eixos manualmente
plt.xlim(min(x) - 5, max(x) + 5)  # 5 unidades além dos dados
plt.ylim(min(y) - 5, max(y) + 5)  # 5 unidades além dos dados

plt.xlabel('x')
plt.ylabel('y')
plt.title('Ajuste por mínimos quadrados - reta estendida')
plt.legend()
plt.grid(True)
plt.show()