import math

def f(x):
    return x ** x

x = 0.0001
h_initial = 0.00005  # Escolhido para garantir x - h_k > 0
tolerance = 1e-6
max_iterations = 20

D_prev_prev = None
D_prev = None
D_current = None
k = 0

while k < max_iterations:
    h_k = h_initial * (10 ** (-k))
    if (x - h_k) <= 0:
        break
    try:
        f_plus = f(x + h_k)
        f_minus = f(x - h_k)
    except:
        break
    D_current = (f_plus - f_minus) / (2 * h_k)
    
    if D_prev is None:
        D_prev = D_current
        k += 1
        continue
    elif D_prev_prev is None:
        D_prev_prev = D_prev
        D_prev = D_current
        k += 1
        continue
    
    diff_current = abs(D_current - D_prev)
    diff_prev = abs(D_prev - D_prev_prev)
    
    if diff_current >= diff_prev:
        D_current = D_prev
        break
    if diff_prev < tolerance:
        break
    
    D_prev_prev = D_prev
    D_prev = D_current
    k += 1

print(f"Aproximação da derivada: {D_current}")
print(f"Número de iterações: {k}")