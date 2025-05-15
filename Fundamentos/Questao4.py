import numpy as np

# Função objetivo
def f(x):
    return 0.5 * (x[0] - 3)**2 + (x[1] - 1)**2

# Gradiente
def grad_f(x):
    return np.array([x[0] - 3, 2 * (x[1] - 1)])

# Busca por seção áurea
def golden_section_search(f_alpha, a, b, tol=1e-4):
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(c - d) > tol:
        if f_alpha(c) < f_alpha(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (b + a) / 2

# Algoritmo de Cauchy com saída limpa
def cauchy_method(x0, tol=1e-6, max_iter=16, modo='golden'):
    x = x0.copy()
    print(f"\nIniciando método de Cauchy ({'seção áurea' if modo == 'golden' else 'passo fixo = 1'})")
    for i in range(max_iter):
        grad = grad_f(x)
        d = -grad
        f_alpha = lambda alpha: f(x + alpha * d)
        alpha = golden_section_search(f_alpha, 0, 2, tol=1e-4)
        x_new = x + alpha * d
       # if i == 0 or i == max_iter - 1 or np.linalg.norm(x_new - x) < tol:
        print(f"Iteração {i+1}: x = {x_new}, f(x) = {f(x_new):.6f}, passo = {alpha:.6f}")
        if np.linalg.norm(x_new - x) < tol:
            print(">>> Convergência atingida.")
            return x_new, i + 1
        x = x_new
    print(">>> Limite de iterações atingido (ela diverge).")
    return x, max_iter

# Teste para os dois métodos
x0 = np.array([1.0, 0.0])

# Com seção áurea
x_opt1, it1 = cauchy_method(x0, modo='golden')
print(f"Solução ótima aproximada (seção áurea): {x_opt1}, em {it1} iterações")

# Com passo fixo
x_opt2, it2 = cauchy_method(x0, modo='fixed')
print(f"Solução final (passo fixo = 1): {x_opt2}, após {it2} iterações")
