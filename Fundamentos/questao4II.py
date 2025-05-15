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

# Algoritmo de Cauchy com saída detalhada de TODAS as iterações
def cauchy_method(x0, tol=1e-6, max_iter=15, modo='golden'):
    x = x0.copy()
    print(f"\nIniciando método de Cauchy ({'seção áurea' if modo == 'golden' else 'passo fixo = 1'})")
    print("Iter |        x[0]        |        x[1]        |      f(x)      |    grad[0]    |    grad[1]    |    d[0]    |    d[1]    |   alpha   |   norma   ")
    print("-"*110)
    for i in range(max_iter):
        grad = grad_f(x)
        d = -grad  # Direção de descida
        if modo == 'golden':
            f_alpha = lambda alpha: f(x + alpha * d)
            alpha = golden_section_search(f_alpha, 0, 2, tol=1e-4)
        elif modo == 'fixed':
            alpha = 1.0
        
        x_new = x + alpha * d
        norma = np.linalg.norm(x_new - x)
        
        # Print todos os dados em uma linha só
        print(f"{i+1:4d} | {x_new[0]:16.8f} | {x_new[1]:16.8f} | {f(x_new):12.6f} | {grad[0]:12.6f} | {grad[1]:12.6f} | {d[0]:9.6f} | {d[1]:9.6f} | {alpha:9.6f} | {norma:9.6f}")
        
        if norma < tol:
            print("\n>>> Convergência atingida!")
            return x_new, i + 1
        x = x_new
    
    print("\n>>> Limite de iterações atingido (diverge).")
    return x, max_iter
# Execução do código
x0 = np.array([1.0, 0.0])

# Com seção áurea
print("\n" + "="*50)
x_opt1, it1 = cauchy_method(x0, modo='golden')
print(f"\nSolução ótima (seção áurea): {x_opt1}, em {it1} iterações")

# Com passo fixo (para comparação)
print("\n" + "="*50)
x_opt2, it2 = cauchy_method(x0, modo='fixed')
print("Por conta de que o passo fixo não convergiu, não há solução ótima.")