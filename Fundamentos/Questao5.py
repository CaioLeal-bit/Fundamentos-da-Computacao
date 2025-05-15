import math

# Função objetivo
def f(x1, x2):
    return 0.5 * (x1 - 3)**2 + (x2 - 1)**2

# Gradiente de f
def gradiente(x):
    return [x[0] - 3, 2*(x[1] - 1)]

# Hessiana de f
def hessiana(x):
    return [
        [1, 0],
        [0, 2]
    ]

# Norma de um vetor
def norma(vetor):
    return math.sqrt(sum([i**2 for i in vetor]))

# Produto escalar
def dot_product(v1, v2):
    return sum([v1[i] * v2[i] for i in range(len(v1))])

# Método da Seção Áurea para busca de passo
def secao_aurea(x, d, a=0, b=2, toler=1e-4):
    phi = (math.sqrt(5) - 1) / 2  # aproximadamente 0.618
    while True:
        t1 = b - phi * (b - a)
        t2 = a + phi * (b - a)
        
        f1 = f(x[0] + t1*d[0], x[1] + t1*d[1])
        f2 = f(x[0] + t2*d[0], x[1] + t2*d[1])

        if f1 < f2:
            b = t2
        else:
            a = t1

        if b - a <= 2 * toler:
            break

    return (a + b) / 2

# Método de Newton para otimização
def newton(X0, toler=1e-6, busca_linha=True):
    x = X0[:]
    k = 0

    while True:
        grad = gradiente(x)
        hess = hessiana(x)

        # Resolver sistema Hessiana * d = -grad (como é diagonal aqui, simples)
        d = [
            -grad[0]/hess[0][0],
            -grad[1]/hess[1][1]
        ]

        if norma(grad) <= toler:
            break

        if busca_linha:
            # Faz busca de passo ótima
            t = secao_aurea(x, d)
        else:
            # Método de Newton puro (t = 1)
            t = 1

        # Atualiza
        x = [x[i] + t*d[i] for i in range(len(x))]
        k += 1

    return x, k

def main():
    X0 = [1, 0]
    toler = 1e-6

    print("Newton com busca de linha (seção áurea):")
    ans, k = newton(X0, toler, busca_linha=True)
    print(f"Solucao aproximada: {ans}")
    print(f"Numero de iteracoes: {k}")

    print("\nNewton puro (passo = 1):")
    ans_puro, k_puro = newton(X0, toler, busca_linha=False)
    print(f"Solucao aproximada: {ans_puro}")
    print(f"Numero de iteracoes: {k_puro}")

if __name__ == "__main__":
    main()
