import numpy as np

def apGrad(f, x):
    """
    Aproximación por diferencias finitas del gradiente de f
    Expected time: O(n)
    Reason: Se evalúa f 2 veces por cada dimensión.
    """
    n = len(x)
    grad = np.zeros(n)
    mach_eps = np.finfo(float).eps

    for i in range(n):

        h_i = (mach_eps ** (1/3)) * (np.abs(x[i]) + 1)
        
        x_plus = x.copy()
        x_plus[i] = x_plus[i] + h_i
        

        x_minus = x.copy()
        x_minus[i] = x_minus[i] - h_i
        

        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h_i)

        
    return grad


def apHess(f, x):
    """Finite difference approximation of the hessian of f
    Expected time: O(n^2)
    Reason: Hay dos ciclos anidados para recorrer la matriz nxn.
    """

    n = len(x)
    hess = np.zeros((n, n))
    
    mach_eps = np.finfo(float).eps

    h = (mach_eps ** 0.25) * (np.abs(x) + 1)
    
    for i in range(n):
        for j in range(n):
            
            # Término: f(x + hi + hj)
            x_pp = x.copy()
            x_pp[i] += h[i]
            x_pp[j] += h[j]
            
            # Término: f(x - hi + hj)
            x_mp = x.copy()
            x_mp[i] -= h[i]
            x_mp[j] += h[j]
            
            # Término: f(x + hi - hj)
            x_pm = x.copy()
            x_pm[i] += h[i]
            x_pm[j] -= h[j]
            
            # Término: f(x - hi - hj)
            x_mm = x.copy()
            x_mm[i] -= h[i]
            x_mm[j] -= h[j]
            
            # Fórmula de diferencias finitas mixtas

            term1 = f(x_pp) - f(x_mp)
            term2 = f(x_pm) - f(x_mm)
            
            hess[i, j] = (term1 - term2) / (4 * h[i] * h[j])
            
    return hess

