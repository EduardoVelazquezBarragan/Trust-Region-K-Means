import numpy as np
import scipy.linalg as LA
from Lib_derivadas import apGrad, apHess


def _pCauchy(g, delta,B):
    w = g/ LA.norm(g)

    alpha_u = 1/np.dot(w, B@w)
    if alpha_u >0 and abs(alpha_u)*LA.norm(g) <= delta:
        res= - alpha_u*g
    else:
        res = -delta*w

    return res

def _pDogLeg(g, delta,B):
    w = g/ LA.norm(g)

    alpha_u = 1/np.dot(w, B@w)
    assert alpha_u >0 , "B no es sim. pos. def."
    if alpha_u *LA.norm(g) >= delta:
        pDL = -delta*w

    else:
         pB = LA.solve(B, -g)
         if  LA.norm(pB) <= delta:
             pDL =pB
         else:
             u = -alpha_u*g
             v= pB - u
             a = np.dot(v,v)
             b= 2*np.dot(u,v)
             c= np.dot(u,u)-delta**2

             discriminante = max(0, b**2 - 4*a*c) #Checar dog leg
             alpha_s = (-b + np.sqrt(discriminante)) / (2*a)
             pDL = u + alpha_s * v  

    return pDL   

def _pDogLeg_Seguro(g, delta, B):
    """
    Método Dogleg adaptado para ser seguro con matrices Cuasi-Newton
    como SR1 que pueden perder la propiedad de ser definidas positivas.
    """
    norm_g = LA.norm(g)
    w = g / norm_g
    gBg = np.dot(g, B @ g)
    
    # 1. Calculamos el punto de Cauchy (pU)
    if gBg <= 0:
        # Si la curvatura es negativa, la parábola se abre hacia abajo.
        # Vamos directo a la frontera de la región de confianza.
        pU = -delta * w
    else:
        alpha_u = (norm_g**2) / gBg
        pU = -alpha_u * g

    # Si el punto de Cauchy ya está fuera o en el borde de la región, lo truncamos y terminamos
    if LA.norm(pU) >= delta:
        return -delta * w

    # 2. Intentamos calcular el paso de Newton completo (pB)
    try:
        pB = LA.solve(B, -g)
    except LA.LinAlgError:
        # Si B es singular o casi singular, nos conformamos con el punto de Cauchy
        return pU

    # Si el paso completo de Newton está dentro de la región, ¡lo tomamos!
    if LA.norm(pB) <= delta:
        return pB

    # 3. Intersección Dogleg: El paso de Newton está fuera, pero el Cauchy adentro.
    # Buscamos el punto en la línea entre pU y pB que cruza la frontera de radio delta.
    u = pU
    v = pB - pU
    a = np.dot(v, v)
    b = 2 * np.dot(u, v)
    c = np.dot(u, u) - delta**2

    # Resolvemos la ecuación cuadrática a*t^2 + b*t + c = 0
    discriminante = max(0, b**2 - 4*a*c) 
    alpha_s = (-b + np.sqrt(discriminante)) / (2*a)
    
    return u + alpha_s * v
def _SR1(y,s,B):
    """"
    Implementacion de la formula de SR1.
    """
    aux = y-B@s
    den = np.dot(aux,s)  

    if abs(den) >= 1e-8 * LA.norm(aux) * LA.norm(s):
        return B + np.outer(aux, aux) / den
    else:
        return B
    
    
def mRC_SR1(f, x0, maxDelta=16,imax = 1000, tol=1e-5, delta=0.1, dogLeg=False):
    eta = 0.01
    x = x0.copy()

    # Inicializamos el gradiente y la Hessiana exacta SOLO en el primer paso
    g = apGrad(f, x)
    B = apHess(f, x)
    for i in range(imax):
        if(LA.norm(g,np.inf)) <= tol:
            print("Happy iteration in "+str(i))
            break


        #aproximate solution of the model
        #Se decide como se va a calcular el punto
        if dogLeg:
            p=_pDogLeg_Seguro(g,delta,B)
        else:
            p= _pCauchy(g, delta, B)
        

        #Comprar reducciones del modelo con la funcion
        rho =( f(x)-f(x+p))/(np.dot(-p,g+B@(0.5*p)))

        #Decision en base a la reduccion 
        if rho < 0.25:
            delta *= 0.25
        elif rho >0.75 and LA.norm(p)>0.95*delta:
            delta = min(2*delta, maxDelta)

        if( rho > eta):
            x_new = x + p
            g_new = apGrad(f, x_new)
            
            # Preparamos s y y para la actualización SR1
            s = x_new - x
            y = g_new - g
            
            # Actualizamos la aproximación de la Hessiana
            B = _SR1( y,s,B)
            
            # Avanzamos al siguiente punto
            x = x_new
            g = g_new

    return x


