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
    DogLeg seguro siguiendo las condiciones:

    1. Si Cauchy está en la frontera, tomar Cauchy.
    2. Si Newton está dentro del radio y es dirección de descenso, tomar Newton.
    3. Si Newton está fuera, verificar:
           (pB - pU)^T pU >= 0.
       Si se cumple, tomar el punto combinado DogLeg.
       Si no se cumple, tomar Cauchy.
    """

    norm_g = LA.norm(g)

    # Si el gradiente es cero, no hay dirección de descenso.
    if norm_g == 0:
        return np.zeros_like(g)

    w = g / norm_g
    gBg = np.dot(g, B @ g)

    # ---------------------------------------------------------
    # 1. Punto de Cauchy pU
    # ---------------------------------------------------------
    if gBg <= 0:
        # Curvatura no positiva: se toma Cauchy en la frontera.
        pU = -delta * w
    else:
        alpha_u = (norm_g ** 2) / gBg
        pU = -alpha_u * g

        # Si Cauchy se sale del radio, se trunca a la frontera.
        if LA.norm(pU) >= delta:
            pU = -delta * w

    # Si Cauchy está en la frontera, se toma Cauchy.
    if LA.norm(pU) >= (1.0 - 1e-12) * delta:
        return pU

    # ---------------------------------------------------------
    # 2. Paso de Newton / cuasi-Newton pB
    # ---------------------------------------------------------
    try:
        pB = LA.solve(B, -g)
    except LA.LinAlgError:
        # Si B es singular o casi singular, regresar a Cauchy.
        return pU

    # Verificamos si Newton es dirección de descenso.
    # Para minimización se requiere:
    #
    #     g^T pB < 0
    #
    es_descenso = np.dot(g, pB) < 0

    # Si Newton está dentro del radio y es dirección de descenso,
    # se toma Newton.
    if LA.norm(pB) <= delta and es_descenso:
        return pB

    # Si Newton no es dirección de descenso, no usamos DogLeg.
    if not es_descenso:
        return pU

    # ---------------------------------------------------------
    # 3. Punto combinado DogLeg
    # ---------------------------------------------------------
    v = pB - pU

    # Condición recomendada por el profesor:
    #
    #     (pB - pU)^T pU >= 0
    #
    if np.dot(v, pU) < 0:
        return pU

    # Buscamos alpha tal que:
    #
    #     ||pU + alpha (pB - pU)|| = delta
    #
    # con 0 <= alpha <= 1.
    a = np.dot(v, v)
    b = 2.0 * np.dot(pU, v)
    c = np.dot(pU, pU) - delta ** 2

    if a == 0:
        return pU

    discriminante = b ** 2 - 4.0 * a * c

    if discriminante < 0:
        return pU

    alpha_s = (-b + np.sqrt(discriminante)) / (2.0 * a)

    # Verificación de que el punto combinado realmente esté
    # en el segmento entre Cauchy y Newton.
    if alpha_s < 0 or alpha_s > 1:
        return pU

    pDL = pU + alpha_s * v

    return pDL

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
    eta = 0.1
    x = x0.copy()

    # Inicializamos el gradiente y la Hessiana exacta SOLO en el primer paso
    g = apGrad(f, x)
    B = apHess(f, x)
    iter=imax
    for i in range(imax):
        if(LA.norm(g,np.inf)) <= tol:
            iter=i
            #print("Happy iteration in "+str(i))
            break


        #aproximate solution of the model
        #Se decide como se va a calcular el punto
        if dogLeg:
            p=_pDogLeg_Seguro(g,delta,B)
        else:
            p= _pCauchy(g, delta, B)
        

        #Comprar reducciones del modelo con la funcion
        redf=( f(x)-f(x+p))
        redm=np.dot(-p,g+B@(0.5*p))
        rho =redf/redm

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

    return x,iter


