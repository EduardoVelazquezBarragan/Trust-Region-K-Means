import numpy as np
import scipy.linalg as LA
from Lib_derivadas import apGrad, apHess


def pCauchy(g, delta,B):
    w = g/ LA.norm(g)

    alpha_u = 1/np.dot(w, B@w)
    if alpha_u >0 and abs(alpha_u)*LA.norm(g) <= delta:
        res= - alpha_u*g
    else:
        res = -delta*w

    return res

def pDogLeg(g, delta,B):
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
             #alpha_s =
             pDL    
    return pDL

def mRC(f, x0, maxDelta=16,imax = 1000, tol=1e-5, delta=0.1):
    eta = 0.01
    x = x0.copy()
    for i in range(imax):
        g = apGrad(f,x)
        if(LA.norm(g,np.inf)) <= tol:
            print("Happy iteration in "+str(i))
            break

        # construct the model
        B = apHess(f,x)

        #aproximate solution of the model
        p= pCauchy(g, delta, B)
        #p=pDogLeg(g, delta, B)


        #Comprar reducciones del modelo con la funcion
        rho =( f(x)-f(x+p))/(np.dot(-p,g+B@(0.5*p)))

        #Decision en base a la reduccion 
        if rho < 0.25:
            delta *= 0.25
        elif rho >0.75 and LA.norm(p)>0.95*delta:
            delta = min(2*delta, maxDelta)

        if( rho > eta):
            x=x+p
    return x


