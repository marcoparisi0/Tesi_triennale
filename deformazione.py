"""
cose da correggere: trovare il modo per fare l=0
trovare un modo più compatto di scriverlo (?)
 fare il plot del lato negativo 
ricontrollare risultato campo di spostamento 
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse
from scipy.special import sph_harm as Y  #Y(m[array di interi],l[uguale],phi[array di float,theta[],*, diff_n)
from scipy.special import spherical_jn as Jv  #jv(v[array],z [array anche complesso],derivative=False o True)
from scipy.optimize import fsolve
from scipy.optimize import brentq

"""
Il  codice è scritto in modo da selezionare il modo che si vuole visualizzare
Potrei anche fare una "raccolta" di modi
"""

def parse_arguments():
    parser = argparse.ArgumentParser(description='Campo di deformazion modi di vibrazione di una sfera')
    parser.add_argument('-s', '--sferoidale',action='store_true', help="Mostra i modi sferoidali")
    parser.add_argument('-t', '--torsionale',action='store_true', help="Mostra i modi torsionali")

    return  parser.parse_args()

args = parse_arguments()



R= 10**(-8) #m  raggio sfera

rho= 2330 #kg/m3
lam= 5*pow(10,10) #Pa
mu= 6*pow(10,10) #Pa
l=int(input('inserisci il modo angolare (>= 1) '))
m=int(input('inserisci la terza componente del modo andolare'))


def psi(l,x):
    return ((-1)**l)*pow(x,-l)*Jv(l,x)
   
def de_psi(l,x):
    return ((-1)**l)*pow(x,-l)*(Jv(l,x,True)-(Jv(l,x)*l/x))
    
vl=np.sqrt((lam+ 2*mu)/rho)
vt=np.sqrt(mu/rho)
rap= vt/vl

def defos(om):
    
    n=int(input('inserisci il modo radiale '))-1
    freq=om[n]
    h=freq/vl
    k=freq/vt
    hR=h*R
    kR=k*R
    bl=-( (k/h)**2 * psi(l,hR)  + 2*(l+2)*de_psi(l,hR)/(hR) ) /(2*l+1)
    dl=(k**2)*l*(psi(l,kR) + de_psi(l, kR)*2*(l+2)/(kR))/(l+1)
    brapp=bl/dl

    r=np.linspace(1e-12,R,100)
    theta= np.linspace(1e-12,math.pi, 100)
    phi=np.linspace(1e-12,2*math.pi,100)
    RRR , TETA, PHI = np.meshgrid(r,theta, phi, indexing='ij')
    """
    questo perchè mi servono 3 matrici 3D di ogni coordinata , costruisce la griglia cartesiana del dominio in coordinate sferiche, cioè tipo RRR[i,j,k] (e anche gli altri) è un preciso valore di r[i] in quel punto dello spazio
    RRR, TETA, PHI sono tutti i punti del reticolo sferico
    """

    Y_griglia = Y(m, l, PHI, TETA)
    W= pow(RRR,l)*Y_griglia #armoniche solide
    P=brapp*W
    
    dr=r[1]-r[0]
    dtheta = theta[1]-theta[0]
    dphi = phi[1]-phi[0]
    dY_r, dY_theta, dY_phi = np.gradient(Y_griglia, dr, dtheta, dphi)
    
    dxW=pow(RRR,l-1)*(np.sin(TETA)*np.cos(PHI)*l*Y_griglia + np.cos(TETA)*np.cos(PHI)*dY_theta - np.sin(PHI)*dY_phi/np.sin(TETA))
    dyW=pow(RRR,l-1)*(np.sin(TETA)*np.sin(PHI)*l*Y_griglia + np.cos(TETA)*np.sin(PHI)*dY_theta + np.cos(PHI)*dY_phi/np.sin(TETA))
    dzW=pow(RRR,l-1)*(np.cos(TETA)*l*Y_griglia - np.sin(TETA)*dY_theta)

   # dxWr= dxW/pow(RRR,2*l+1) - W*np.sin(TETA)*np.cos(PHI)*pow(RRR, -2*l - 2)*(2+l +1)
   # dyWr= dyW/pow(RRR, 2*l + 1) - W*np.sin(TETA)*np.sin(PHI)*(2*l +1) * pow(RRR, -(2*l +2))
   # dzWr=dzW/pow(RRR, 2*l +1) - np.cos(TETA)*(2*l +1)*pow(RRR, -(2*l +2))

    
    

    u=np.cos(freq)*(-(psi(l,h*RRR)+h*RRR*de_psi(l,h*RRR)/(2*l + 1))*(dxW)/(h**2) + de_psi(l,h*RRR)*(RRR*dxW - W*np.sin(TETA)*np.cos(PHI)*(2*l +1))/(h*(2*l +1)) + psi(l-1, k*RRR)*brapp*dxW - l*brapp*psi(l+1,k*RRR)*k*k*(RRR*RRR*dxW - W*np.sin(TETA)*np.cos(PHI)*(2*l + 1)*RRR)/(l+1))
    v=np.cos(freq)*(-(psi(l,h*RRR)+h*RRR*de_psi(l,h*RRR)/(2*l + 1))*(dyW)/(h**2) + de_psi(l,h*RRR)*(RRR*dyW - W*np.sin(TETA)*np.sin(PHI)*(2*l +1))/(h*(2*l +1)) + psi(l-1, k*RRR)*brapp*dyW - l*brapp*psi(l+1,k*RRR)*k*k*(RRR*RRR*dyW - W*np.sin(TETA)*np.sin(PHI)*(2*l + 1)*RRR)/(l+1))
    w=np.cos(freq)*(-(psi(l,h*RRR)+h*RRR*de_psi(l,h*RRR)/(2*l + 1))*(dzW)/(h**2) + de_psi(l,h*RRR)*(RRR*dzW - W*np.cos(TETA)*(2*l +1))/(h*(2*l +1)) + psi(l-1, k*RRR)*brapp*dzW - l*brapp*psi(l+1,k*RRR)*k*k*(RRR*RRR*dzW - W*np.cos(TETA)*(2*l + 1)*RRR)/(l+1))

    return np.real(u) ,np.real(v) ,np.real(w)

"""
da fare
(devo capire cosa sono le chi)

def defot(om):
    n=int(input('inserisci il modo radiale '))-1
    freq=om[n]
    h=freq/vl
    k=freq/vt
    hR=h*R
    kR=k*R
"""



#--------------------------------------------------- frequenze modi sferoidali-------------------------------------------------------------------------------------------------
if args.sferoidale==True:
    
    def f(hR):
        """
        uso come variabile hR perchè sennò avrei valori piccoli per le ordinate e non me li fa
        poi so che h=omega/vl e che h=hR/R
        
        in questo modo la mia variabile non è un qualcosa di troppo piccolo (non sono due variabili perchè hR a posso scrivere in funzione di kR
        """
        
        kR=hR/rap
        h=hR/R
        k=kR/R
        al=((k**2)*(R**2)*psi(l,hR) + 2*(l-1)*psi(l-1, hR))/((2*l +1)*(h**2)) #okay
        bl=-( (k/h)**2 * psi(l,hR)  + 2*(l+2)*de_psi(l,hR)/(hR) ) /(2*l+1)
        cl=psi(l,kR)*(kR)**2 + (2*l - 2)*psi(l-1, kR)
        dl=(k**2)*l*(psi(l,kR) + de_psi(l, kR)*2*(l+2)/(kR))/(l+1)
        """
        ho riscritto k e h e le ho  lasciate  perchè avevo già scritto le a b c d  con questio valori isolati e non ho voglia di riscriverle
        """
        
        return al*dl - bl*cl
    
    
    #L’indice n è il modo radiale, cioè quante volte il campo radiale si annulla dentro la sfera; si ottiene dall’ordine delle radici dell’equazione.
    
    
    hrs = np.linspace(1e-6,30, 5000)   
    fvals = np.array([f(w) for w in hrs])
    """
    plt.plot(hrs,fvals)
    plt.grid(True)
    plt.show()
    """
    hR_roots= np.empty(0)
    
    for i in range(1, len(fvals)):
        if fvals[i-1]*fvals[i] <0:
            a=hrs[i-1]
            b=hrs[i]
            rut=brentq(f,a,b)
            hR_roots = np.append(hR_roots, rut)
            
            
    omegas_s=np.sort(hR_roots)*vl/R
    print(omegas_s)

    s_s=defos(omegas_s) #vari vettori spostamento per vari valori di r teta  phi,relativi ad un modo specifico --->sostanzialmente ho calcolato il campo di deformazione cartesiano su una griglia sferica
    #è una tupla a 3 elementi, ognuno è un array 3D


    r = np.linspace(1e-12, R, 100)
    theta = np.linspace(1e-12, math.pi, 100)
    phi = np.linspace(1e-12, 2 * math.pi, 100)
    RRR, TETA, PHI = np.meshgrid(r, theta, phi, indexing='ij')  # !!! LE TRE DIMENSIONI DI OGNUNO, SONO r theta e phi !!!!!!!!!!!!11
    u_xz, v_xz, w_xz = s_s[0][:, :, 0], s_s[1][:, :,0],s_s[2][:, :,0]
    RR=RRR[:,:,0]
    TT=TETA[:,:,0]
    XX=RR*np.sin(TT)
    ZZ=RR*np.cos(TT)

    fig1, ax1 = plt.subplots()
    ax1.quiver(XX[::5,::5], ZZ[::5,::5],u_xz[::5,::5]*3e9,w_xz[::5,::5]*3e9, units='width')
    # imposto  limiti coerenti con il raggio della sfera
    ax1.set_xlim([-R, R])
    ax1.set_ylim([-R, R])
    # Assicura che la scala degli assi sia uguale (cerchio non deformato)
    ax1.set_aspect('equal', 'box')
    plt.show()
#--------------------------------------------------------frequenze modi torsionali------------------------------------------------------------------------

if args.torsionale == True:
    
    def pl(kR):
        return (l-1)*psi(l,kR) + kR*psi(l,kR)

    krs = np.linspace(1e-6,30, 5000)   
    pvals = np.array([pl(w) for w in krs])
    """
    plt.plot(krs,pvals)
    plt.grid(True)
    plt.show()
    """
    kR_roots= np.empty(0)
    
    for j in range(1, len(pvals)):
        if pvals[j-1]*pvals[j] <0:
            a=krs[j-1]
            b=krs[j]
            rut=brentq(pl,a,b)
            kR_roots = np.append(kR_roots, rut)
            
            
    omegas_t=np.sort(kR_roots)*vt/R
    print(omegas_t)

    s_t=defo(omegas_t)

    
