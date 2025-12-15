import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse
import scipy.integrate as integrate
from scipy.special import sph_harm as Y  #Y(m[array di interi],l[uguale],phi[array di float,theta[],*, diff_n)
from scipy.special import spherical_jn as Jv  #jv(v[array],z [array anche complesso],derivative=False o True)
from scipy.optimize import fsolve
from scipy.optimize import brentq


"""
Il  codice è scritto in modo da selezionare il modo che si vuole visualizzare. Potrei anche fare una "raccolta" di modi

L'unico aggiustamento da fare magari può essere di sviluppare analiticamente alcune cose ("semplificare" sviluppando e sostituendo alcune cose angolari come armoniche sferiche ecc)
"""

def parse_arguments():
    parser = argparse.ArgumentParser(description='Campo di deformazion modi di vibrazione di una sfera')
    parser.add_argument('-s', '--singolo_m',action='store_true', help="Mostra i modi sferoidali")
    parser.add_argument('-t', '--tutti_m',action='store_true', help="Mostra i modi torsionali")

    return  parser.parse_args()

args = parse_arguments()



R= 2*(10**(-9)) #m  raggio sfera
rho= 2.5*pow(10,3) #kg/m3
lam= 3.4*pow(10,10) #Pa
mu= 3.1*pow(10,10) #Pa
vl=np.sqrt((lam+ 2*mu)/rho)
vt=np.sqrt(mu/rho)
rap= vt/vl

l=int(input('inserisci il modo angolare '))


def psi(l,x):
    return ((-1)**l)*pow(x,-l)*Jv(l,x)
   
def de_psi(l,x):
    return ((-1)**l)*pow(x,-l)*(Jv(l,x,True)-(Jv(l,x)*l/x))


#--------------------------------------------------- frequenze modi sferoidali-------------------------------------------------------------------------------------------------

def f(hR):
    """
    uso come variabile hR perchè sennò avrei valori piccoli per le ordinate e non me li fa
    poi so che h=omega/vl e che h=hR/R
        
    in questo modo la mia variabile non è un qualcosa di troppo piccolo (non sono due variabili perchè hR a posso scrivere in funzione di kR
    """
    kR=hR/rap
    h=hR/R
    k=kR/R

    if l==0:
        return psi(l,hR)+(4*hR*de_psi(l,hR)/(kR)**2)
    else:
        
        al=((k**2)*(R**2)*psi(l,hR) + 2*(l-1)*psi(l-1, hR))/((2*l +1)*(h**2)) #okay
        bl=-( (k/h)**2 * psi(l,hR)  + 2*(l+2)*de_psi(l,hR)/(hR) ) /(2*l+1)
        cl=psi(l,kR)*(kR)**2 + (2*l - 2)*psi(l-1, kR)
        dl=(k**2)*l*(psi(l,kR) + de_psi(l, kR)*2*(l+2)/(kR))/(l+1)
        """
        ho riscritto k e h e le ho  lasciate  perchè avevo già scritto le a b c d  con questio valori isolati e non ho voglia di riscriverle
        """
        
        return al*dl - bl*cl
    
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

if args.singolo_m == True:
    m=int(input('inserisci la terza componente del modo angolare'))

if args.tutti_m ==True:
    mvals=np.arange(-l,l+1)

n=int(input('inserisci il modo radiale '))-1       #L’indice n è il modo radiale; si ottiene dall’ordine delle radici dell’equazione.
freq=omegas_s[n]
h=freq/vl
k=freq/vt
hR=h*R
kR=k*R

r=np.linspace(1e-12,R,100)
theta= np.linspace(1e-8,math.pi, 100)
phi=np.linspace(1e-8,2*math.pi,100)
RRR , PHI, TETA = np.meshgrid(r,phi, theta, indexing='ij') # !!! LE TRE DIMENSIONI DI OGNUNO, SONO r phi e theta 
"""
questo perchè mi servono 3 matrici 3D di ogni coordinata, meshgrid costruisce la griglia cartesiana del dominio in coordinate sferiche, cioè tipo RRR[i,j,k] (e anche gli altri) è un preciso valore di r[i] in quel punto dello spazio
RRR, TETA, PHI sono tutti i punti del reticolo sferico
"""

def sss(m):
    
    if l==0 :
        u=-np.sin(TETA)*np.cos(PHI)*de_psi(l,h*RRR)/(h**2)
        v=-np.sin(TETA)*np.sin(PHI)*de_psi(l,h*RRR)/(h**2)
        w=-np.cos(TETA)*de_psi(l,h*RRR)/(h**2)
        
    else:
        bl=-( (k/h)**2 * psi(l,hR)  + 2*(l+2)*de_psi(l,hR)/(hR) ) /(2*l+1)
        dl=(k**2)*l*(psi(l,kR) + de_psi(l, kR)*2*(l+2)/(kR))/(l+1)
        brapp=bl/dl
        
        Y_griglia = Y(m, l, PHI, TETA)
        W= pow(RRR,l)*Y_griglia #armoniche solide
        P=brapp*W
        
        dr=r[1]-r[0]
        dtheta = theta[1]-theta[0]
        dphi = phi[1]-phi[0]
        dY_r, dY_phi,  dY_theta = np.gradient(Y_griglia, dr, dphi , dtheta) #usa la differenza centrale 
        
        dxW=pow(RRR,l-1)*(np.sin(TETA)*np.cos(PHI)*l*Y_griglia + np.cos(TETA)*np.cos(PHI)*dY_theta - np.sin(PHI)*dY_phi/np.sin(TETA))
        dyW=pow(RRR,l-1)*(np.sin(TETA)*np.sin(PHI)*l*Y_griglia + np.cos(TETA)*np.sin(PHI)*dY_theta + np.cos(PHI)*dY_phi/np.sin(TETA))
        dzW=pow(RRR,l-1)*(np.cos(TETA)*l*Y_griglia - np.sin(TETA)*dY_theta)
        
        #NELL'UNITÀ DI TEMPO
        
        u=np.cos(freq)*(-(psi(l,h*RRR)+h*RRR*de_psi(l,h*RRR)/(2*l + 1))*(dxW)/(h**2) + de_psi(l,h*RRR)*(RRR*dxW - W*np.sin(TETA)*np.cos(PHI)*(2*l +1))/(h*(2*l +1)) + psi(l-1, k*RRR)*brapp*dxW - l*brapp*psi(l+1,k*RRR)*k*k*(RRR*RRR*dxW - W*np.sin(TETA)*np.cos(PHI)*(2*l + 1)*RRR)/(l+1))
        v=np.cos(freq)*(-(psi(l,h*RRR)+h*RRR*de_psi(l,h*RRR)/(2*l + 1))*(dyW)/(h**2) + de_psi(l,h*RRR)*(RRR*dyW - W*np.sin(TETA)*np.sin(PHI)*(2*l +1))/(h*(2*l +1)) + psi(l-1, k*RRR)*brapp*dyW - l*brapp*psi(l+1,k*RRR)*k*k*(RRR*RRR*dyW - W*np.sin(TETA)*np.sin(PHI)*(2*l + 1)*RRR)/(l+1))
        w=np.cos(freq)*(-(psi(l,h*RRR)+h*RRR*de_psi(l,h*RRR)/(2*l + 1))*(dzW)/(h**2) + de_psi(l,h*RRR)*(RRR*dzW - W*np.cos(TETA)*(2*l +1))/(h*(2*l +1)) + psi(l-1, k*RRR)*brapp*dzW - l*brapp*psi(l+1,k*RRR)*k*k*(RRR*RRR*dzW - W*np.cos(TETA)*(2*l + 1)*RRR)/(l+1))
        
    uu=abs(u)
    vv=abs(v)
    ww=abs(w)
    
    inte_r=integrate.simpson(r*r*(uu**2 + vv**2 +ww**2),r)
    inte_te=integrate.simpson(inte_r*np.sin(theta),theta)
    inte_fi=integrate.simpson(inte_te,phi)
    
    A=np.sqrt(1/(rho*inte_fi))
    
    return (A*uu , A*vv, A*ww)

#vari vettori spostamento per vari valori di r teta  phi,relativi ad un modo specifico --->sostanzialmente ho calcolato il campo di spostamento cartesiano su una griglia sferica
#è una tupla a 3 elementi, ognuno è un array 3D

if args.singolo_m == True:
    s_s=sss(m)
    u_xz, v_xz, w_xz = s_s[0][:, 0, :], s_s[1][:, 0,:],s_s[2][:, 0,:]
    RR=RRR[:,0,:]
    TT=TETA[:,0,:]
    XX=RR*np.sin(TT)
    ZZ=RR*np.cos(TT)
    
    norm=np.sqrt(u_xz[::5,::5]**2 + w_xz[::5,::5]**2)
    fig1, ax1 = plt.subplots()
    ax1.quiver(XX[::5,::5], ZZ[::5,::5],u_xz[::5,::5],w_xz[::5,::5], norm, cmap="tab20b",  headwidth = 2)
    ax1.set_xlim([-R, R])
    ax1.set_ylim([-R, R])
    ax1.set_aspect('equal', 'box')
    plt.show()

    
if args.tutti_m == True:
    Cps=np.zeros(100)
    qqs=np.linspace(0,37*10**7,100)
    for m in mvals:
        s_s=sss(m)
        def Cp(q):
            I_r=integrate.simpson(np.exp(-1j*(q*RRR*np.cos(TETA)))*RRR*RRR*s_s[2],r,axis=0)   #ricorda che per l'indexing ho che 0 è r, 1 phi, 2 teta
            I_te=integrate.simpson(I_r*np.sin(TETA),theta,axis=1)           #integrando riduco le variabili quindi ora phi è 0 e teta è 1
            I_fi=integrate.simpson(I_te,phi,axis=0)
            return q*q*pow(abs(I_fi),2)
        Cps=Cps+Cp(qqs)
    In=Cps/(freq**2)
    plt.plot(qqs*R,In)
    plt.show()
    





### troppo tempo di calcolo e risultati errati 
















#--------------------------------------------------------frequenze modi torsionali------------------------------------------------------------------------
"""
if args.torsionale == True:
    
    def pl(kR):
        return (l-1)*psi(l,kR) + kR*psi(l,kR)

    krs = np.linspace(1e-6,30, 5000)   
    pvals = np.array([pl(w) for w in krs])
    
    plt.plot(krs,pvals)
    plt.grid(True)
    plt.show()
    
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

    
"""
