import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse
from scipy.special import sph_harm as Y  #Y(n[array di interi],m[uguale],theta[array di float,phi[],*, diff_n)
from scipy.special import spherical_jn as Jv  #jv(v[array],z [array anche complesso],derivative=False o True)
from scipy.optimize import fsolve
from scipy.optimize import brentq



def parse_arguments():
    parser = argparse.ArgumentParser(description='Campo di deformazion modi di vibrazione di una sfera')
    parser.add_argument('-s', '--sferoidale',    action='store_true', help="Mostra i modi sferoidali")
    parser.add_argument('-t', '--torsionale',    action='store_true', help="Mostra i modi torsionali")

    return  parser.parse_args()

args = parse_arguments()



R= 10**(-7) #m  raggio sfera

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


#--------------------------------------------------- modi sferoidali-------------------------------------------------------------------------------------------------
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
    
    plt.plot(hrs,fvals)
    plt.grid(True)
    plt.show()
    
    hR_roots= np.empty(0)
    
    for i in range(1, len(fvals)):
        if fvals[i-1]*fvals[i] <0:
            a=hrs[i-1]
            b=hrs[i]
            rut=brentq(f,a,b)
            hR_roots = np.append(hR_roots, rut)
            
            
    omegas_s=np.sort(hR_roots)*vl/R
    print(omegas_s)
        

#--------------------------------------------------------modi torsionali------------------------------------------------------------------------

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

    
