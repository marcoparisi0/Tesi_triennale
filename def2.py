import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse
import scipy.integrate as integrate
from scipy.constants import hbar
from scipy.special import sph_harm as Y  #Y(m[array di interi],l[uguale],phi[array di float,theta[],*, diff_n)
from scipy.special import spherical_jn as Jv  #jv(v[array],z [array anche complesso],derivative=False o True)
from scipy.optimize import fsolve
from scipy.optimize import brentq
from matplotlib.animation import FuncAnimation


R= 5*(10**(-7)) #m  raggio sfera
rho= 1850
vl=4226
vt=2530
rap= vt/vl
n_r=1.44


eps = 1e-12
r=np.linspace(eps,R,101)  #101 perchè in questo modo ho che il punto medio  è 50 (mi serve per il grafico)
theta=np.linspace(eps,math.pi-eps, 101)
phi=np.linspace(0,2*math.pi,101)
RRR , PHI, TETA = np.meshgrid(r,phi, theta, indexing='ij') # !!! LE TRE DIMENSIONI DI OGNUNO, SONO r phi e theta 


def psi(l,x):
    return ((-1)**l)*pow(x,-l)*Jv(l,x)
   
def de_psi(l,x):
    return ((-1)**l)*pow(x,-l)*(Jv(l,x,True)-(Jv(l,x)*l/x))


def f(hR):

    kR=hR/rap
    h=hR/R
    k=kR/R
    #tutto in funz di hR

    if l==0:
        return psi(l,hR)+(4*hR*de_psi(l,hR)/(kR)**2)
    else:
        #ho riscritto k e h in funzione di rap e hR
        al=((k**2)*(R**2)*psi(l,hR) + 2*(l-1)*psi(l-1, hR))/((2*l +1)*(h**2)) #okay
        bl=-( (k/h)**2 * psi(l,hR)  + 2*(l+2)*de_psi(l,hR)/(hR) ) /(2*l+1)
        cl=psi(l,kR)*(kR)**2 + (2*l - 2)*psi(l-1, kR)
        dl=(k**2)*l*(psi(l,kR) + de_psi(l, kR)*2*(l+2)/(kR))/(l+1)
        
        
        """
        al=(((hR/rap)**2)*psi(l,hR) + 2*(l-1)*psi(l-1, hR))/((2*l +1)*((hR/R)**2)) #okay
        bl=-( (1/rap)**2 * psi(l,hR)  + 2*(l+2)*de_psi(l,hR)/(hR) ) /(2*l+1)
        cl=psi(l,hR/rap)*(hR/rap)**2 + (2*l - 2)*psi(l-1, hR/rap)
        dl=((hR/(R*rap))**2)*l*(psi(l,hR/rap) + de_psi(l, hR/rap)*2*(l+2)/(hR/rap))/(l+1)
        """
        return al*dl - bl*cl

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
        
        dr=(r[1]-r[0])
        dtheta = (theta[1]-theta[0])
        dphi = (phi[1]-phi[0])
       
        dY_phi=1j*m*Y_griglia
        dY_theta= (m*Y_griglia/np.tan(TETA))+ (np.sqrt(l*(l+1)- m*(m+1))*np.exp(-1j*PHI)*Y(m+1,l,PHI,TETA))
        
        dxW=pow(RRR,l-1)*(np.sin(TETA)*np.cos(PHI)*l*Y_griglia + np.cos(TETA)*np.cos(PHI)*dY_theta - np.sin(PHI)*dY_phi/np.sin(TETA))
        dyW=pow(RRR,l-1)*(np.sin(TETA)*np.sin(PHI)*l*Y_griglia + np.cos(TETA)*np.sin(PHI)*dY_theta + np.cos(PHI)*dY_phi/np.sin(TETA))
        dzW=pow(RRR,l-1)*(np.cos(TETA)*l*Y_griglia - np.sin(TETA)*dY_theta)
        
        #in questo momento sto prendendo il massimo delle componenti, dato che c'è un coseno a moltiplicare
        
        u=(-(psi(l,h*RRR)+h*RRR*de_psi(l,h*RRR)/(2*l + 1))*(dxW)/(h**2) + de_psi(l,h*RRR)*(RRR*dxW - W*np.sin(TETA)*np.cos(PHI)*(2*l +1))/(h*(2*l +1)) + psi(l-1, k*RRR)*brapp*dxW - l*brapp*psi(l+1,k*RRR)*k*k*(RRR*RRR*dxW - W*np.sin(TETA)*np.cos(PHI)*(2*l + 1)*RRR)/(l+1))
        v=(-(psi(l,h*RRR)+h*RRR*de_psi(l,h*RRR)/(2*l + 1))*(dyW)/(h**2) + de_psi(l,h*RRR)*(RRR*dyW - W*np.sin(TETA)*np.sin(PHI)*(2*l +1))/(h*(2*l +1)) + psi(l-1, k*RRR)*brapp*dyW - l*brapp*psi(l+1,k*RRR)*k*k*(RRR*RRR*dyW - W*np.sin(TETA)*np.sin(PHI)*(2*l + 1)*RRR)/(l+1))
        w=(-(psi(l,h*RRR)+h*RRR*de_psi(l,h*RRR)/(2*l + 1))*(dzW)/(h**2) + de_psi(l,h*RRR)*(RRR*dzW - W*np.cos(TETA)*(2*l +1))/(h*(2*l +1)) + psi(l-1, k*RRR)*brapp*dzW - l*brapp*psi(l+1,k*RRR)*k*k*(RRR*RRR*dzW - W*np.cos(TETA)*(2*l + 1)*RRR)/(l+1))
        

    
    return np.array([u,v, w])

lll=np.array([0,2,3,4,5,6])
tuot=0
for l in lll:
    if l==1:
        hr0=1e-7
    else:                                    
        hr0=1e-3
        hrs = np.linspace(hr0,20, 50000)   
    fvals = np.array([f(w) for w in hrs])
   
    hR_roots= np.empty(0)
    
    for i in range(1, len(fvals)):
        if fvals[i-1]*fvals[i] <0:
            a=hrs[i-1]
            b=hrs[i]
            rut=brentq(f,a,b)
            hR_roots = np.append(hR_roots, rut)
            
            
    omegas_s=np.sort(hR_roots)*vl/R
   
    m=0
    n=0
    freq=omegas_s[n]
    h=freq/vl
    k=freq/vt
    hR=h*R
    kR=k*R
    
    s=sss(m)*np.cos(freq)
    inte_r=integrate.simpson(RRR*RRR*np.sin(TETA)*(abs(s[0])**2 + abs(s[1])**2 + abs(s[2])**2),r,axis=0)
    inte_te=integrate.simpson(inte_r,theta,axis=1)
    inte_fi=integrate.simpson(inte_te,phi)
    
    A=np.sqrt(1/(rho*inte_fi))
    s_s=s*A
    
    Cps=np.empty(0)
    qBS=4*math.pi*n_r/(532*pow(10,-9))
    qqs=np.linspace(0,qBS,500)
    

    def Cp(q):
        
        
        igr=np.exp(-1j*(q*RRR*np.cos(TETA)))*RRR*RRR*s_s[2]*np.sin(TETA)
        I_r=integrate.simpson(igr,r,axis=0)   #ricorda che per l'indexing ho che 0 è r, 1 phi, 2 teta
        I_te=integrate.simpson(I_r,theta,axis=1)           #integrando riduco le variabili quindi ora phi è 0 e teta è 1
        #I_fi=integrate.simpson(I_te,phi,axis=0)
        I_fi = I_te[0] * 2 * math.pi
        # Prendo il primo valore tanto sono tutti uguali
        
        return q*q*(abs(I_fi))**2
    
        
    #plot in funzione di qR
    for i in qqs:
        Cps=np.append(Cps,Cp(i))
    In=Cps/(freq**2)
    plt.plot(qqs*R,In, label=f"modo (1.{l})")
    plt.xlabel("qR")
    plt.ylabel("Intensità [U.A.]")
plt.legend()
plt.show()



"""
    #plot in funzione delle frequenze
    qbs=4*math.pi*1.44/(532*pow(10,-9))
    #qbs= 3.3/R
    freqzzz=np.linspace(10**10, 8*(10**10), 10000)
    sigma = 0.015*freq
    In=Cp(qbs)/(freq**2)
    spectrum = In*np.exp(-(freqzzz-freq)**2/(2*sigma**2))
    tuot=tuot +spectrum
    plt.plot(freqzzz,spectrum, label=f"modo (1.{l})")
plt.plot(freqzzz, tuot+ 1e-29, color="red", label="tot")
plt.legend()
plt.show()
"""
        
