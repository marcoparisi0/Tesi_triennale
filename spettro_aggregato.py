import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse
import scipy.integrate as integrate
from scipy.constants import hbar
from scipy.special import sph_harm as Y  
from scipy.special import spherical_jn as Jv 
from scipy.optimize import brentq




data_I = np.genfromtxt("aggregato_Potenza=100mW_pin=300_pout=700_2525scans.DAT", skip_header=10, invalid_raise=False)
data_sub=np.genfromtxt("substrato_Potenza=100mW_pin=300_pout=700_1000scans.DAT", skip_header=10, invalid_raise=False)*1.5 #per "pareggiare" i conteggi
#print(len(data_I))
#print(len(data_sub))

c=299792458
d=16*(pow(10,-3))
FSR=c/(2*d)
#FSR=16.9603*1e9
#print(FSR)
x=np.linspace(-FSR,FSR,len(data_I))
#x=np.linspace(-FSR/2,FSR/2,len(data_I))


plt.plot(x,data_I, color="red", label="instensità aggregato sul substrato")
plt.plot(x, data_sub, color='black', label='intensità substrato')
plt.legend()
plt.xlabel("freq")
plt.ylabel("conteggio")
plt.show()

I_sott=data_I-(data_sub)
I_sott[I_sott < 0] = 0
plt.plot(x, I_sott, color="green")
plt.show()


R= 5*(10**(-7)) #m  raggio sfera
rho= 1850
n_r=1.41 #CALCOLATO DALL'ARTICOLO


###-----------------------------------------------------------------------PARAMETRI MODIFICABILI--------------------------------------------------------------------------------------------------------
#con questi (1,2) mi viene in corripsondenza del picco più alto 
#vt=4050
#vl=6430
#rap = vt/vl

#con questi fitta quasi bene ma 0,2 non è il più alto
#vl= 4700
#rap=0.57


#con questi sembra tutto traslato di 1 GHz
#vl=4450
#rap=0.85

#MIGLIORI PER ORA
#rap=0.61
#vl=4200

#rap=0.49  #deve essere tra 0.55 e 0.47 perchè è una zona in cui 0,3 ha inensità max , vl lo regolo se voglio aumentare o diminuire la frequenza (ma lo tocco poco)
#vl=5000

#Ritorno ai primi parametri perchè quasi sicuramente il primo "picco" non è effettivamente un picco
#rap=0.7
#vl=6000  #QUASI PERFETTI

#rap=0.68
#vl=6100


#ECCOLI -->  anti-Stokes
rap=0.69
vl=6050

###------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

    if l==0:
        return psi(l,hR)+(4*hR*de_psi(l,hR)/(kR)**2)
    else:
        al=(((hR/rap)**2)*psi(l,hR) + 2*(l-1)*psi(l-1, hR))/((2*l +1)*((hR/R)**2)) #okay
        bl=-( (1/rap)**2 * psi(l,hR)  + 2*(l+2)*de_psi(l,hR)/(hR) ) /(2*l+1)
        cl=psi(l,hR/rap)*(hR/rap)**2 + (2*l - 2)*psi(l-1, hR/rap)
        dl=((hR/(R*rap))**2)*l*(psi(l,hR/rap) + de_psi(l, hR/rap)*2*(l+2)/(hR/rap))/(l+1)
        
        return al*dl - bl*cl

def sss(m):
    
    if l==0 :
        u=-np.sin(TETA)*np.cos(PHI)*de_psi(l,h*RRR)/(h**2)
        v=-np.sin(TETA)*np.sin(PHI)*de_psi(l,h*RRR)/(h**2)
        w=-np.cos(TETA)*de_psi(l,h*RRR)/(h**2)
    else:
        bl=-( (k/h)**2 * psi(l,hR)  + 2*(l+2)*de_psi(l,hR)/(hR) ) /(2*l+1)
        dl=(k**2)*l*(psi(l,kR) + de_psi(l, kR)*2*(l+2)/(kR))/(l+1)
        brapp=-bl/dl
        
        Y_griglia = Y(m, l, PHI, TETA)
        W= pow(RRR,l)*Y_griglia #armoniche solide
        P=brapp*W
        
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



lll=np.array([0,1,2,3])
tuot=0
#freqzzz=np.linspace(0,FSR/2,10000)
freqzzz=np.linspace(0,FSR,10000)
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
    print(omegas_s/(2*math.pi))
    m=0
    for n in range(2):
        if n==0 and l==1:
            continue
        
        freq=omegas_s[n]
        h=freq/vl
        k=h/rap
        hR=h*R
        kR=k*R
        
        s=sss(m)/2  #ho moltiplicato per il mosulo cos^2 mediato nel tempo
        inte_r=integrate.simpson(RRR*RRR*np.sin(TETA)*(abs(s[0])**2 + abs(s[1])**2 + abs(s[2])**2),r,axis=0)
        inte_te=integrate.simpson(inte_r,theta,axis=1)
        inte_fi=integrate.simpson(inte_te,phi)
        
        A=np.sqrt(1/(rho*inte_fi))
        s_s=s*A
        
        Cps=np.empty(0)
        qBS=4*math.pi*n_r/(532*pow(10,-9))
        qqs=np.linspace(0,qBS,300)
        
        
        def In(q):
            
            
            igr=np.exp(-1j*(q*RRR*np.cos(TETA)))*RRR*RRR*s_s[2]*np.sin(TETA)
            I_r=integrate.simpson(igr,r,axis=0)   #ricorda che per l'indexing ho che 0 è r, 1 phi, 2 teta
            I_te=integrate.simpson(I_r,theta,axis=1)           #integrando riduco le variabili quindi ora phi è 0 e teta è 1
            #I_fi=integrate.simpson(I_te,phi,axis=0)
            I_fi = I_te[0] * 2 * math.pi
            # Prendo il primo valore tanto sono tutti uguali
            
            return (q*q*(abs(I_fi))**2)/(freq**2)

        
        #plot in funzione delle frequenze
        qBS=4*math.pi*n_r/(532*pow(10,-9))
        #qBS=5/R  #HO GUARDATO DAL GRAFICO  I(qR) la parte in cui (1,3) avesse intensità maggiore, dato che con questi parametri fitta bene tranne le intensità
        sigma = 10**8
        #q_int=np.linspace(0.92*qBS,qBS, 100)
        #In_val=np.array([In(j) for j in q_int])
        #In_BS=integrate.simpson(In_val*q_int,q_int) #qui moltiplica intensità per 1e19
        In_BS=In(qBS)
        spectrum = In_BS*np.exp(-(freqzzz-(freq/(2*math.pi)))**2/(2*sigma**2))  #Stokes
        tuot=tuot +spectrum
        plt.plot(-freqzzz,spectrum*pow(10,34)/2, label=f"modo ({n+1},{l})")

plt.plot(x, I_sott, color="red", label="exp")

#plt.plot(freqzzz, tuot*1e35, color="green", label="tot")
plt.legend()
plt.show()

        
