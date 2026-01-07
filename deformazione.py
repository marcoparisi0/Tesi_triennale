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


"""
Il  codice è scritto in modo da selezionare il modo che si vuole visualizzare. Potrei anche fare una "raccolta" di modi

L'unico aggiustamento da fare magari può essere di sviluppare analiticamente alcune cose ("semplificare" sviluppando e sostituendo alcune cose angolari come armoniche sferiche ecc)
"""

def parse_arguments():
    parser = argparse.ArgumentParser(description='Campo di deformazion modi di vibrazione di una sfera')
    parser.add_argument('-s', '--spostamento',action='store_true', help="Mostra il grafico del campo di spostamento")
    parser.add_argument('-a', '--animazione',action='store_true', help="Mostra il campo di spostamento in evoluzione temporale")
    parser.add_argument('-i', '--intensità',action='store_true', help="Mostra l'intensità Brillouin per modi con m=0")

    return  parser.parse_args()

args = parse_arguments()



R= 5*(10**(-7)) #m  raggio sfera
rho= 2.5*pow(10,3) #kg/m3
lam= 15*pow(10,9) #Pa
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
    #tutto in funz di hR

    if l==0:
        return psi(l,hR)+(4*hR*de_psi(l,hR)/(kR)**2)
    #elif l==1:
        #return kR*psi(l,kR)+2*de_psi(l,kR)
    else:
        """
        al=((k**2)*(R**2)*psi(l,hR) + 2*(l-1)*psi(l-1, hR))/((2*l +1)*(h**2)) #okay
        bl=-( (k/h)**2 * psi(l,hR)  + 2*(l+2)*de_psi(l,hR)/(hR) ) /(2*l+1)
        cl=psi(l,kR)*(kR)**2 + (2*l - 2)*psi(l-1, kR)
        dl=(k**2)*l*(psi(l,kR) + de_psi(l, kR)*2*(l+2)/(kR))/(l+1)
        
        ho riscritto k e h e le ho  lasciate  perchè avevo già scritto le a b c d  con questio valori isolati e non ho voglia di riscriverle
        """
        al=(((hR/rap)**2)*psi(l,hR) + 2*(l-1)*psi(l-1, hR))/((2*l +1)*((hR/R)**2)) #okay
        bl=-( (1/rap)**2 * psi(l,hR)  + 2*(l+2)*de_psi(l,hR)/(hR) ) /(2*l+1)
        cl=psi(l,hR/rap)*(hR/rap)**2 + (2*l - 2)*psi(l-1, hR/rap)
        dl=((hR/(R*rap))**2)*l*(psi(l,hR/rap) + de_psi(l, hR/rap)*2*(l+2)/(hR/rap))/(l+1)
        
        return al*dl - bl*cl
    
hrs = np.linspace(1e-7,20, 50000)   
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
#print(omegas_s)
#print(np.sort(hR_roots))

n=int(input('inserisci il modo radiale'))-1       #L’indice n è il modo radiale; si ottiene dall’ordine delle radici dell’equazione.
freq=omegas_s[n]
h=freq/vl
k=freq/vt
hR=h*R
kR=k*R

eps = 1e-12
r=np.linspace(eps,R,101)  #101 perchè in questo modo ho che il punto medio  è 50 (mi serve per il grafico)
theta=np.linspace(eps,math.pi-eps, 101)
phi=np.linspace(0,2*math.pi,101)
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
        """
         # qui con np.gradient che usa la differenza centrale
        dr=(r[1]-r[0])
        dtheta = (theta[1]-theta[0])
        dphi = (phi[1]-phi[0])
        dY_r = np.gradient(Y_griglia, dr, axis=0)
        dY_phi = np.gradient(Y_griglia, dphi, axis=1)
        dY_theta = np.gradient(Y_griglia, dtheta, axis=2)
       
        """
        #qui con la definizione analitica delle derivate
        dY_phi=1j*m*Y_griglia
        dY_theta= (m*Y_griglia/np.tan(TETA))+ (np.sqrt((l+m+1)*(l-m))*np.exp(-1j*PHI)*Y(m+1,l,PHI,TETA))
        
        dxW=pow(RRR,l-1)*(np.sin(TETA)*np.cos(PHI)*l*Y_griglia + np.cos(TETA)*np.cos(PHI)*dY_theta - np.sin(PHI)*dY_phi/np.sin(TETA))
        dyW=pow(RRR,l-1)*(np.sin(TETA)*np.sin(PHI)*l*Y_griglia + np.cos(TETA)*np.sin(PHI)*dY_theta + np.cos(PHI)*dY_phi/np.sin(TETA))
        dzW=pow(RRR,l-1)*(np.cos(TETA)*l*Y_griglia - np.sin(TETA)*dY_theta)
        
        #in questo momento sto prendendo il massimo delle componenti, dato che c'è un coseno a moltiplicare
        
        u=(-(psi(l,h*RRR)+h*RRR*de_psi(l,h*RRR)/(2*l + 1))*(dxW)/(h**2) + de_psi(l,h*RRR)*(RRR*dxW - W*np.sin(TETA)*np.cos(PHI)*(2*l +1))/(h*(2*l +1)) + psi(l-1, k*RRR)*brapp*dxW - l*brapp*psi(l+1,k*RRR)*k*k*(RRR*RRR*dxW - W*np.sin(TETA)*np.cos(PHI)*(2*l + 1)*RRR)/(l+1))
        v=(-(psi(l,h*RRR)+h*RRR*de_psi(l,h*RRR)/(2*l + 1))*(dyW)/(h**2) + de_psi(l,h*RRR)*(RRR*dyW - W*np.sin(TETA)*np.sin(PHI)*(2*l +1))/(h*(2*l +1)) + psi(l-1, k*RRR)*brapp*dyW - l*brapp*psi(l+1,k*RRR)*k*k*(RRR*RRR*dyW - W*np.sin(TETA)*np.sin(PHI)*(2*l + 1)*RRR)/(l+1))
        w=(-(psi(l,h*RRR)+h*RRR*de_psi(l,h*RRR)/(2*l + 1))*(dzW)/(h**2) + de_psi(l,h*RRR)*(RRR*dzW - W*np.cos(TETA)*(2*l +1))/(h*(2*l +1)) + psi(l-1, k*RRR)*brapp*dzW - l*brapp*psi(l+1,k*RRR)*k*k*(RRR*RRR*dzW - W*np.cos(TETA)*(2*l + 1)*RRR)/(l+1))
        

    inte_r=integrate.simpson(RRR*RRR*np.sin(TETA)*(abs(u)**2 + abs(v)**2 + abs(w)**2),r,axis=0)
    inte_te=integrate.simpson(inte_r,theta,axis=1)
    inte_fi=integrate.simpson(inte_te,phi)
    
    A=np.sqrt(1/(rho*inte_fi))
    #A=np.sqrt(hbar/(freq*2*rho*inte_fi))
    return np.array([u*A,v*A, w*A])

#vari vettori spostamento per vari valori di r teta  phi,relativi ad un modo specifico --->sostanzialmente ho calcolato il campo di spostamento cartesiano su una griglia sferica
#è un array  a 3 elementi, ognuno è un array 3D

if args.spostamento == True or args.animazione== True:
    m=int(input('inserisci la terza componente del modo angolare'))
    s_s=np.real(sss(m))
    u_xz, v_xz, w_xz = s_s[0][:, 0,:], s_s[1][:, 0,:],s_s[2][:, 0,:]
    um_xz, vm_xz, wm_xz = s_s[0][:, 50,:], s_s[1][:,50,:],s_s[2][:, 50,:]
    RR,RRm=RRR[:,0,:],RRR[:,50,:]
    TT,TTm=TETA[:,0,:],TETA[:,50,:]
    XX=RR*np.sin(TT)
    ZZ=RR*np.cos(TT)
    
    step=10
    ur_xz,wr_xz, urm_xz,wrm_xz=u_xz[::step,::step],w_xz[::step,::step], um_xz[::step,::step],wm_xz[::step,::step]
    XXr,ZZr= XX[::step,::step], ZZ[::step,::step]
    norm=np.sqrt(ur_xz**2 + wr_xz**2)
    
    fig, ax = plt.subplots()
    ax.scatter(XXr, ZZr, color='red', s=2)
    ax.scatter(-XXr, ZZr, color='red', s=2)
    #ax.set_xlim([-R, R])
    #ax.set_ylim([-R, R])
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    
if args.spostamento== True:
    Q=ax.quiver(XXr,ZZr,ur_xz,wr_xz, norm, cmap="turbo",pivot="tail") #NON MOLTIPLICO PER IL COS(FREQ) CHE AVREI DAVANTI PERCHÈ COSI MOSTRO LO SPOSTAMENTO MAX
    Qm=ax.quiver(-XXr,ZZr,urm_xz,wrm_xz, norm, cmap="turbo", pivot="tail")
    cbar = fig.colorbar(Q, ax=ax)
    cbar.set_label('Norma del campo di spostamento')
    ax.set_title(f"Vettore spostamento modo sferoidale (n,l,m)=({n+1},{l},{m})") 
    plt.show()
    
if args.animazione==True:
    Q=ax.quiver(XXr,ZZr,ur_xz,wr_xz,color="navy",pivot="tail") #NON MOSTRO I COLORI PERCHÈ NON HA SENSO DATO CHE Ò' "INTENSITÀ" VARIA COL COSENO
    Qm=ax.quiver(-XXr,ZZr,urm_xz,wrm_xz,color="navy", pivot="tail")
    def func(frame):
        dt = 0.1 / freq
        t = frame * dt
        #ggestisco questo parametro nell'animaz. per avere un'oscillazione più lenta  (x avere visione qualitativa pulita) (dato che la freq è molto elecata
        c=np.cos(freq*t)
        Q.set_UVC(c*ur_xz, c*wr_xz)
        Qm.set_UVC(c*urm_xz, c*wrm_xz)
        return Q
    
    anim=FuncAnimation(fig,func,repeat=True,frames=200, interval=30)  #interval mi dice il tempo tra un frame e l'altro 
   
    plt.show()

    
if args.intensità == True:
    m=0
    #senza la somma sugli m, avendo la direzione privilegiata devo usare m=0 (simmetria cilindrica)
    Cps=np.empty(0)
    qqs=np.linspace(0,5*10**7,200)
    s_s=sss(m)*np.cos(freq)
    def Cp(q):
        """
        #versione "semplice"
        igr=np.exp(-1j*(q*RRR*np.cos(TETA)))*RRR*RRR*s_s[2]*np.sin(TETA)
        I_r=integrate.simpson(igr,r,axis=0)   #ricorda che per l'indexing ho che 0 è r, 1 phi, 2 teta
        I_te=integrate.simpson(I_r,theta,axis=1)           #integrando riduco le variabili quindi ora phi è 0 e teta è 1
        #I_fi=integrate.simpson(I_te,phi,axis=0)
        I_fi = I_te[0] * 2 * math.pi
        # Prendo il primo valore tanto sono tutti uguali
        
        return pow(q*abs(I_fi),2)
        """
        #versione analitica
        integranda_g=s_s[2]*np.conj(Y(m,l,PHI,TETA))*np.sin(TETA)
        inte1=integrate.simpson(integranda_g,theta, axis=2)
        g_l0 = integrate.simpson(inte1, phi, axis=1)
        integranda=(r**2)*Jv(l,q*r)*g_l0 #lasciando l'esponenziale complesso avrei avuto moltissime oscillazioni--> difficile da calcolare
        ingr= integrate.simpson(integranda, r, axis=0)
        return 4*math.pi*q*q*(2*l +1)*(abs(ingr))**2
        
        
        
    for i in qqs:
        Cps=np.append(Cps,Cp(i))
    In=Cps/(freq**2)
    plt.plot(qqs*R,In, color="navy")
    plt.show()

    #------------< buono per i primi 4 l, l=5 l=7  schizza, prima frequenza è troppo bassa--> perchè??? qualcosa nell'eq delle frequenze forse  /// l=1 molto basso ok, non è zero per imprecisioni computazionali
#forse intensità un po' sballate 



















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
