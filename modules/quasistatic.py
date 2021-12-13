from functions import *
import numpy as np
import time

# some constant
bohr2nm = 0.0529177211 # 1 bohr = 0.0529177211 nm
ha2ev = 27.211396132   # 1 Ha = 27.211396132 eV
hbar_Haps = 0.000024188678 # hartree picoseconds, i.e. 1 ps = 0.000024188678 Ha
hbar_eVps = 0.00065826367  # eV picoseconds
light_nmps = 299792.458 # nm/ps
epsilon0_CVnm = 8.854187817*1e-21 # vacuum permittivity C/(V*nm)
ha2j = 4.35974417*1e-18 # 1 Ha = 4.35974417*1e-18 J
hatime2s = 2.418884326505*1e-17 # 1 hbar/Ha = 2.418884326505*1e-17 s

def oscillator(op,fi,oi,gi,o):
    #if not ismp([op,fi,oi,gi,o],mpf): raise Exception('Input must be of type mpf.')
    return (fi*op**2)/(oi**2-o**2-1j*o*gi)

def eps_drude(eb,gd,op,o):
    if isinstance(o,list):
        eps = []
        for j in range(len(o)):
            r = eb + oscillator(op,1.0,0.0,gd,o[j])
            eps.append(r)
    else:
        eps = eb + oscillator(op,1.0,0.0,gd,o)
    return [o, eps]

def eps_drudelorentz(eb,op,fi,oi,gi,o):
    if len(fi) == len(oi) == len(gi):
        if isinstance(o,list):
            eps = []
            for j in range(len(o)):
                r = eb
                for i in range(len(fi)):
                    r += oscillator(op,fi[i],oi[i],gi[i],o[j])
                eps.append(r)
        else:
            eps = eb
            for i in range(len(fi)):
                eps += oscillator(op,fi[i],oi[i],gi[i],o)
    return [o, eps]

def absorption(freq,eps,epsm,R0,ni):
    '''Returns the quasistatic absorption cross (nm^2) section of a nanoparticle immersed in a medium with
    > frequency grid freq (eV)
    > radius R0 (nm),
    > material dielectric function eps,
    > medium dielectric function epsm=[freq_grid_eV/hbar,epsm],
    > medium refractive index ni
    '''
    ab = R0**3*np.pi*4.0*freq*ni*np.imag((eps-epsm)/(eps+2.0*epsm))/light_nmps/hbar_eVps
    #ab = []
    #for i,o in enumerate(freq):
    #    k = o*ni[i]/light_nmps/hbar_eVps
    #    aux = R0**3*np.pi*4.0*k*np.imag((eps[i]-epsm[i])/(eps[i]+2.0*epsm[i]))
    #    ab.append(aux)
    return ab

def plasmon_radpot(rgrid,eps,epsm,R0,field):
    p = []
    for r in rgrid:
        if r < R0:
            aux = -(3*epsm)/(eps+2*epsm)*r
            p.append(field*aux)
        else:
            aux = -r + (eps-epsm)/(eps+2*epsm)*R0**3/r**2
            p.append(field*aux)
    return np.array(p)

def hcpower(elec,hole,Ef):
    '''Return the hot carrier power in watt for given elec, hole distributions in atomic units
    '''
    epow = 0.0
    hpow = 0.0
    for en,_,prob in elec:
        epow += (en - Ef)*prob
    for en,_,prob in hole:
        hpow += (en - Ef)*prob
    totpow = (epow - hpow)*ha2j/hatime2s
    return totpow

def hcnumber(carrier):
    '''Return the carrier number for given distribution in atomic units
    '''
    num = 0.0
    for _,_,prob in carrier:
        num += prob
    return num

def hcfom(carrier,thre,sign):
    '''Return the figure of merit: number of carriers above (or below) the threshold.
    sign = 1 for electrons (above)
    sign = -1 for holes (below)
    '''
    num = 0.0
    for en,_,prob in carrier:
        if sign > 0.0:
            if en > thre:
                num += prob
        elif sign < 0.0:
            if en < thre:
                num += prob
    return num

def plasma_freq(a):
    omega, absor = a
    maximum = 0.0
    for i,value in enumerate(absor):
        if value > maximum:
            maximum = value
            index = i
    return omega[index]
