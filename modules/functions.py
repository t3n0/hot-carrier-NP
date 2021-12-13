import time
import numpy as np
import scipy.special as spy
from scipy import interpolate

def sph_hn1(n, z):
    # much more precise than h = j + i*y, but still not accurate enough when computing h/h'
    return spy.hankel1(n+0.5,z)*np.sqrt(np.pi/(2.*z))

def sph_hnp1(n,z):
    return n/z*sph_hn1(n,z)-sph_hn1(n+1,z)

def exphnp1(n,z):
    # derivatives of exponentially scaled hankel functions
    return (n/z-1j)*spy.hankel1e(n,z) - spy.hankel1e(n+1,z)

def hl_hlp1(n,z):
    # success! ratio between exponentially scaled hankel functions!
    return spy.hankel1e(n+0.5,z) / ((1j-0.5/z)*spy.hankel1e(n+0.5,z) + exphnp1(n+0.5,z))

def rgrid_split(rgrid,R0):
    r1 = rgrid[rgrid<R0]
    r2 = rgrid[rgrid>=R0]
    return r1,r2

def first_item(item):
    return item[0]

def gauss(x,m,s):
    r = 1./(s*np.sqrt(2.*np.pi))*np.exp(-(x-m)**2/2./s**2)
    return r

def cauchy(x,m,s):
    r = s/np.pi/((x-m)**2+s**2)
    return r

def short_spec(en,threshold,mode='less'):
    if   mode == 'less':  b = [x for x in en if x > threshold]
    elif mode == 'great': b = [x for x in en if x < threshold]
    return np.array(b) 

def fermi_position(sp,ns):
    occ = 0
    for i,s in enumerate(sp):
        e,l = s
        occ += 2*l + 1
        if occ>=ns: break
    return i, occ

def slice_speclist(speclist,Ef,phot,maxl):
    occlist = [(e,l) for e,l in speclist if e <= Ef and l<=maxl]
    unocclist = [(e,l) for e,l in speclist if Ef < e <= max(Ef + phot,0.0) and l<=maxl]
    return occlist, unocclist

def slice_specdick(specdick,Ef,phot,maxl):
    occdick = {}
    unoccdick = {}
    for k in specdick.keys():
        occdick[k] = [(e,l) for e,l in specdick[k] if e <= Ef and l<=maxl]
        unoccdick[k] = [(e,l) for e,l in specdick[k] if Ef < e <= max(Ef + phot,0.0) and l<=maxl]
    return occdick, unoccdick

def cutspec(Ef,sp):
    cut = 0
    for value in sp:
        if value[0]<=Ef:
            cut += 1
    return cut

def segment(length,ncores):
    r = []
    base = length / ncores
    for i in range(ncores):
        r.append(base)
    rest = length % ncores
    for i in range(rest):
        r[i]+=1    
    return r

def add_degeneracy(spec):
    new_spec = []
    for e,l in spec:
        for m in range(-l,l+1):
            new_spec.append((e,l,m))
    return new_spec

def degeneracy(sp):
    i = 0
    spec = []
    while i < len(sp):
        l = sp[i][1]
        spec.append((sp[i][0],sp[i][1]))
        i += 2*l + 1
    return spec

def degeneracy_count(sp):
    length = len(sp)
    spec = []
    count = 1
    e_old, l_old, m = sp[0]
    for i in range(1,length):
        e,l,m = sp[i]
        if l == l_old and e == e_old:
            count += 1
        else:
            spec.append((e_old,count))
            l_old = l
            e_old = e
            count = 1
    spec.append((e_old,count))
    return spec

def maxl(sp):
    lmax = 0
    for e,l,m in sp:
        if l>lmax: lmax = l
    return lmax

def maxlef(sp,Ef):
    #sp_ef = [(e,l,m) for e,l,m in sp if e <= Ef]
    #return maxl(sp_ef)
    lmax = 0
    for e,l in sp:
        if e <= Ef and l>lmax: lmax = l
    return lmax

def kdelta(a,b):
    if a == b:
        return 1
    else:
        return 0

def cmplx2reim(ciao):
    if isinstance(ciao,list):
        r = []
        i = []
        for x in ciao:
            r.append(re(x))
            i.append(im(x))
    else:
        r = re(ciao)
        i = im(ciao)
    return r, i

def occ_states(den,R0):
    '''Occupied states from conduction electron density (nm^-3) and radius of NP (nm)
    '''
    vol = 4./3.*np.pi*R0**3/2.
    return int(den*vol)

def occ_states_old(val,atom_mass,density,R0):
    '''valence electrons, atomic mass (g/mol), density (kg/m^3), radius (bohr)
    '''
    const = 0.00018689723 #this constant is avogadro, m->bohr, kg->g, 4/3 pi, 1/2 occupation
    occ = const*val*density*R0**3/atom_mass
    return int(occ)

def spec_ha2ev(spec,zero):
    evspec = []
    for i in range(len(spec)):
        try:
            evspec.append(((spec[i][0]-zero)*27.211396132,spec[i][1],spec[i][2]))
        except:
            evspec.append(((spec[i][0]-zero)*27.211396132,spec[i][1]))
    return evspec

class progress(object):

    def __init__(self):
        print
        print
        self.old_wall_time = time.time()
        self.elaps_wall_time = 0
        self.average = 0

    def elapsed(self,text,count,final=None):
        new_wall_time = time.time()
        CURSOR_UP_ONE = '\x1b[1A'
        ERASE_LINE = '\x1b[2K'
        print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
        print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
        delta = new_wall_time - self.old_wall_time
        self.elaps_wall_time += delta
        self.average = self.elaps_wall_time/float(count)
        if final == None:
            timing = 'Loop number %s. Elapsed %.4f s.' % (count,self.elaps_wall_time)
        else:
            remain = (final-count)*self.average
            timing = 'Loop number %s out of %s. Elapsed %.4f s, remaining %.4f s.' % (count,final,self.elaps_wall_time,remain)
        print text
        print timing
        self.old_wall_time = new_wall_time

def chi3(rs):
    a3 = (4./9./np.pi)**(1./3.)
    c = np.sqrt(a3*rs/4.0/np.pi)*np.arctan(np.sqrt(np.pi/a3/rs))+1./2./(1.+np.pi/a3/rs)
    return c

def elel_scat(diff,density):
    Ef = 0.5*(3.*np.pi**2*density)**(2./3)
    Es = 2./np.pi*np.sqrt(2.*Ef)
    r = diff**2/4./np.pi/Es**(3./2.)/np.sqrt(Ef)*(2.*np.sqrt(Ef*Es)/(4.*Ef+Es)+np.arctan(np.sqrt(4.*Ef/Es))) # Ha
    return r
    #return 0.000001 #to kill antiresonant term

def elel_scat_old(e,Ef,T,con):
    # con is a constant that depends on the fermi level
    # some approx give it as 2*k_f/N^2(0)*chi3(rs), k_f fermi vector, N(0) dos at the fermi level per unit volume
    # chi3 is the function above
    kb = 0.00000316681 # boltzmann constant Hartree / kelvin
    if T == 0.:
        tauinv = (e-Ef)**2*con
    else:
        tauinv = ((e-Ef)**2+(np.pi*kb*T)**2)/(1.0+np.exp(-1.0/kb/T*(e-Ef)))*con
    return tauinv

def jdos(spec,ns,brd,en_grid):
    occ = spec[:ns]
    unocc = spec[ns:]
    occ_deg = degeneracy_count(occ)
    unocc_deg = degeneracy_count(unocc)
    res = []
    for i,en in enumerate(en_grid):
        y = 0.0
        print i
        for ini in occ_deg:
            for fin in unocc_deg:
                if en-fin[0]+ini[0] < -3.0*brd:
                    break
                elif en-fin[0]+ini[0] > 3.0*brd:
                    continue
                y += ini[1]*fin[1]*gauss(en-fin[0]+ini[0],0.0,brd)
        res.append(y)
    return res

def reduced_jdos(spec,ns,brd,en_grid):
    pass

    
