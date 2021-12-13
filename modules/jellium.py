import functions as fun
import numpy as np
import scipy.special as spy
import scipy.integrate as spint
import inout as io
import time
import matplotlib.pyplot as plt

#np.seterr(all='raise')

def dk1k2(e,l,V0,R0):
    # success! the ration between the spherical hankel and it's derivative
    # is performed without loss of significant digits!
    k1 = np.sqrt(2.0*(e-V0))
    k2 = 1j*np.sqrt(-2.0*e)
    x1 = k1*R0
    x2 = k2*R0
    jl  = spy.spherical_jn(l,x1)
    jlp = spy.spherical_jn(l,x1,True)
    hlhlp = fun.hl_hlp1(l,x2) # where the magic happens
    result = np.imag(hlhlp*jlp/jl-k2/k1)
    return result

def psiR1(e,l,V0,R0,R1):
    # success! I am sufficiently satisfied: R1 must be at least 20*R0 
    # in order to avoid fake states due to some strange effects related to the NP radius R0
    # the exception is not a big deal as long as the sign of psiR1 is tested to be strictly greater than zero
    # and as long that those exceptions appear at an angular momentum greater than the lmax of the NP
    # in that case there are no transition to be considered whatsoever
    k1 = np.sqrt(2.0*(e-V0))
    x1 = k1*R0
    k2 = np.sqrt(2.0*e)
    x2 = k2*R0
    jl1  = spy.spherical_jn(l,x1)
    jlp1 = spy.spherical_jn(l,x1,True)
    jl2  = spy.spherical_jn(l,x2)
    jlp2 = spy.spherical_jn(l,x2,True)
    jl11 = spy.spherical_jn(l+1,x1)
    jl21 = spy.spherical_jn(l+1,x2)
    yl2  = spy.spherical_yn(l,x2)
    ylp2 = spy.spherical_yn(l,x2,True)
    if np.isnan(ylp2):
        beta = 0.0
    else:
        num = (k2*jl1*jlp2 - k1*jl2*jlp1)
        den = (k2*jl1*ylp2 - k1*yl2*jlp1)
        beta = num/den
    psi = spy.spherical_jn(l,k2*R1) - spy.spherical_yn(l,k2*R1)*beta
    return psi

def lspec_unbound(es,l,V0,R0,R1):
    samples = len(es)/4
    emax = es[-1]
    iimax = 1
    es_copy = np.copy(es)
    for ii in range(iimax):
        spec = []
        zero = []
        exit = True
        maxpop = 2
        lastpop = maxpop
        for i in range(len(es_copy)):
            zero.append(psiR1(es_copy[i],l,V0,R0,R1))
        for i in range(len(zero)-1):
            if zero[i]*zero[i+1] < 0.0:
                spec.append(((es_copy[i]+es_copy[i+1])/2.0,l))
        #i = 0
        #diffold = spec[i+1][0] - spec[i][0]
        #i += 1
        #while i<len(spec)-1:
        #    diff = spec[i+1][0] - spec[i][0]
        #    if diff < 0.8*diffold and lastpop >= maxpop:
        #        spec.pop(i+1)
        #        lastpop = 0
        #    elif diff > 1.95*diffold and ii < iimax:
        #        samples *= 2
        #        es_copy = egrid(0.0,emax,samples)
        #        exit = False
        #        break
        #    elif diff > 1.8*diffold and ii == iimax:
        #        i += 1
        #        lastpop += 1
        #    else:
        #        diffold = diff
        #        i += 1
        #        lastpop += 1
        #if exit: break
    return spec

def spec_unbound(es,maxl,V0,R0,R1):
    speclist = []
    specdick = {}
    if es.size == 0: return speclist, specdick
    es_copy = np.copy(es)
    spec_unbound_prog = fun.progress()
    thre = 0.9 # when cutting the spec we keep 10% more states just in case
    if maxl >=0:
        for l in range(maxl+1):
            sp = lspec_unbound(es_copy,l,V0,R0,R1)
            if sp == []: break
            es_copy = fun.short_spec(es_copy,sp[0][0]*thre)
            speclist += sp
            specdick[l] = sp
            spec_unbound_prog.elapsed('Calculating unbound spectrum.',l+1,maxl+1)
    else:
        l = 0
        while True:
            sp = lspec_unbound(es_copy,l,V0,R0,R1)
            if sp == []: break
            es_copy = fun.short_spec(es_copy,sp[0][0]*thre)
            speclist += sp
            specdick[l] = sp
            l += 1
            spec_unbound_prog.elapsed('Calculating unbound spectrum.',l+1)
    speclist.sort(key=fun.first_item)
    return speclist, specdick

def lspec_bound(es,l,V0,R0):
    '''This function calculates the jellium spectrum for the given l, angular momentum, and Es, energy grid.
    It returns the spectrum as a list of pairs [e,l] in ascending order with respect to the energy.
    l must be an integer. Es must be a list of energies in Hartree. If Es is not given, the default built-in Es will be used.'''
    samples = len(es)/4
    emax = es[-1]
    es_copy = np.copy(es)
    for ii in range(10):
        spec = []
        diff = []
        disc = 0
        lensp = 0
        for e in es_copy:
            diff.append(dk1k2(e,l,V0,R0))
        for i in range(len(diff)-2):
            if diff[i]-diff[i+1]>0.0 and diff[i+1]-diff[i+2]<0.0: disc += 1 # looking for the discontinuties
        for i in range(len(diff)-1):
            if diff[i]*diff[i+1]<0.0 and diff[i]-diff[i+1]<0.0:
                lensp += 1
                spec.append(((es_copy[i]+es_copy[i+1])/2.0,l))
        if disc == lensp or disc == lensp-1: break
        samples *= 2
        es_copy = egrid(V0,emax,samples)
    return spec

def spec_bound(es,maxl,V0,R0,prog=True):
    '''This function calculates the full jellium spectrum up to maxl, angular momentum, for the given Es, energy grid.
    It returns the spectrum as a list of pairs [e,l] in ascending order with respect to the energy.
    maxl must be an integer. If maxl is not given, it will try to cover the full spectrum.
    Es must be a list of energies in Hartree.'''
    speclist = [] # list of tuples [(e1,l1),(e2,l2),...]
    specdick = {} # dict of lists of tuples l:[(e1,l1),(e2,l2),...]
    es_copy = np.copy(es)
    thre = 1.1 # when cutting the spec we keep 10% more states just in case
    if prog: spec_bound_prog = fun.progress()
    if maxl >= 0:
        for l in range(0,maxl+1):
            sp = lspec_bound(es_copy,l,V0,R0)
            if sp == []: break
            es_copy = fun.short_spec(es_copy,sp[0][0]*thre)
            speclist += sp
            specdick[l] = sp
            if prog: spec_bound_prog.elapsed('Calculating bound spectrum.',l+1,maxl+1)
    else:
        l = 0
        while True:
            sp = lspec_bound(es_copy,l,V0,R0)
            if sp == []: break
            es_copy = fun.short_spec(es_copy,sp[0][0]*thre)
            speclist += sp
            specdick[l] = sp
            l += 1
            if prog: spec_bound_prog.elapsed('Calculating bound spectrum.',l+1)
    speclist.sort(key=fun.first_item)
    return speclist, specdick

def spectrum(es,maxl,V0,R0,R1):
    es_b  = np.array([x for x in es if x<0.0])
    es_ub = np.array([x for x in es if x>=0.0])
    bspeclist, bspecdick = spec_bound(es_b,maxl,V0,R0)
    uspeclist, uspecdick = spec_unbound(es_ub,maxl,V0,R0,R1)
    speclist = bspeclist + uspeclist
    specdick = {}
    for l in bspecdick.keys():
        try:
            specdick[l] = bspecdick[l] + uspecdick[l]
        except:
            # uspecdick[l] is an empty list, phot_mult too low and/or big radius too small
            specdick[l] = bspecdick[l]
    return speclist, specdick

def wspec_bound(V0,R0,Ef,deltaef,oldns,deltapot,samples):
    '''This function calculates the full spectrum of the nanoparticle. V0 is the initial guess of the potential.
    R0 is the radius of the particle. W is the workfunction, deltaw the precision on W. ns is the number of occupied states.
    It returns the final potential V0.
    '''
    print '--------------------------'
    print 'Calculating correct potential depth.'
    print 'V0 initial guess: %s.\nRadius R0: %s.\nDesired Fermi level Ef: %s.\
            \nDesired occupied states: %s.\n' % (V0,R0,Ef,oldns)
    wspec_prog = fun.progress()
    oldlen = 1
    alpha = 1.
    maxl = -1
    for count in range(1000):
        es = egrid(V0,0.0,samples)
        #es_copy = fun.short_spec(es,Ef+2.*deltaef,'great')
        es_copy = np.copy(es)
        sp, _ = spec_bound(es_copy,maxl,V0,R0,prog=False)
        Efindex, occ = fun.fermi_position(sp,oldns)
        lensp = len(sp)
        if lensp<Efindex:
            text = 'Total states: %s. Wanted: %s. Potential: %s.' % (lensp,Efindex,V0)
            V0 -= deltapot*alpha
        elif lensp>=Efindex and sp[Efindex][0]<Ef-deltaef:
            text = 'Total states: %s. Wanted: %s. Potential: %s. Fermi level: %s. Wanted: %s.'\
                    % (lensp,Efindex,V0,sp[Efindex][0],Ef)
            V0 += (Ef-sp[Efindex][0])*alpha
        elif lensp>=Efindex and sp[Efindex][0]>Ef+deltaef:
            text = 'Total states: %s. Wanted: %s. Potential: %s. Fermi level: %s. Wanted: %s.'\
                    % (lensp,Efindex,V0,sp[Efindex][0],Ef)
            V0 -= -(Ef-sp[Efindex][0])*alpha
        else:
            Ef = sp[Efindex][0]
            #maxl = fun.maxl(sp) + 1
            maxl = fun.maxlef(sp,Ef) + 1
            print '-------------Done-------------'
            text = 'Final potential: %s.\nAdjusted Fermi energy: %s.\
                    \nAdjusted occupied states: %s.\nMax l at Fermi level: %s.'
            print text % (V0,Ef,occ,maxl)
            print '-------------Done-------------'
            return V0, Ef, maxl, occ, Efindex
        if (count+1) % 10 == 0: deltaef *=1.1
        #if abs((lensp-oldlen)/float(oldlen)) > 0.15:
            #deltae /= 2.0
            #deltapot /= 2.0
        #if abs((lensp-oldlen)/float(oldlen)) < 0.05:
        #    deltapot *= 2.0
        #print abs((lensp-oldlen)/float(oldlen)), deltapot, deltae, lensp, V0
        oldlen = lensp
        wspec_prog.elapsed(text,count+1)
    raise Exception('Convergence not reached.')

def egrid(V0,emax,samples):
    t1 = (emax-V0)*0.1
    t2 = (emax-V0)*0.3
    t3 = (emax-V0)*0.6
    deltae = abs(t1)/float(samples)
    es1 = np.linspace(V0+deltae,V0+t1,num=samples,endpoint=False)
    es2 = np.linspace(V0+t1,V0+t2,num=samples,endpoint=False)
    es3 = np.linspace(V0+t2,V0+t3,num=samples,endpoint=False)
    es4 = np.linspace(V0+t3,emax,num=samples,endpoint=False)
    return np.concatenate((es1,es2,es3,es4))

def calculate_dos(spec,brd,en_grid):
    '''This function plots the density of states of the jellium model.
    spec is the spectrum in the form of a list of pairs [e,l]. brd is the energy broadening parameter.
    grid is the number of energy points used to calculate the DOS. 1000 is the default.
    '''
    d = []
    dos_prog = fun.progress()
    for i,e in enumerate(en_grid):
        y = 0.0
        for j in range(len(spec)):
            if abs(e-spec[j][0]) < 4.*brd:
                l = spec[j][1]
                y += (2*l+1)*fun.gauss(e,spec[j][0],brd)
        d.append(y)
        dos_prog.elapsed('Calculating DOS.',i+1,len(en_grid))
    return [en_grid,d]

def psi_bound(e,l,V0,R0,rgrids):
    k1 = np.sqrt(2.0*(e-V0))
    x1 = k1*R0
    k2 = 1j*np.sqrt(-2.0*e)
    x2 = k2*R0
    #psi_rad = np.zeros(len(rgrid),dtype=np.complex)
    hl = fun.sph_hn1(l,x2)
    jl = spy.spherical_jn(l,x1)
    AB = hl/jl
    rgrid, r1, r2 = rgrids
    #for i,r in enumerate(rgrid):
    #    if r < R0:
    #        psi_rad[i] = spy.spherical_jn(l,k1*r)#*AB
    #    else:
    #        psi_rad[i] = fun.sph_hn1(l,k2*r)/AB
    psi_rad1 = spy.spherical_jn(l,k1*r1)
    psi_rad2 = fun.sph_hn1(l,k2*r2)/AB
    psi_rad = np.concatenate((psi_rad1,psi_rad2))
    g = np.conj(psi_rad)*psi_rad*rgrid**2
    g = np.nan_to_num(g)
    A = np.trapz(g,rgrid)
    result = psi_rad/np.sqrt(A)
    return result

def psi_unbound(e,l,V0,R0,rgrids):
    k1 = np.sqrt(2.0*(e-V0))
    x1 = k1*R0
    k2 = np.sqrt(2.0*e)
    x2 = k2*R0
    #psi_rad = np.zeros(len(rgrid),dtype=np.complex)
    jl1  = spy.spherical_jn(l,x1)
    jlp1 = spy.spherical_jn(l,x1,True)
    jl2  = spy.spherical_jn(l,x2)
    jlp2 = spy.spherical_jn(l,x2,True)
    jl11 = spy.spherical_jn(l+1,x1)
    jl21 = spy.spherical_jn(l+1,x2)
    yl2  = spy.spherical_yn(l,x2)
    ylp2 = spy.spherical_yn(l,x2,True)
    if np.isnan(ylp2):
        beta = 0.0
    else:
        num = (k2*jl1*jlp2 - k1*jl2*jlp1)
        den = (k2*jl1*ylp2 - k1*yl2*jlp1)
        beta = num/den
    alpha = beta / (k2*R0**2*(k1*jl2*jl11-k2*jl1*jl21))
    rgrid, r1, r2 = rgrids
    #for i,r in enumerate(rgrid):
    #    if r < R0:
    #        psi_rad[i] = spy.spherical_jn(l,k1*r)*alpha
    #    else:
    #        psi_rad[i] = spy.spherical_jn(l,k2*r) - spy.spherical_yn(l,k2*r)*beta
    psi_rad1 = spy.spherical_jn(l,k1*r1)*alpha
    psi_rad2 = spy.spherical_jn(l,k2*r2) - spy.spherical_yn(l,k2*r2)*beta
    psi_rad = np.concatenate((psi_rad1,psi_rad2))
    g = np.conj(psi_rad)*psi_rad*rgrid**2
    g = np.nan_to_num(g)
    A = np.trapz(g,rgrid)
    result = psi_rad/np.sqrt(A)
    return result

def psir(e,l,V0,R0,rgrids):
    '''Calculates the radial part of the wavefunction.
    '''
    if e>0.0:
        psi = psi_unbound(e,l,V0,R0,rgrids)
    else:
        psi = psi_bound(e,l,V0,R0,rgrids)
    return psi

def angular_element(l):
    ang_square = 0.0
    for m in range(-l,l+1):
        a = (l+1.0-m)*(l+1.0+m)/(2.0*l+1.0)/(2.0*l+3.0)
        ang_square += a
    return ang_square

def _angular_element(l,m,lp,mp):
    a = np.sqrt((l+1.0-m)*(l+1.0+m)/(2.0*l+1.0)/(2.0*l+3.0))*fun.kdelta(m,mp)*fun.kdelta(l+1,lp)
    b = np.sqrt((l-m)*(l+m)/(2.0*l-1.0)/(2.0*l+1.0))*fun.kdelta(m,mp)*fun.kdelta(l-1,lp)
    return a + b

def radial_element(psii,psif,pot,rgrid):
    g = np.conj(psii)*psif*pot*rgrid**2
    g = np.nan_to_num(g)
    A = np.trapz(g,rgrid)
    return np.real(np.conj(A)*A)

def _radial_element(psii,psif,pot,rgrid):
    g = np.conj(psii)*psif*pot*rgrid**2
    g = np.nan_to_num(g)
    A = np.trapz(g,rgrid)
    return A

class newMatrix(object):
    def __init__(self,occlist,unoccdick,R0,V0,pot,rgrid,lmax):
        self.elem = {}
        self.rad = {}
        self.ang = {}
        self.lmax = lmax
        r1, r2 = fun.rgrid_split(rgrid,R0)
        c2 = 0
        matrix_prog = fun.progress()
        for e1,l1 in occlist:
            psif = psir(e1,l1,V0,R0,(rgrid,r1,r2))
            if l1==0:
                unocclist = unoccdick[l1+1]
            elif l1==lmax:
                unocclist = unoccdick[l1-1]
            else:
                unocclist = unoccdick[l1-1] + unoccdick[l1+1]
            for e2,l2 in unocclist:
                psii = psir(e2,l2,V0,R0,(rgrid,r1,r2))
                rad = radial_element(psii,psif,pot,rgrid)
                ang = angular_element(min(l1,l2))
                elem = rad * ang
                #if -0.184 < e1 < -0.183  and l1 == 29 and l2 == 28:
                #    print e1,l1,e2,l2,elem

                #    pollo = np.conj(psif)*psii*pot*rgrid**2
                #    integ_nan = np.trapz(pollo,rgrid)
                #    integ_spint = spint.simps(pollo,rgrid)
                #    rad_nan = np.real(np.conj(integ_nan)*integ_nan)
                #    plt.plot(rgrid,np.real(pollo),rgrid,np.imag(pollo),label='%s %s %s with nan'%(e1,e2,elem))

                #    pollo = np.nan_to_num(pollo)
                #    integ_nonan = np.trapz(pollo,rgrid)
                #    rad_nonan = np.real(np.conj(integ_nonan)*integ_nonan)
                #    plt.plot(rgrid,np.real(pollo),rgrid,np.imag(pollo),label='%s %s %s without nan'%(e1,e2,elem))

                #    print integ_nan, integ_spint, rad_nan, rad_nonan, rad
                #    #plt.xlim(0.0,R0*2.)
                #    plt.legend(loc='upper right')
                #    plt.show()

                #elem = 1.
                #self.rad['%s %s %s %s'%(e1,l1,e2,l2)] = rad
                #self.rad['%s %s %s %s'%(e2,l2,e1,l1)] = rad
                #self.ang['%s %s'%(l1,l2)] = ang
                #self.ang['%s %s'%(l2,l1)] = ang
                #self.elem['%s %s %s %s'%(e1,l1,e2,l2)] = elem
                #self.elem['%s %s %s %s'%(e2,l2,e1,l1)] = elem
                self.elem[(e1,l1,e2,l2)] = elem
                self.elem[(e2,l2,e1,l1)] = elem
            c2 += 1
            matrix_prog.elapsed('Calculating matrix elements.',c2,len(occlist))

    def element(self,tup):
        return self.elem[tup]

def new_hot_carrier(final,initial,zero,Ef,photon,elph,den,matrix):
    carrier = []
    lmax = matrix.lmax
    c2 = 0
    hot_prog = fun.progress()
    for e1,l1 in final:
        t1 = fun.elel_scat(e1-Ef,den)
        aux = 0.0
        if l1==0:
            initlist = initial[l1+1]
        elif l1==lmax:
            initlist = initial[l1-1]
        else:
            initlist = initial[l1-1] + initial[l1+1]
        for e2,l2 in initlist:
            mat = matrix.element((e1,l1,e2,l2))
            t2 = fun.elel_scat(e2-Ef,den)
            res = fun.cauchy(photon,e1-e2,t1+t2+2.*elph) # e1>e2, e1 is the final energy of the electron
            ant = fun.cauchy(photon,e2-e1,t1+t2+2.*elph)
            aux += mat * (res + ant)
        carrier.append((e1-zero,l1,aux*4.0*np.pi))
        c2 += 1
        hot_prog.elapsed('Calculating hot carriers transitions.',c2,len(final))
    return carrier

def energy_dist(carrier,grid,brd):
    pop = []
    dist_prog = fun.progress()
    for c2,e in enumerate(grid):
        aux = 0.0
        for ef,_,prob in carrier:
            if abs(e-ef) < 5.*brd:
                g2 = fun.gauss(e,ef,brd)
                aux += prob*g2
        pop.append(aux)
        dist_prog.elapsed('Calculating carrier distribution.',c2+1,len(grid))
    assert len(grid) == len(pop)
    return np.array(pop)

def _sph_harm(sp,theta,phi):
    # maybe in the future
    e, l, m = sp
    arm = spy.sph_harm(m,l,theta,phi)
    return arm

def _psi_square(sp,V0,R0,x,y,z):
    # maybe in the furture
    r = np.sqrt(x**2+y**2+z**2)
    try:
        theta = np.arccos(z/r)
    except:
        theta = 0.
    phi = np.arctan2(y,x)
    arm = sph_harm(sp,theta,phi)
    rad = psir(sp,V0,R0,r)
    return arm*rad

############################################################################
# old radial psi calculation with delta normalization and other stuff
###########################################################################

def final_list(fin,ini,V0,R0,plasma,sigma,r,pot,zero):
    final_states = []
    c1 = 0
    for ii, ftra in enumerate(fin):
        fst, m = ftra
        ef, l = fst
        psif = psir(fst,V0,R0,r)
        aux = 0.0
        for itra in ini[ii]:
            ist, mp = itra
            ei, lp = ist
            psii = psir(ist,V0,R0,r)
            init = (psii,l,m)
            final = (psif,lp,mp)
            mat = mat_el_mod2(init,final,pot)
            #g1 = fun.gauss(0.0,plasma-ef+ei,sigma)
            g1 = fun.cauchy(0.0,plasma-ef+ei,sigma)
            g2 = fun.cauchy(0.0,plasma+ef-ei,sigma)
            aux += mat*(g1 + g2)
            c1 += 1
        final_states.append((ef-zero,4.0*np.pi*aux)) #fermi's golden rule times 2 for the spin
    return final_states

def hot_carrier(initial,final,zero,plasma,sigma,matrix,matmode='complete',resmode='both'):
    carrier = []
    sign = 1.0
    if final[0][0] < initial[0][0]: sign=-1.0
    maxl = min(fun.maxl(final),fun.maxl(initial))
    notunit = not (matmode == 'unit')
    for fin in final:
        e1,l1,m1 = fin
        aux = 0.0
        if l1>maxl+1 and notunit: continue
        for ini in initial:
            e2,l2,m2 = ini
            if l2>maxl+1 and notunit: continue
            if (abs(l1-l2) != 1 or m1 != m2) and notunit: continue
            res = ant = 0.0
            if resmode == 'res' or resmode == 'both': res = fun.cauchy(plasma,sign*(e1-e2),sigma) # e1>e2, e1 is the final energy of the electron
            if resmode == 'ant' or resmode == 'both': ant = fun.cauchy(plasma,sign*(e2-e1),sigma)
            if matmode == 'complete': mat = matrix.element(fin,ini)
            if matmode == 'angular': mat = matrix.angular(fin,ini)
            if matmode == 'unit': mat = 1.0
            aux += mat*(res+ant)
        carrier.append((e1-zero,aux*4.0*np.pi))
    return carrier

def hot_carrier_autoscat(initial,final,zero,plasma,Ef,den,elph,matrix,matmode='complete',resmode='both'):
    carrier = []
    sign = 1.0
    if final[0][0] < initial[0][0]: sign=-1.0
    maxl = min(fun.maxl(final),fun.maxl(initial)) 
    hot_prog = fun.progress()
    lenfin = len(final)
    for count,fin in enumerate(final):
        e1,l1,m1 = fin
        aux = 0.0
        #if l1>maxl+1: continue 
        for ini in initial:
            e2,l2,m2 = ini
            #if l2>maxl+1: continue 
            if (abs(l1-l2) != 1 or m1 != m2): continue 
            res = ant = 0.0
            t1 = fun.elel_scat(e1-Ef,den)
            t2 = fun.elel_scat(e2-Ef,den)
            res = fun.cauchy(plasma,sign*(e1-e2),t1+t2+2.*elph) # e1>e2, e1 is the final energy of the electron
            ant = fun.cauchy(plasma,sign*(e2-e1),t1+t2+2.*elph)
            mat = matrix.element(fin,ini)
            #if resmode == 'res' or resmode == 'both': res = fun.cauchy(plasma,sign*(e1-e2),t1+t2+2.*elph) 
            # e1>e2, e1 is the final energy of the electron
            #if resmode == 'ant' or resmode == 'both': ant = fun.cauchy(plasma,sign*(e2-e1),t1+t2+2.*elph)
            #if matmode == 'complete': mat = matrix.element(fin,ini)
            #if matmode == 'angular': mat = matrix.angular(fin,ini)
            #if matmode == 'unit': mat = 1.0
            aux += mat*(res+ant)
        if (count+1) % (lenfin/1000+1) == 0: hot_prog.elapsed('Calculating hot carrier generation rates.',count+1,lenfin)
        carrier.append((e1-zero,aux*4.0*np.pi))
    return carrier

def mat_el_mod2(init,final,pot,rgrid):
    psii,l,m = init
    psif,lp,mp = final
    ang = angular_element(l,m,lp,mp)
    rad = radial_element(psii,psif,pot)
    m = np.conj(rad*ang)*rad*ang
    return m

def radmat_elements(sp,ns,R0,V0,rgrid,pot):
    fermi = sp[ns][0]
    degsp = fun.degeneracy(sp)
    cut = fun.cutspec(fermi,degsp)
    degocc = degsp[:cut]
    degunocc = degsp[cut:]
    radmat = {}
    ctot = len(degocc)
    rad_prog = fun.progress()
    for c1,ini in enumerate(degocc):
        e1,l1 = ini
        psii = psir(e1,l1,V0,R0,rgrid)
        for fin in degunocc:
            e2,l2 = fin
            if abs(l1-l2) != 1: continue
            psif = psir(e2,l2,V0,R0,rgrid)
            aux = _radial_element(psii,psif,pot,rgrid)
            rad = float(np.real(np.conj(aux)*aux))
            radmat['%s %s %s %s'%(e1,l1,e2,l2)] = rad
            radmat['%s %s %s %s'%(e2,l2,e1,l1)] = rad
        rad_prog.elapsed('Calculating radial elements',c1+1,ctot)
    print radmat
    return radmat

def angmat_elements(sp,ns):
    angmat = {}
    occ = sp[:ns]
    maxl = fun.maxl(occ)
    ang_prog = fun.progress()
    for l1 in range(maxl+1):
        for m in range(-l1,l1+1):
            ang = _angular_element(l1,m,l1+1,m)**2
            angmat['%s %s %s %s'%(l1,l1+1,m,m)] = ang
            angmat['%s %s %s %s'%(l1+1,l1,m,m)] = ang
        ang_prog.elapsed('Calculating angular elements.',l1+1,maxl+1)
    return angmat

class Matrix(object):
    def __init__(self,sp,ns,R0,V0,rgrid,pot):
        self.radmat = radmat_elements(sp,ns,R0,V0,rgrid,pot)
        self.angmat = angmat_elements(sp,ns)

    def element(self,sp1,sp2):
        e1,l1,m1 = sp1
        e2,l2,m2 = sp2
        radtup = (e1,l1,e2,l2)
        angtup = (l1,l2,m1,m2)
        try:
            r = self.radmat[radtup]*self.angmat[angtup]
        except:
            r = 0.0
        return r

    def angular(self,sp1,sp2):
        e1,l1,m1 = sp1
        e2,l2,m2 = sp2
        angtup = (l1,l2,m1,m2)
        try:
            r = self.angmat[angtup]
        except:
            r = 0.0
        return r

class _prob9(object):
    def __init__(self,prob):
        self.prob = prob # if prob is defined outside, it will change!!

    def increment(self,inc):
        if len(inc) != len(self.prob): raise Exception('prob and increment must have same length.')
        for i in range(len(self.prob)):
            self.prob[i] += inc[i]

    def get(self):
        return self.prob

def _sum_over_m(sp,ns,matrix,plasma,sigma,mode):
    fermi = sp[ns][0]
    degsp = fun.degeneracy(sp)
    cut = fun.cutspec(fermi,degsp)
    degocc = degsp[:cut]
    degunocc = degsp[cut:]
    summ = {}
    for ini in degocc:
        e1,l1 = ini
        for fin in degunocc:
            e2,l2 = fin
            minl=min(l1,l2)
            aux = 0.0
            for m in range(-minl,minl+1):
                aux += matrix.element((e1,l1,m),(e2,l2,m))
            g1 = g2 = 0.0
            if mode == 'res' or mode == 'both': g1 = fun.cauchy(plasma,e2-e1,sigma)
            if mode == 'ant' or mode == 'both': g2 = fun.cauchy(plasma,e1-e2,sigma)
            summ[(ini,fin)] = aux*(g1+g2)
    return summ

class _MatrixSum(object):
    def __init__(self,sp,ns,matrix,plasma,sigma,mode):
        self.summ = sum_over_m(sp,ns,matrix,plasma,sigma,mode)

    def element(self,sp1,sp2):
        try:
            r = self.summ[(sp1,sp2)]
        except:
            r = 0.0
        return r

def _energy_dist_old(carrier,sp,Ef,grid,plasma,sigma,brd,pot,V0,R0,r):
    cut = fun.cutspec(Ef,sp)
    if carrier == 'e':
        sp_i = fun.degeneracy(sp[:cut])
        sp_f = fun.degeneracy(sp[cut:])
        text = 'Calculating electron transitions.'
    elif carrier == 'h':
        sp_f = fun.degeneracy(sp[:cut])
        sp_i = fun.degeneracy(sp[cut:])
        text = 'Calculating hole transitions.'
    else:
        print 'suka'
    transition_prog = fun.progress()
    c1 = 0
    final_states = []
    for fst in sp_f:
        ef, l = fst
        psif = psir(fst,V0,R0,r)
        for m in range(-l,l+1):
            aux = 0.0
            for ist in sp_i:
                ei, lp = ist
                if abs(lp-l) != 1: continue
                if lp < abs(m): continue
                #if abs(plasma-ef+ei) > 3.0*sigma: continue
                psii = psir(ist,V0,R0,r)
                mp = m
                c1 += 1
                #print c1, ei, ef, plasma, abs(plasma-ef+ei)/3.0/sigma, l, lp, m, mp
                init = (psii,l,m)
                final = (psif,lp,mp)
                mat = mat_el_mod2(init,final,pot)
                #g1 = fun.gauss(0.0,plasma-ef+ei,sigma)
                g1 = fun.cauchy(0.0,plasma-ef+ei,sigma)
                g2 = fun.cauchy(0.0,plasma+ef-ei,sigma)
                aux += mat*(g1 + g2)
                transition_prog.elapsed(text,c1)
            final_states.append((ef,aux))
    pop = []
    dist_prog = fun.progress()
    for c2,e in enumerate(grid):
        aux = 0.0
        for ef,prob in final_states:
            g2 = fun.gauss(e,ef,brd)
            aux += prob*g2
        pop.append(4.0*np.pi*aux)
        dist_prog.elapsed('Calculating final energy distribution.',c2+1,len(grid))
    assert len(grid) == len(pop)
    return [grid, pop]

def _energy_dist_parallel(ncpu,carrier,sp,Ef,grid,plasma,sigma,brd,pot,V0,R0,r):
    cut = fun.cutspec(Ef,sp)
    if carrier == 'e':
        sp_i = fun.degeneracy(sp[:cut])
        sp_f = fun.degeneracy(sp[cut:])
        text = 'Calculating electron transitions.'
    elif carrier == 'h':
        sp_f = fun.degeneracy(sp[:cut])
        sp_i = fun.degeneracy(sp[cut:])
        text = 'Calculating hole transitions.'
    else:
        print 'suka'
    c1 = 0
    c2 = 0
    fin = []
    ini = []
    for fst in sp_f:
        ef, l = fst
        for m in range(-l,l+1):
            fin.append((fst,m))
            aux = []
            for ist in sp_i:
                ei, lp = ist
                if abs(lp-l) != 1: continue
                if lp < abs(m): continue
                aux.append((ist,m))
                c2 += 1
            ini.append(aux)
    seg = fun.segment(len(fin),100*ncpu)
    segs = []
    maxs = 0
    mins = 0
    for s in seg:
        maxs += s
        segs.append((fin[mins:maxs],ini[mins:maxs]))
        mins += s
    print 'Starting parallel computation with %s cores.' % ncpu
    start = time.time()
    final_states_list = Parallel(n_jobs=ncpu)(delayed(final_list)(fin,ini,V0,R0,plasma,sigma,r,pot,Ef) for fin, ini in segs)
    print 'Elapsed %s s.' % (time.time() - start)
    final_states = [item for sublist in final_states_list for item in sublist]
    pop = []
    dist_prog = fun.progress()
    for c2,e in enumerate(grid):
        aux = 0.0
        for ef,prob in final_states:
            g2 = fun.gauss(e,ef,brd)
            aux += prob*g2
        pop.append(aux)
        dist_prog.elapsed('Calculating final energy distribution.',c2+1,len(grid))
    assert len(grid) == len(pop)
    return [grid, pop]    

def _psi_boundasy(e,l,V0,R0,x):
    k1 = np.sqrt(2.0*(e-V0))
    x1 = k1*R0
    k2 = 1j*np.sqrt(-2.0*e)
    x2 = k2*R0
    psi_rad = []
    hl = (-1j)**(l+1)*np.exp(1j*x2)/x2 # asymptotic expansion
    jl = spy.spherical_jn(l,x1)
    AB = hl/jl
    for i in range(0,len(x)):
        if x[i]<R0:
            try:
                psi_rad.append(AB*spy.spherical_jn(l,k1*x[i]))
            except:
                #print AB*spy.spherical_jn(l,k1*x[i])
                psi_rad.append(complex(AB*mp.mpc(spy.spherical_jn(l,k1*x[i]))))
        else:
            psi_rad.append((-1j)**(l+1)*np.exp(1j*k2*x[i])/(k2*x[i]))
    g = []
    for i in range(0,len(psi_rad)):
        try:
            prod = np.conj(psi_rad[i])*psi_rad[i]*x[i]**2
        except:
            #print e,l
            prod = complex(mp.conj(mp.mpc(psi_rad[i]))*mp.mpc(psi_rad[i])*mp.mpf(x[i])**2)
        g.append(prod)
    A = np.trapz(g,x)
    for i in range(0,len(x)):
        psi_rad[i] = psi_rad[i]/np.sqrt(A)
    return psi_rad

def _psi_unbound_delta(e,l,V0,R0,xmax,deltax,emax,deltae):
    A = norm_unbound(e,l,V0,R0,xmax=xmax,deltax=deltax,emax=emax,deltae=deltae)
    psi = unbound(e,l,V0,R0,xmax=xmax,deltax=deltax)
    for i in range(0,len(psi[0])):
        psi[1][i] = psi[1][i]/sqrt(A)
    return psi

def _f_e1e2(e1,e2,l,V0,R0,xmax,deltax):
    psi1 = unbound(e1,l,V0,R0,xmax,deltax)
    psi2 = unbound(e2,l,V0,R0,xmax,deltax)
    g = []
    for i in range(len(psi1[1])):
        g.append(conj(psi1[1][i])*psi2[1][i]*psi1[0][i]**2)
    A = np.trapz(g,psi1[0])
    return A

def _norm_unbound(e1,l,V0,R0,xmax,deltax,emax,deltae):
    e2 = arange(mpf('0.0001'),emax,deltae)        
    f = []
    for i in range(0,len(e2)):
        f.append(f_e1e2(e1,e2[i],l,V0,R0,xmax,deltax))
    A = np.trapz(f,e2)
    #B = mpf(1)/sqrt(A)
    return A
