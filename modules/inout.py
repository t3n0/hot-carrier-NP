import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator, FormatStrFormatter
from scipy.interpolate import interp1d
import numpy as np
from textwrap import dedent
import labellines as lablin
import wl2rgb

inch2point = 72.

def inputfile():
    inp = {}
    inp['material']    = 'cacca'                            # material
    inp['R0']          = [1.0,2.0,3.0]                      # list of NP radii (nm)
    inp['R1']          = 40.0                               # big radius (nm)
    inp['V0']          = -10.0                              # initial guess potential depth (eV)
    inp['density']     = 10.0                               # electron density (nm^-3)
    inp['deltapot']    = 0.01                               # delta potential for the work spectrum (eV)
    inp['Ef']          = -4.5                               # fermi level (eV), i.e. the workfunction
    inp['deltaef']     = 0.01                               # fermi level threshold delta (eV)
    inp['deltar']      = 0.1                                # radial grid delta (nm), should be less than min(pi/k1) = min(pi/sqrt(2(E-V0))) = pi/sqrt(-2V0)
    inp['esamples']    = 1000                               # samples energy grid step
    inp['medium_path'] = '/path/to/dielectric/medium.txt'   # path to medium dielectric
    inp['medium']      = 'water'                            # medium
    inp['elph']        = 0.01                               # elph = hbar/tau, electron phonon scattering (eV)
    inp['brd']         = 0.01                               # broadening of the final energy distribution (eV)
    inp['photon']      = [[2.0,4.0,10],[4.0,6.0,40]]        # list of lists of linspace parameters for the photon energy grid
    inp['phot_mult']   = 2.0                                # highest energy for the vacuum states calculation (eV)
    inp['efield']      = 1e-4                               # electric field strength (V/nm)
    inp['dielectric']  = '/path/to/dielectric/material.txt' # path to the material dielectric function
    inp['flag']        = 'ciaone'                           # folder name for the calculation
    return inp

def header1(var):
    cacca = '''
    ---------------------------------------------------------------------
    Plasmon-induced hot-carriers in %(material)s NPs.
    Radius list (nm): %(R0)s.
    Big radius (vacuum): %(R1)s nm.
    Electron density and fermi level: %(density)s nm^-3, %(Ef)s eV.
    Photon energies (eV) min, max, samples: %(photon)s.
    %(medium)s dielectric function: %(medium_path)s.
    Material dielectric function: %(dielectric)s.
    ---------------------------------------------------------------------
    ''' % var
    return dedent(cacca)

def diel_header(mat,path):
    cacca = '''
    %s dielectric function splined from %s.
    First row > list of frequencies (eV)
    Second row > epsilon
    ''' % (mat, path)
    return dedent(cacca)

def absor_header(mat,R0,med):
    cacca = '''
    %s, %.3f nm radius, absorption cross section (nm^2).
    Surrounding medium: %s.
    Data = [freq_grid_eV, absorption]
    ''' % (mat,R0,med)
    return dedent(cacca)

def spec_header(mat, R0, R1, V0, ns, Ef):
    cacca = '''
    %s spectrum. Radius: %.3f nm. Vacuum radius: %.3f nm. Potential: %.3f eV. Occupied states: %i.
    Fermi level: %s eV.
    The spectrum is degenerate in m.
    Data = [(e1,l),(e2,l),...]
    ''' % (mat, R0, R1, V0, ns, Ef)
    return dedent(cacca)

def dos_header(mat, R0, V0, ns, brd, Ef):
    cacca = '''
    %s density of states. Radius: %.3f nm. Potential: %.3f eV. Occupied states: %i.
    Broadening: %.3f eV. Fermi level: %.3f eV.
    Data = [energy, dos]
    ''' % (mat, R0, V0, ns, brd, Ef)
    return dedent(cacca)

def raw_dist_header(mat,car,R0,R1,medium,phot,Ef,efield):
    cacca = '''
    %s hot %s raw distribution.
    R0 = %s nm. R1 = %s nm. Medium = %s. Photon energy = %s Ha. Fermi level = %s Ha. Efield = %s V/nm.
    List of probabilities of reaching a state with energy ef_i and angular momentum l_i.
    Energies in Ha.
    %s = [(ef1,l1,prob1), (ef2,l2,prob2), ...]
    ''' % (mat,car,R0,R1,medium,phot,Ef,efield,car)
    return dedent(cacca)

def dist_header(mat,R0,R1,med,phot,elph,brd,Ef,efield,pow_ratio):
    cacca = '''
    %s hot carrier distribution. hc_power / qs_power = %s
    R0 = %s nm. R1 = %s nm. Medium = %s. Photon energy = %s eV. El-Ph scattering = %s eV.
    Brd = %s eV. Fermi level = %s eV. Efield = %s V/nm.
    Data = [energy_grid, elec_dist, energy_grid, hole_dist]
    ''' % (mat,pow_ratio,R0,R1,med,phot,elph,brd,Ef,efield)
    return dedent(cacca)

def carrier_header(mat,car,R0,R1,med,elph,brd,Ef,efield,highen):
    cacca = '''
    %s hot %s number vs photon energy.
    R0 = %s nm. R1 = %s nm. Medium = %s. El-Ph scattering = %s eV.
    Brd = %s eV. Fermi level = %s eV. Efield = %s V/nm.
    Highest empty state energy: %s eV.
    Data = [energy grid, hot %s number]
    ''' % (mat,car,R0,R1,med,elph,brd,Ef,efield,highen,car)
    return dedent(cacca)

def power_header(mat,R0,R1,med,elph,Ef,efield,intensity):
    cacca = '''
    %s QS power, HC power and HC/QS ratio vs photon energy.
    R0 = %s nm. R1 = %s nm. Medium = %s. El-Ph scattering = %s eV.
    Fermi level = %s eV. Efield = %s V/nm. Intensity = %s W/nm^2.
    Data = [energy grid, QS power, HC power, HC/QS]
    ''' % (mat,R0,R1,med,elph,Ef,efield,intensity)
    return dedent(cacca)

def load_dielectric(path):
    dielectric_data = np.loadtxt(path)
    freq_eV = []
    nindex = []
    kindex = []
    for d in dielectric_data:
        freq_eV.append(d[0])
        nindex.append(d[1])
        kindex.append(d[2])
    freq_eV = np.array(freq_eV)
    nindex = np.array(nindex)
    kindex = np.array(kindex)
    eps = nindex**2 - kindex**2 + 2j*nindex*kindex
    nspline = interp1d(freq_eV,nindex)
    kspline = interp1d(freq_eV,kindex)
    espline = interp1d(freq_eV,eps)
    return freq_eV, espline, nspline, kspline

def plot_specs(specs,colors=None):
    try:
        i = 0
        for spec,color in zip(specs,colors):
            plot_spec(spec,color,len(specs),i)
            i += 1
    except:
        plot_spec(specs)
    plt.show()
    #plt.savefig(path)
    plt.clf()

def plot_spec(spec,color,lensp,c):
    '''This function plots the jellium spectrum in the form of energy vs angular momentum.
    spec must be a list of pairs [e,l]. eunit must be a string specifying the unit of energy used.
    Default is Hartree.'''
    y = []
    xmin = []
    xmax = []
    i = 0
    while i<len(spec):
        l = spec[i][1]
        step = float(c)/float(lensp)
        xmin.append(spec[i][1]+step)
        xmax.append(spec[i][1]+step+1./float(lensp))
        y.append(spec[i][0])
        i +=1
    plt.hlines(y,xmin,xmax,color=color)
    plt.xlabel('Angular momentum')
    plt.ylabel('Energy (eV)')
#    if path == None:
#        plt.show()
#    else:
#        plt.savefig(path)
#        plt.clf()

def float2dec(num):
    e = int(-np.log10(num))
    if e == 0:
        r = ''
    elif e == 1:
        r = '10'
    else:
        r = '10$^{%s}$'%str(e)
    return r

def zoom(d,rt,lt):
    maxd = max(max(d[1]),max(d[3]))
    d0 = d[0][d[0]>rt]
    d1 = d[1][d[0]>rt]
    d2 = d[2][d[2]<lt]
    d3 = d[3][d[0]<lt]
    fac = int(min(maxd/max(d1),maxd/max(d3)))
    d1 = fac*d1
    d3 = fac*d3
    return (d0,d1,d2,d3), fac

def dist_plotter_log(data,material=None,photon=None,radii=None,Ef=None,xlim=None,ylim=None,filename=None,**kwargs):
    plt.rcParams['font.family'] = 'Arial'
    nplts = len(data)
    fig, axs = plt.subplots(nplts,1,sharex=True,sharey=True,figsize=(3.24,2.31),squeeze=True)

    left = 0.05
    right = 0.9
    bottom = 0.11
    top = 0.95

    ecolor = '#ef8a62'
    hcolor = '#67a9cf'

    for d,ax,i in zip(data,axs,range(nplts)):
        # quanto odio matplotlib! satana!
        ax.semilogy(d[0][d[0]>Ef],d[1][d[0]>Ef],color=ecolor,lw=0.5)
        ax.semilogy(d[2][d[2]<Ef],d[3][d[2]<Ef],color=hcolor,lw=0.5)
        ax.yaxis.tick_right()
        ax.minorticks_off()
        ax.yaxis.set_major_locator(LogLocator(numticks=5))
#        ax.xaxis.set_major_locator(MaxNLocator())
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        yticks = ax.yaxis.get_major_ticks()
        #if i != nplts-1: yticks[1].label2.set_visible(False)
        if i != 0: yticks[-3].label2.set_visible(False)
        ax.set_ylim(ylim)
        ax.tick_params(axis='both', which='both', direction='out',pad=2, labelsize=6)
        if not Ef == None: ax.axvline(Ef,0.,1.,ls='--',lw=0.5,c='k')


    #if not ylim == None: plt.ylim(ylim)
    if not xlim == None: plt.xlim(xlim)

    fig.text(0.5*(left+right),0,'Carrier energy (eV)',ha='center',va='bottom',fontsize=8)
    fig.text(0,0.5*(top+bottom),'Hot carrier distribution (eV$^{-1}$ps$^{-1}$nm$^{-3}$)',ha='left',va='center',rotation='vertical',fontsize=8)
    adjust = {'hspace':0,'left':left,'right' : right,'bottom' : bottom,'top' : top}
    fig.subplots_adjust(**adjust)

    step = (top-bottom)/nplts
    if not radii == None:
        for i,r in enumerate(radii):
            fig.text(0.1,top-i*step-step/3.,'%d nm'%r,ha='left', va='center',fontsize=6)

    if not (material == None or photon == None):
        fig.text(0.5*(left+right),1,material.capitalize()+', $\hbar \omega$ = %.2f eV' % photon,ha='center',va='top',fontsize=8)

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename,dpi=450)
        plt.clf()

def dist_plotter(data,material=None,photon=None,mult=1.,radii=None,Ef=None,xlim=None,ylim=None,filename=None,**kwargs):
    plt.rcParams['font.family'] = 'Arial'

    left = 0.09
    right = 0.98
    bottom = 0.09
    top = 0.95

    nplts = len(data)
    step = (top-bottom)/nplts
    fig, axs = plt.subplots(nplts,1,sharex=True,sharey=True,figsize=(3.24,2.31))
    multstr = float2dec(mult)
    ecolor = '#ef8a62'
    hcolor = '#67a9cf'
    iran = range(nplts)
    for i,d,ax in zip(iran,data,axs):
        d = (d[0],mult*d[1],d[2],mult*d[3])
        ax.plot(d[0][d[0]>Ef],d[1][d[0]>Ef],color=ecolor,linewidth= 1.0)
        ax.plot(d[2][d[0]<Ef],d[3][d[0]<Ef],color=hcolor,linewidth= 1.0)
        if not Ef == None: ax.axvline(Ef,0.,1.,ls='--',c='k',lw=1)
        
        ax.set_ylim(ylim[0],ylim[1]*mult)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.tick_params(axis='both', which='both', direction='in',pad=2, labelsize=6)

    if not xlim == None: plt.xlim(xlim)

    if not radii == None:
        for i,r in enumerate(radii):
            fig.text(right-0.01,top-i*step-step/8.,'%d nm'%r,ha='right', va='center',fontsize=6)
    fig.text(0.5*(left+right),0,'Carrier energy (eV)',ha='center',va='bottom',fontsize=8)
    ylab = r'Hot carrier distribution (%seV$^{-1}$ps$^{-1}$nm$^{-3}$)' % multstr
    fig.text(0, 0.5, ylab, ha='left', va='center',rotation='vertical',fontsize=8)
    #fig.text(left_bor + 0.1,top_bor-0.05,'h$^+$', ha='center', va='center',color = hcolor,fontsize=8)
    #fig.text(right_bor - 0.1,top_bor-0.05,'e$^-$',ha='center', va='center',color = ecolor,fontsize=8)
    if not (material == None or photon == None):
        fig.text(0.5*(left+right),1,material.capitalize()+', $\hbar \omega$ = %.2f eV' % photon,ha='center',va='top',fontsize=8)
    adjust = {'hspace':0,'left':left,'right' : right,'bottom' : bottom,'top' : top}
    fig.subplots_adjust(**adjust)

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename,dpi=450)
        plt.clf()

def tot_carrier_plotter(ab,el,interval,mult=1.,radii=None,material=None,filename=None,figsize=None,xlim=None,**kwargs):
    plt.rcParams['font.family'] = 'Arial'
    fig, ax1 = plt.subplots(figsize=(3.24,2.31))
    left = 0.08
    right = 0.91
    bottom = 0.11
    top = 0.95

    colors = ['#fc8d59','#1a9850','#3288bd']

    ax2 = ax1.twinx()
    iran = range(len(ab))
    step = (top-bottom)/len(ab)
    if not radii == None:
        for i in iran:
            ab[i][1] = ab[i][1]/np.pi/radii[i]**2

    for i in iran:
        el[i][1] = el[i][1]*mult
    multstr = float2dec(mult)

    low, hig = interval
    ratio = max(ab[-1][1][(low<ab[-1][0]) & (ab[-1][0]<hig)])/max(el[-1][1])

    offset1 = offset2 = 0.0
    for i,color in zip(iran,colors):
        #shortab = [a for x,a in zip(ab[i][0],ab[i][1]) if xlim[0] < x < xlim[1]]
        #shortel = [a for x,a in zip(el[i][0],el[i][1]) if xlim[0] < x < xlim[1]]
        ax1.plot(ab[i][0],ab[i][1] + offset1,'--',color=color,lw=1) # - min(shortab))
        ax2.plot(el[i][0],el[i][1] + offset2,color=color,lw=1) # - min(el[i][1]),'--')

        #offset1 += max(shortab)-min(shortab)
        offset1 += max(ab[i][1][(low<ab[-1][0]) & (ab[-1][0]<hig)])
        offset2 = offset1 / ratio
        #ax2.axhline(offset2,lw=0.5,color='k',ls='--')
        #ax2.text(0.5,0.5,'elec')
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=4,prune='upper'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.tick_params(axis='x', which='both', direction='out',pad=2, labelsize=6)
        ax1.tick_params(axis='y', which='both', direction='in',pad=2, labelsize=6)
        ax2.tick_params(axis='both', which='major', direction='in',pad=2, labelsize=6)
    
    ymax1 = 1.05*offset1
    ymax2 = 1.05*offset2
    ax1.set_ylim(0,ymax1)
    ax2.set_ylim(0,ymax2)
    #ax1.axhline(ymax1-0.1*ymax1,0.05,0.15,color='k',ls='--')
    #ax2.axhline(ymax2-0.1*ymax2,0.85,0.95,color='k',ls='-')

    if not radii == None:
        for i,r in enumerate(radii):
            fig.text(right-0.05,bottom+i*step+step/3.,'%d nm'%r,ha='right', va='center',fontsize=8)

    if not material == None:
        fig.text(0.5*(left+right),1,material.capitalize(),ha='center',va='top',fontsize=8)

    y1lab = r'Absorption cross section/$\pi R^2$'
    fig.text(0, 0.5*(bottom+top), y1lab, ha='left', va='center',rotation='vertical',fontsize=8)
    y2lab = r'Hot carrier number (%sps$^{-1}$nm$^{-3}$)' % multstr
    fig.text(1, 0.5*(bottom+top), y2lab, ha='right', va='center',rotation='vertical',fontsize=8)
    fig.text(0.5*(left+right),0,'Photon energy (eV)',ha='center',va='bottom',fontsize=8)

    adjust = {'hspace':0,'left':left,'right' : right,'bottom' : bottom,'top' : top}
    fig.subplots_adjust(**adjust)

    if not xlim == None:
        plt.xlim(xlim)

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename,dpi=450)
        plt.clf()

def scattering_plotter(data,materials,colors,filename=None):
    plt.rcParams['font.family'] = 'Arial'
    fig, ax1 = plt.subplots(figsize=(3.24,2.31))

    for d,mat,color in zip(data,materials,colors):
        ax1.semilogy(d[0],d[2],label=mat,color=color,lw=1)

    ylim = (5.e-1,4.e2)
    ax1.set_ylim(ylim)

    left = 0.09
    right = 0.99
    bottom = 0.09
    top = 0.99

    ax1.tick_params(axis='both', which='both', direction='in',pad=2, labelsize=6)
    fig.text(0.5*(left+right),0,'Carrier energy (eV)',ha='center',va='bottom',fontsize=8)
    fig.text(0,0.5*(top+bottom),'Lifetime (fs)',ha='left',va='center',rotation='vertical',fontsize=8)
    adjust = {'hspace':0,'left':left,'right' : right,'bottom' : bottom,'top' : top}
    fig.subplots_adjust(**adjust)
    plt.legend(loc='best',fontsize=8)

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename,dpi=450)
        plt.clf()

def _scattering_plotter(data,materials,colors,xvals,figsize=(7,5),ylim=None,filename=None):
    plt.rcParams['font.family'] = 'Serif'
    fig, ax1 = plt.subplots(figsize=figsize)

    left, width = 0.005, 0.99
    bottom, height = 0.005, 0.99
    right = left + width
    top = bottom + height

    title_space = 0.01
    small_space = 0.03
    big_space = 0.05

    small_fs = figsize[1]*inch2point*small_space*0.8
    norm_fs = figsize[1]*inch2point*big_space*0.8
    big_fs = figsize[1]*inch2point*title_space*0.8

    top_bor = 1.-title_space
    right_bor = 1.-(1.*small_space)*figsize[1]/figsize[0]
    bot_bor = small_space + big_space
    left_bor = (2.*small_space + big_space)*figsize[1]/figsize[0]

    for d,mat,color in zip(data,materials,colors):
        ax1.semilogy(d[0],d[2],label=mat,color=color,lw=2)

    ax1.tick_params(axis='both', which='major', labelsize=small_fs)
    if not ylim == None:
        ax1.set_ylim(ylim)

    lablin.labelLines(ax1.get_lines(),zorder=2.5,align=True,xvals=xvals,bbox=dict(facecolor='w',edgecolor='none',pad=1),fontsize=norm_fs)

    fig.text(0.5*(left_bor+right_bor),bottom,'Carrier energy (eV)',ha='center',va='bottom',fontsize=norm_fs)
    fig.text(left,0.5*(top_bor+bot_bor),'Lifetime (fs)',ha='left',va='center',rotation='vertical',fontsize=norm_fs)
    adjust = {'hspace':0,'left':left_bor,'right' : right_bor,'bottom' : bot_bor,'top' : top_bor}
    fig.subplots_adjust(**adjust)

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename,dpi=450)
        plt.clf()

def absorption_plotter(data,freq,irradiance,colors,materials,filename=None):
    plt.rcParams['font.family'] = 'Arial'
    fig, ax1 = plt.subplots(figsize=(5.0,3.5))

    left = 0.06
    right = 0.98
    bottom = 0.09
    top = 0.98

    nplts = len(data)
    for i,d,color in zip(range(nplts),data,colors):
        for cac, st in zip(d,['-','--','-.']):
            ax1.plot(cac[0],cac[1]/max(cac[1])+i,color=color,ls=st,lw=1)

    visfreq = [f for f in freq if 1.65 < f < 3.27]
    for i in range(len(freq)-1):
        dx = np.array([freq[i+1],freq[i]])
        dy = np.array([irradiance[i+1],irradiance[i]])
        if freq[i] < 1.65:
            ax1.fill_between(dx,dy*len(data)/max(irradiance),color=(0.5,0.5,0.5,0.2))
        elif freq[i] > 3.27:
            ax1.fill_between(dx,dy*len(data)/max(irradiance),color=(0.5,0.5,0.5,0.2))
        else:
            lam = 1239.84/freq[i]
            c = wl2rgb.wavelength_to_rgb(lam)
            ax1.fill_between(dx,dy*len(data)/max(irradiance),color=c,alpha=0.2)

    for i,mat,color in zip(range(nplts),materials,colors):
        ax1.text(7,i+0.5,mat.capitalize(),ha='center',va='center',color=color,fontsize=8)

    ax1.set_xlim(0,10)
    ax1.set_ylim(0,6)
    ax1.tick_params(axis='both', which='both', direction='in',pad=2, labelsize=6)
    fig.text(0.5*(left+right),0,'Photon energy (eV)',ha='center',va='bottom',fontsize=8)
    fig.text(0,0.5*(top+bottom),'Absorption (a.u.)',ha='left',va='center',rotation='vertical',fontsize=8)
    adjust = {'hspace':0,'left':left,'right' : right,'bottom' : bottom,'top' : top}
    fig.subplots_adjust(**adjust)

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename,dpi=450)
        plt.clf()

def _absorption_plotter(data,freq,irradiance,figsize=(7,5),filename=None,colors=None,materials=None):
    plt.rcParams['font.family'] = 'Serif'
    fig, ax1 = plt.subplots(figsize=figsize)

    left, width = 0.005, 0.99
    bottom, height = 0.005, 0.99
    right = left + width
    top = bottom + height

    title_space = 0.03
    small_space = 0.03
    big_space = 0.05

    small_fs = figsize[1]*inch2point*small_space*0.8
    norm_fs = figsize[1]*inch2point*big_space*0.8
    big_fs = figsize[1]*inch2point*title_space*0.8

    top_bor = 1.-title_space
    right_bor = 1.-(1.*small_space)*figsize[1]/figsize[0]
    bot_bor = 1.5*(small_space + big_space)
    left_bor = (2.*small_space + big_space)*figsize[1]/figsize[0]

    nplts = len(data)
    for i,d,color in zip(range(nplts),data,colors):
        for cac, st in zip(d,['-','--','-.']):
            ax1.plot(cac[0],cac[1]/max(cac[1])+i,color=color,ls=st,lw=1.0)

    visfreq = [f for f in freq if 1.65 < f < 3.27]
    for i in range(len(freq)-1):
        dx = np.array([freq[i+1],freq[i]])
        dy = np.array([irradiance[i+1],irradiance[i]])
        if freq[i] < 1.65:
            ax1.fill_between(dx,dy*len(data)/max(irradiance),color=(0.5,0.5,0.5,0.2))
        elif freq[i] > 3.27:
            ax1.fill_between(dx,dy*len(data)/max(irradiance),color=(0.5,0.5,0.5,0.2))
        else:
            lam = 1239.84/freq[i]
            c = wl2rgb.wavelength_to_rgb(lam)
            ax1.fill_between(dx,dy*len(data)/max(irradiance),color=c,alpha=0.2)

    for i,mat,color in zip(range(nplts),materials,colors):
        ax1.text(7,i+0.5,mat.capitalize(),ha='center',va='center',color=color,fontsize=norm_fs)

    ax1.set_xlim(0,10)
    ax1.set_ylim(0,6.05)
    ax1.tick_params(axis='both', which='major', labelsize=small_fs)
    fig.text(0.5*(left_bor+right_bor),bottom,'Photon energy (eV)',ha='center',va='bottom',fontsize=norm_fs)
    fig.text(left,0.5*(top_bor+bot_bor),'Absorption (a.u.)',ha='left',va='center',rotation='vertical',fontsize=norm_fs)
    adjust = {'hspace':0,'left':left_bor,'right' : right_bor,'bottom' : bot_bor,'top' : top_bor}
    fig.subplots_adjust(**adjust)

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename,dpi=450)
        plt.clf()

def spec2hlines(spec):
    y = []
    xmin = []
    xmax = []
    i = 0
    for i in range(len(spec)):
        l = spec[i][1]
        xmin.append(spec[i][1])
        xmax.append(spec[i][1]+1)
        y.append(spec[i][0])
    return [y,xmin,xmax]

def plot_dist(data,dic={},eunit='Ha',path=None):
    plt.plot(*data,**dic)
    plt.xlabel('Energy (%s)' % eunit)
    plt.ylabel('Hot carrier population')
    if path == None:
        plt.show()
    else:
        plt.savefig(path)
        plt.clf()

def plot_psi(psi,dic=None):
    rep=[]
    imp=[]
    for i in range(len(psi[1])):
        rep.append(re(psi[1][i]))
        imp.append(im(psi[1][i]))
    plt.plot(psi[0],rep,label='Real part')
    plt.plot(psi[0],imp,label='Imag part')
    plt.xlabel('Radial distance (Bohr)')
    plt.ylabel('Wavefunction')
    plt.legend()
    if dic == None:
        plt.show()
    elif isinstance(dic,dict):
        for key in dic:
            getattr(plt,key)(dic[key])
    plt.clf() #clear the current figure, i.e. the next call to plot_psi will NOT draw in the same canvas

def data2file(data,path,info=''):
    fileout = open(path,'w')
    fileout.write(info+'\n')
    for i in range(len(data[0])):
        text = ''
        for d in range(len(data)):
            text += str(data[d][i])+' '
        text += '\n'
        fileout.write(text)

def print_file(info,data,fileout=None):
    if fileout is None:
        name = inspect.stack()[1][3]
        fileout = name+".txt"
    out = open(fileout,'w') # open for writing, deletes any already existing file        
    out.write(info+'\n')
    x,y = data
    for i in range(len(x)):
        if isinstance(y[i],mpf):
            out.write(str(x[i])+" "+str(y[i])+"\n")
        elif isinstance(y[i],mpc):
            out.write(str(x[i])+" "+str(re(y[i]))+" "+str(im(y[i]))+"\n")
        else:
            print 'Print to file error.'

def print_spec(info,spec,fileout=None):
    if fileout is None:
        name = inspect.stack()[1][3]
        fileout = name+".txt"
    out = open(fileout,'w') # open for writing, deletes any already existing file        
    out.write(info+'\n')
    for i in range(len(spec)):
        out.write(str(spec[i][0])+" "+str(spec[i][1])+"\n")

