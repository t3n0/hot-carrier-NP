import sys
import os
import json
#from scipy.interpolate import interp1d
import numpy as np

WORK_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(WORK_DIR))

DATA   = os.path.join(WORK_DIR,'data')
IMAGES = os.path.join(WORK_DIR,'images')

import modules.jellium as jel
import modules.inout as io
import modules.functions as fun
import modules.quasistatic as qs

try:
    inp = open(sys.argv[1],'r')
    var = json.loads(inp.read())
    inp.close()
except:
    print 'No input file given, please find an input file skeleton in "input_skeleton.txt".'
    inp = open('input_skeleton.txt','w')
    inp.write(json.dumps(io.inputfile(),indent=4))
    inp.close()
    sys.exit(0)

# unpacking var
material               = var['material']        # material
R0_list                = var['R0']              # list of NP radii
R1_list                = var['R1']              # list of big radii
V0_eV                  = var['V0']              # initial guess potential depth (eV)
density_nm             = var['density']         # electron density (nm^-3)
deltapot_eV            = var['deltapot']        # delta potential for the work spectrum (eV)
Ef_eV                  = var['Ef']              # fermi level (eV), i.e. the workfunction
deltaef_eV             = var['deltaef']         # fermi level threshold delta (eV)
deltar_nm              = var['deltar']          # radial grid delta (nm), should be less than min(pi/2/k1max) = min(pi/2/sqrt(2(Emax-V0)))
samples                = var['esamples']        # samples energy grid step
eps_medium_path        = var['medium_path']     # path to medium dielectric
medium                 = var['medium']          # medium
elph_eV                = var['elph']            # elph = hbar/tau, electron phonon scattering (eV)
brd_eV                 = var['brd']             # broadening of the final energy distribution (eV)
phot_list_list_eV      = var['photon']          # list of lists of linspace parameters for the photon energy grid
phot_mult              = var['phot_mult']       # photon multiplier: max empty state energy = phot_mult * max_phot
efield_nm              = var['efield']          # electric field strength (V/nm)
eps_material_path      = var['dielectric']      # path to the material dielectric function
flag                   = var['flag']            # folder name for the calculation

# intensity of light
intensity_wattnm2 = 0.5*qs.light_nmps*qs.epsilon0_CVnm*efield_nm**2*1e12

# photon energy grid
cacca = []
for lis in phot_list_list_eV:
    cacca.append(np.linspace(lis[0],lis[1],lis[2],endpoint=False))
photons_eV = np.concatenate(cacca)
photons_Ha = photons_eV/qs.ha2ev

# create flag subdirectory
flag_path = os.path.join(DATA,material,flag)
if not os.path.exists(flag_path):
    os.makedirs(flag_path)

print io.header1(var)

# change to atomic units
V0_Ha = V0_eV/qs.ha2ev
Ef_Ha = Ef_eV/qs.ha2ev
deltapot_Ha = deltapot_eV/qs.ha2ev
deltaef_Ha = deltaef_eV/qs.ha2ev
density_bohr = density_nm*qs.bohr2nm**3
brd_Ha = brd_eV/qs.ha2ev
elph_Ha = elph_eV/qs.ha2ev

# load dielectric functions from file
coarse_mat_freq_eV, eps_material, n_material, k_material = io.load_dielectric(eps_material_path)
coarse_med_freq_eV, eps_medium, n_medium, k_medium = io.load_dielectric(eps_medium_path)

# write spline dielectric to file
mat_diel_path = os.path.join(flag_path,'dielectric_material.txt')
med_diel_path = os.path.join(flag_path,'dielectric_medium.txt')
mat_freq_eV = np.linspace(min(coarse_mat_freq_eV),max(coarse_mat_freq_eV),2000)
med_freq_eV = np.linspace(min(coarse_med_freq_eV),max(coarse_med_freq_eV),2000)
np.savetxt(mat_diel_path,(np.real(mat_freq_eV),np.real(eps_material(mat_freq_eV)),np.imag(eps_material(mat_freq_eV))),header=io.diel_header(material,eps_material_path))
np.savetxt(med_diel_path,(np.real(med_freq_eV),np.real(eps_medium(med_freq_eV)),np.imag(eps_medium(med_freq_eV))),header=io.diel_header(medium,eps_medium_path))

# absorption cross section nm^2
fmin = max(min(coarse_mat_freq_eV),min(coarse_med_freq_eV))
fmax = min(max(coarse_mat_freq_eV),max(coarse_med_freq_eV))
freq_eV = np.linspace(fmin,fmax,2000)
eps = eps_material(freq_eV)
epsm = eps_medium(freq_eV)
indm = n_medium(freq_eV)
for R0_nm in R0_list:
    absor = qs.absorption(freq_eV,eps,epsm,R0_nm,indm)
    absor_path = os.path.join(flag_path,'absorption_R0=%.3f_medium=%s.txt' % (R0_nm,medium))
    np.savetxt(absor_path,(freq_eV,absor),header=io.absor_header(material,R0_nm,medium))

# loop over radii
for R0_nm, R1_nm in zip(R0_list,R1_list): # R0, R1 in nanometer

    R1_bohr = R1_nm / qs.bohr2nm         # R1 in bohr
    R0_bohr = R0_nm / qs.bohr2nm         # R0 in bohr
    ns = fun.occ_states(density_nm,R0_nm)

    # calculating spectrum
    V0_Ha, Ef_Ha, maxl, ns, Efindex = jel.wspec_bound(V0_Ha,R0_bohr,Ef_Ha,deltaef_Ha,ns,deltapot_Ha,samples)

    # updating eV units
    V0_eV = V0_Ha*qs.ha2ev
    Ef_eV = Ef_Ha*qs.ha2ev

    max_phot_eV = np.max(photons_eV)
    max_phot_Ha = np.max(photons_Ha)
    highest_state_eV = Ef_eV + phot_mult*max_phot_eV
    highest_state_Ha = Ef_Ha + phot_mult*max_phot_Ha

    # calculate spectrum
    if highest_state_Ha > 0.0:
        es_Ha_unbound = jel.egrid(0.0,highest_state_Ha,samples)
    else:
        es_Ha_unbound = []
        highest_state_eV = 0.0
        highest_state_Ha = 0.0
    es_Ha_bound   = jel.egrid(V0_Ha,0.0,samples)
    es_Ha = np.concatenate((es_Ha_bound,es_Ha_unbound))
    print 'Calculating the full spectrum up to %.6f eV.\n' % highest_state_eV
    splist_Ha, spdick_Ha = jel.spectrum(es_Ha,-1,V0_Ha,R0_bohr,R1_bohr)
    splist_eV = fun.spec_ha2ev(splist_Ha,0.0)
    print 'Highest occ level %s (should be same as Ef=%s).' % (splist_eV[Efindex][0],Ef_eV)

    # radial grid
    if deltar_nm == 'auto':
        deltar_nm = np.pi/2./np.sqrt(2.*(highest_state_Ha - V0_Ha))*qs.bohr2nm/6.
    Rmin_nm    = 1e-6
    grid_sam   = int((R1_nm - Rmin_nm)/deltar_nm)
    rgrid_nm   = np.linspace(Rmin_nm, R1_nm, grid_sam) # radial grid
    rgrid_bohr = rgrid_nm / qs.bohr2nm

    print 'CHECK: deltar: %s nm. Must be < %s nm.' % (deltar_nm, np.pi/2./np.sqrt(2.*(highest_state_Ha - V0_Ha))*qs.bohr2nm)

    # save spectrum to file
    spec_path = os.path.join(flag_path,'spectrum_eV_R0=%.3f_V0=%.3f.txt' % (R0_nm,V0_eV))
    np.savetxt(spec_path,splist_eV,header=io.spec_header(material, R0_nm, R1_nm, V0_eV, ns, Ef_eV))

    # save dos to file
    dos_path = os.path.join(flag_path,'dos_eV_R0=%.3f_V0=%.3f.txt' % (R0_nm,V0_eV))
    emin_eV = splist_eV[0][0]-4.*brd_eV
    emax_eV = splist_eV[-1][0]+4.*brd_eV
    en_grid_eV = np.linspace(emin_eV,emax_eV,3000)
    en_grid_Ha = en_grid_eV/qs.ha2ev
    dos = jel.calculate_dos(splist_eV,brd_eV,en_grid_eV)
    np.savetxt(dos_path,dos,header=io.dos_header(material, R0_nm, V0_eV, ns, brd_eV, Ef_eV))

    tot_elecs = np.zeros(len(photons_eV))
    tot_holes = np.zeros(len(photons_eV))
    hc_powers = np.zeros(len(photons_eV))
    qs_powers = np.zeros(len(photons_eV))

    for jj, phot_freq_eV in enumerate(photons_eV):
        phot_freq_Ha = phot_freq_eV/qs.ha2ev
        print '-----------------------------------------------'
        print "Photon energy %.3f eV.\n" % (phot_freq_eV)

        #plasmon potential and QS power at photon frequency
        eps = eps_material(phot_freq_eV)
        epsm = eps_medium(phot_freq_eV)
        indm = n_medium(phot_freq_eV)
        abs_nm2 = qs.absorption(phot_freq_eV,eps,epsm,R0_nm,indm)
        qs_power_watt = intensity_wattnm2 * abs_nm2
        pot_eV = qs.plasmon_radpot(rgrid_nm,eps,epsm,R0_nm,efield_nm) # in Volts
        # change units of pot to Ha/e
        pot_Ha = pot_eV/qs.ha2ev

        # slice spectrum in occupied and unoccupied states
        occlist_Ha, unocclist_Ha = fun.slice_speclist(splist_Ha,Ef_Ha,phot_mult*phot_freq_Ha,maxl)
        occdick_Ha, unoccdick_Ha = fun.slice_specdick(spdick_Ha,Ef_Ha,phot_mult*phot_freq_Ha,maxl)

        mat = jel.newMatrix(unocclist_Ha,occdick_Ha,R0_bohr,V0_Ha,pot_Ha,rgrid_bohr,maxl)

        ## matrix elements to file
        #matrix_path1 = os.path.join(flag_path,'mat_elements_rad_v2.txt')
        #matrix_path2 = os.path.join(flag_path,'mat_elements_ang_v2.txt')
        #inp = open(matrix_path1,'w')
        #inp.write(json.dumps(mat.rad,indent=4))
        #inp.close()
        #inp = open(matrix_path2,'w')
        #inp.write(json.dumps(mat.ang,indent=4))
        #inp.close()

        electrons = jel.new_hot_carrier(unocclist_Ha,occdick_Ha,0.0,Ef_Ha,phot_freq_Ha,elph_Ha,density_bohr,mat)
        holes     = jel.new_hot_carrier(occlist_Ha,unoccdick_Ha,0.0,Ef_Ha,phot_freq_Ha,elph_Ha,density_bohr,mat)

        # hot carrier number and power
        volume_nm = 4.0/3.0*np.pi*(R0_nm)**3
        hc_power_watt = qs.hcpower(electrons, holes, Ef_Ha)

        tot_elecs[jj] = qs.hcnumber(electrons) / qs.hatime2s / 1e12 / volume_nm
        tot_holes[jj] = qs.hcnumber(holes)     / qs.hatime2s / 1e12 / volume_nm
        hc_powers[jj] = hc_power_watt
        qs_powers[jj] = qs_power_watt

        # power ratio for energy conservation
        power_ratio = hc_power_watt / qs_power_watt

        # save raw distribution to file
        raw_elec_dist_path = os.path.join(flag_path,'raw_elec_dist_R0=%.3f_medium=%s_phot=%.3f.txt' % (R0_nm,medium,phot_freq_eV))
        raw_hole_dist_path = os.path.join(flag_path,'raw_hole_dist_R0=%.3f_medium=%s_phot=%.3f.txt' % (R0_nm,medium,phot_freq_eV))
        np.savetxt(raw_elec_dist_path, electrons, header=io.raw_dist_header(material,'electrons',R0_nm,R1_nm,medium,phot_freq_Ha,Ef_Ha,efield_nm))
        np.savetxt(raw_hole_dist_path, holes, header=io.raw_dist_header(material,'holes',R0_nm,R1_nm,medium,phot_freq_Ha,Ef_Ha,efield_nm))

        # add broadening to the elec hole distribution
        # using a single energy grid for both electrons and holes
        # probabily faster if using two different ones
        emax_dist_Ha = splist_Ha[-1][0] + 5.*brd_Ha
        emin_dist_Ha = splist_Ha[0][0] - 5.*brd_Ha
        en_grid_dist_Ha = np.linspace(emin_dist_Ha,emax_dist_Ha,1000)
        en_grid_dist_eV = en_grid_dist_Ha*qs.ha2ev
        elec_dist = jel.energy_dist(electrons,en_grid_dist_Ha,brd_Ha)
        hole_dist = jel.energy_dist(holes,en_grid_dist_Ha,brd_Ha)

        # volume normalisation and energy rescaling
        elec_dist_scal = elec_dist / qs.hbar_eVps / volume_nm / power_ratio
        hole_dist_scal = hole_dist / qs.hbar_eVps / volume_nm / power_ratio
        elec_dist_noscal = elec_dist / qs.hbar_eVps / volume_nm
        hole_dist_noscal = hole_dist / qs.hbar_eVps / volume_nm
        data_scal = (en_grid_dist_eV,elec_dist_scal,en_grid_dist_eV,hole_dist_scal)
        data_noscal = (en_grid_dist_eV,elec_dist_noscal,en_grid_dist_eV,hole_dist_noscal)

        # file name paths
        dist_path = os.path.join(flag_path,'dist_scal_R0=%.3f_medium=%s_phot=%.3f.txt' % (R0_nm,medium,phot_freq_eV))
        np.savetxt(dist_path,data_scal,header=io.dist_header(material,R0_nm,R1_nm,medium,phot_freq_eV,elph_eV,brd_eV,Ef_eV,efield_nm,power_ratio))
        dist_path = os.path.join(flag_path,'dist_noscal_R0=%.3f_medium=%s_phot=%.3f.txt' % (R0_nm,medium,phot_freq_eV))
        np.savetxt(dist_path,data_noscal,header=io.dist_header(material,R0_nm,R1_nm,medium,phot_freq_eV,elph_eV,brd_eV,Ef_eV,efield_nm,power_ratio))

    # save tot_num and power vs photon, scaled and not scaled
    ratio = hc_powers / qs_powers
    elec_noscal = (photons_eV, tot_elecs)
    elec_scal = (photons_eV, tot_elecs/ratio)
    hole_noscal = (photons_eV, tot_holes)
    hole_scal = (photons_eV, tot_holes/ratio)
    powers = (photons_eV, qs_powers, hc_powers, ratio)

    elec_num_path = os.path.join(flag_path,'elec_num_noscal_R0=%.3f_medium=%s.txt' % (R0_nm,medium))
    np.savetxt(elec_num_path,elec_noscal,header=io.carrier_header(material,'electron',R0_nm,R1_nm,medium,elph_eV,brd_eV,Ef_eV,efield_nm,highest_state_eV))
    elec_num_path = os.path.join(flag_path,'elec_num_scal_R0=%.3f_medium=%s.txt' % (R0_nm,medium))
    np.savetxt(elec_num_path,elec_scal,header=io.carrier_header(material,'electron',R0_nm,R1_nm,medium,elph_eV,brd_eV,Ef_eV,efield_nm,highest_state_eV))

    hole_num_path = os.path.join(flag_path,'hole_num_noscal_R0=%.3f_medium=%s.txt' % (R0_nm,medium))
    np.savetxt(hole_num_path,hole_noscal,header=io.carrier_header(material,'hole',R0_nm,R1_nm,medium,elph_eV,brd_eV,Ef_eV,efield_nm,highest_state_eV))
    hole_num_path = os.path.join(flag_path,'hole_num_scal_R0=%.3f_medium=%s.txt' % (R0_nm,medium))
    np.savetxt(hole_num_path,hole_scal,header=io.carrier_header(material,'hole',R0_nm,R1_nm,medium,elph_eV,brd_eV,Ef_eV,efield_nm,highest_state_eV))

    power_path = os.path.join(flag_path,'power_R0=%.3f_medium=%s.txt' % (R0_nm,medium))
    np.savetxt(power_path,powers,header=io.power_header(material,R0_nm,R1_nm,medium,elph_eV,Ef_eV,efield_nm,intensity_wattnm2))
    print
    print 'DONE: %s, radius %s.' % (material,R0_nm)

