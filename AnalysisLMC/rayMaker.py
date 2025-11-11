#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Zipeng Hu, Modified by Hilay Shah

import numpy as np
from numba import njit, prange, jit
from cfpack import stop, constants
import h5py
import yt
import yt.units as u
import matplotlib.pyplot as plt
import unyt
import cooling_rate as cr
from meshoid import Meshoid
import cloudy_controls as cc
import os
import time
from scipy.spatial import cKDTree
import numba as nb
from scipy import interpolate
import pickle

# @njit(fastmath=True)
def norm(coordinate):
    N = len(coordinate)
    vectorNorm = np.zeros(N)
    for i in range(N):
        vectorNorm[i] = np.sqrt(np.sum(coordinate[i]**2))
    return vectorNorm
# @njit(fastmath=True)
def gizmoKernel(q):
    k = np.zeros(len(q))
    k[q <= 0.5] = 1 - 6*q[q <= 0.5]**2 + 6*q[q <= 0.5]**3
    mid = np.where((q > 0.5) & (q < 1))
    k[mid]= 2 * (1-q[mid])**3
    return k
# @njit(fastmath=True)
def rayInterpolate(p, coordinate, rayLength, bins, sml, ne, BLOS, RMDM): # RMDM could either be 'DM' or 'RM'
#p, coordinate, rayLength, sml in parsec;
#VLOS in km/s; T in K; nHI in cm**-3; Mu dimensionless; BLOS in uG
    print(p)
    d = np.linalg.norm(coordinate[:,0:2]-p*unyt.pc, axis=1)
    ratio = d/sml; ray = np.where(ratio < 1); ne = ne[ray]; sml1 = sml[ray]
    B1 = BLOS[ray]
    rayBLOS = np.zeros(bins)
    rayne = np.zeros(bins); dl = rayLength/bins
    rayIntegral = np.zeros(bins)
    pz = []
    for l in range(bins):
        p1 = np.array([p[0],p[1], (l+0.5)*dl-rayLength/2])*unyt.pc
        pz.append(p1)
        d1 = np.linalg.norm(coordinate[ray]-p1, axis=1)*unyt.pc
        ratio1 = d1/sml1; ngb = np.where(ratio1 < 1)
        if len(ngb[0]) > 0:
            kernel = gizmoKernel(ratio1[ngb]);
            rayne[l] = np.average(ne[ngb], weights = kernel)
            if RMDM == 'RM':
                rayBLOS[l] = np.average(B1[ngb], weights = kernel)
                rayIntegral[l] = 0.81*rayne[l]*rayBLOS[l]*dl
            if RMDM == 'DM':
                rayIntegral[l] = rayne[l]*dl
            if RMDM == 'EM':
                rayIntegral[l] = rayne[l]**2*dl
        else:
            rayIntegral[l] = 0
    return p, rayIntegral.sum(), np.array(pz), np.array(rayIntegral)

def PulsarRayInterpolate(p, coordinate, rayLength, bins, sml, ne):
    # This function would actually calculate DM as if they were from pulsars along the line of sight
    #p, coordinate, rayLength, sml in parsec; ne in cm**-3
    print(p)
    d = np.linalg.norm(coordinate[:,0:2]-p[0:2]*unyt.pc, axis=1)
    # los = np.where(coordinate[:,2] > p[2])
    ratio = d/sml; ray = np.where(ratio < 1); ne = ne[ray]; sml1 = sml[ray]

    rayne = np.zeros(bins); dl = rayLength/bins
    rayIntegral = np.zeros(bins)
    pz = []
    for l in range(bins):
        p1 = np.array([p[0],p[1], l*dl+p[2]])*unyt.pc
        pz.append(p1)
        d1 = np.linalg.norm(coordinate[ray]-p1, axis=1)*unyt.pc
        # stop()
        ratio1 = d1/sml1; ngb = np.where(ratio1 < 1)
        if len(ngb[0]) > 0:
            kernel = gizmoKernel(ratio1[ngb]);
            rayne[l] = np.average(ne[ngb], weights = kernel)
            rayIntegral[l] = rayne[l]*dl
        else:
            rayIntegral[l] = 0
    return p, rayIntegral.sum(), np.array(pz), np.array(rayIntegral)

def PropRayInterpolate(p, coordinate, rayLength, bins, sml, prop):
    # This function would actually calculate properties along LOS from a given point to the rayLength
    print(p)
    d = np.linalg.norm(coordinate[:,0:2]-p[0:2]*unyt.pc, axis=1)
    # los = np.where(coordinate[:,2] > p[2])
    ratio = d/sml; ray = np.where(ratio < 1); prop = prop[ray]; sml1 = sml[ray]
    # ind_rayLOS = los[0][rayLOS]

    rayprop = np.zeros(bins); dl = rayLength/bins
    rayIntegral = np.zeros(bins)
    pz = []
    for l in range(bins):
        p1 = np.array([p[0], p[1], (l+0.5)*dl-rayLength/2])*unyt.pc
        pz.append(p1)
        d1 = np.linalg.norm(coordinate[ray]-p1, axis=1)*unyt.pc
        # stop()
        ratio1 = d1/sml1; ngb = np.where(ratio1 < 1)
        if len(ngb[0]) > 0:
            kernel = gizmoKernel(ratio1[ngb]);
            rayprop[l] = np.average(prop[ngb], weights = kernel)
            # rayIntegral[l] = rayprop[l]*dl
        else:
            rayprop[l] = 0
            # rayIntegral[l] = 0
    return p, rayprop.sum(), np.array(pz), np.array(rayprop)

def histogram(array, bins):
    hist, bin_edges = np.histogram(array, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    return hist, bin_centers

def convert_T_to_mu(grackle_T):
    T_mu = np.load("/scratch/jh2/hs9158/results/data/T_mu_interp.npy")
    T = T_mu[0]; mu = T_mu[1]
    # concatenate the arrays to make sure the interpolation works
    T = np.concatenate((T, [1e11, 1e13]))
    mu = np.concatenate((mu, [0.5983, 0.5983]))
    T_mu_interp = interpolate.interp1d(T, mu)
    mu = T_mu_interp(grackle_T)
    return mu

def convert_mu_to_nenH(mu):
    nenH = 1.33/mu - 1.079
    # make all negative values of nenH to 0
    nenH[np.where(nenH < 0)] = 0
    return nenH

def CorrectIons(args, data, i):
    print("Correcting ionization fractions using Grackle data...")
    ad = data.all_data()
    T = ad["PartType0","temperature"].in_units('K')
    HII = ad["PartType0","GrackleHII"]
    HII[(T > 10766.883) & (T<10766.884)] = 0.745
    def _HII(field, data):
        return HII
    
    print("HII to be corrected and added!")

    data.add_field(name=("PartType0","HII"), function=_HII, sampling_type = 'local', units="dimensionless", force_override=True)

    print("HII corrected and added!")

    # print("HII corrected and added!")
    print("Correcting ne using Grackle data...")
    def _ne(field, data):
        neHII = data["PartType0","density"].in_units('g/cm**3')*data["PartType0","HII"]/u.mh
        HeII = data["PartType0","GrackleHeII"]; HeIII = data["PartType0","GrackleHeIII"]
        # T = data["PartType0","temperature"].in_units('K'); ion = np.where((T > 10766.8) & (T<10766.9))
        # HeII[ion] = 0.235; HeIII[ion] = 0
        neHeII = data["PartType0","density"].in_units('g/cm**3')*HeII/(4*u.mh)
        neHeIII = data["PartType0","density"].in_units('g/cm**3')*HeIII/(2*u.mh)
        return(neHII + neHeII + neHeIII)
    data.add_field(name=("PartType0","ne"), function=_ne, sampling_type = 'local', units="1/cm**3", force_override=True)
    print("ne corrected and added!")
    def _np(field, data):
        return(data["PartType0","density"].in_units('g/cm**3')*data["PartType0","HII"]/u.mh)
    data.add_field(name=("PartType0","np"), function=_np, sampling_type = 'local', units="1/cm**3", force_override=True)
    print("Proton number densities corrected and added!")

    def _hydrogen_number_density(field, data):
        return data["PartType0","density"].in_units('g/cm**3')*(data["PartType0","GrackleHI"]+data["PartType0","GrackleHII"])/u.mh  # Hydrogen number density
    
    data.add_field(name=("PartType0","hydrogen_number_density"), function=_hydrogen_number_density, sampling_type = 'local', units="1/cm**3", force_override=True)
    print("Hydrogen number densities added!")

    def _molecular_weight(field, data):
        nHI = data["PartType0","density"].in_units('g/cm**3')*data["PartType0","GrackleHI"]/u.mh  # HI number density
        nHII = data["PartType0","density"].in_units('g/cm**3')*data["PartType0","HII"]/u.mh  # HII number density
        nHeI = data["PartType0","density"].in_units('g/cm**3')*data["PartType0","GrackleHeI"]/(4*u.mh)
        nHeII = data["PartType0","density"].in_units('g/cm**3')*data["PartType0","GrackleHeII"]/(4*u.mh)
        nHeIII = data["PartType0","density"].in_units('g/cm**3')*data["PartType0","GrackleHeIII"]/(4*u.mh)
        mu = (nHI + nHII + 4*nHeI + 4*nHeII + 4*nHeIII)/(nHI + 2*nHII + nHeI + 2*nHeII + 3*nHeIII) # mass per free particle
        # change nan values of mu to 1.22
        mu[np.where(np.isnan(mu))] = 1.22
        return mu
    
    data.add_field(name=("PartType0","molecular_weight"), function=_molecular_weight, sampling_type = 'local', units="dimensionless", force_override=True)
    print("Molecular weights corrected and added!")
    # stop()

    def _particle_number_density(field, data):
        return data["PartType0","density"].in_units('g/cm**3')/(data["PartType0","molecular_weight"]*u.mh)
    
    data.add_field(name=("PartType0","particle_number_density"), function=_particle_number_density, sampling_type = 'local', units="1/cm**3", force_override=True)
    
    def _temperature(field, data):
        return data["PartType0","InternalEnergy"].in_units('ergs/g')*data["PartType0","molecular_weight"]*u.mh*2/3/(u.kboltz_cgs)
    data.add_field(name=("PartType0","calc_temperature"), function=_temperature, sampling_type = 'local', units="K", force_override=True)
    # stop()
    if os.path.exists(args.snappath+'snapshot_{}_temperature.npy'.format(str(i).zfill(3))):
        grackle_temperature = np.load(args.snappath+'snapshot_{}_temperature.npy'.format(str(i).zfill(3)), allow_pickle=True)        
        def _grackle_temperature(field, data):
            return grackle_temperature*unyt.K
        data.add_field(name=("PartType0","grackle_temperature"), function=_grackle_temperature, sampling_type = 'local', units="K", force_override=True)
        print("Corrected temperatures added!")
        mu = convert_T_to_mu(grackle_temperature)
        nenH = convert_mu_to_nenH(mu)
    else: print("No temperature correction file found!")
    X = 0.7524; Y = 0.2376; Z = 0.01
    ad = data.all_data()

    nH = ad["PartType0","hydrogen_number_density"]
    ind_correct = np.where(nH == 0)[0]
    nH[ind_correct] = ad["PartType0","density"][ind_correct].in_units('g/cm**3')*(X)/u.mh
    ne = ad["PartType0","ne"]
    if os.path.exists(args.snappath+'snapshot_{}_temperature.npy'.format(str(i).zfill(3))):
        ne[ind_correct] = nH[ind_correct]*nenH[ind_correct]

    def _hydrogen_number_density(field, data):
        return nH.in_units('1/cm**3')
    def _ne(field, data):
        return ne.in_units('1/cm**3')
    data.add_field(name=("PartType0","hydrogen_number_density"), function=_hydrogen_number_density, sampling_type = 'local', units="1/cm**3", force_override=True)
    data.add_field(name=("PartType0","ne"), function=_ne, sampling_type = 'local', units="1/cm**3", force_override=True)
    print("Hydrogen number densities and electron number densities corrected and added!")
    # ad = data.all_data()
    # T = cr.calculate_temperature(density = ad["PartType0", "particle_number_density"].in_units('1/cm**3'),
    #                                     temperature = ad["PartType0", "Temperature"].in_units('K'),
    #                                     internal_energy = ad["PartType0", "InternalEnergy"].in_units('km**2/s**2'))
    # def _grackle_temperature(field, data):
    #     return grackle_temperature*unyt.K
    # data.add_field(name=("PartType0","grackle_temperature"), function=_grackle_temperature, sampling_type = 'local', units="K", force_override=True)
    # print("Temperatures corrected and added!")
    return data

def compute_density_gradient(args, data, Tmin, Tmax):
    "This function computes density gradient for a given snapshot using Meshoid module"
    "and returns the gradient"
    ptype = "PartType0"
    ad_ion_rot = data.all_data()
    coords = ad_ion_rot[ptype, 'Coordinates'].in_units('cm')
    rho = ad_ion_rot[ptype, 'density'].in_units('g/cm**3')
    sml = ad_ion_rot[ptype, 'SmoothingLength'].in_units('cm')
    B = ad_ion_rot[ptype, 'MagneticField']*1e6
    if os.path.exists(args.snappath+'snapshot_{}_processed_quantities_T{}.npy'.format(str(args.snapnum).zfill(3), args.TRe_key)):
        print("Processed quantities already exist for this snapshot.")
        arr = np.load(args.snappath+'snapshot_{}_processed_quantities_T{}.npy'.format(str(args.snapnum).zfill(3), args.TRe_key), allow_pickle=True)
        arrdata = arr.item()
        # stop()
        ne_corrected_photoionzed = arrdata['ne_corrected_photoionized']
        ne_corrected = arrdata['ne_corrected']
        T = arrdata['grackle_temperature_photo']
    else:
        print("Process the quantities for this snapshot.", error=True)
        # ad_ion_rot = data.all_data()
        ne_corrected_photoionzed = data[ptype, 'ne_corrected_photoionized']
        ne_corrected = data[ptype, 'ne_corrected']
    ind_T = np.where((T > Tmin) & (T < Tmax))[0]
    ind_not_T = np.isin(np.arange(len(ne_corrected_photoionzed)), ind_T, invert=True)
    # make values of coords, rho, sml, B only exist for ind_T, make rest of the values zero
    coords[ind_not_T] = 0; rho[ind_not_T] = 0; sml[ind_not_T] = 0; B[ind_not_T] = 0
    M = Meshoid(coords, rho, sml)
    grad_weights = M.D(rho)

    grad = np.linalg.norm(grad_weights, axis=1)*unyt.g/unyt.cm**4
    coldens = rho**2/grad
    nH = coldens*0.73/constants.m_p/unyt.g
    # add the fields to the dataset
    def _nH(field, data):
        return nH
    def _grad_rho(field, data):
        return grad
    def _coldens(field, data):
        return coldens
    def _B_T(field, data):
        return B
    stop()
    data.add_field(name=("PartType0","nHcol"), function=_nH, sampling_type = 'local', units="cm**-(2)", force_override=True)
    data.add_field(name=("PartType0","grad_rho"), function=_grad_rho, sampling_type = 'local', units="g/cm**3", force_override=True)
    data.add_field(name=("PartType0","coldens"), function=_coldens, sampling_type = 'local', units="g/cm**2", force_override=True)
    data.add_field(name=("PartType0","B_T"), function=_B_T, sampling_type = 'local', units="g/cm**2", force_override=True)
    print("Fields nH, grad_rho, coldens, B_T added to the dataset.")
    return data

def make_temperature_density_cut(args, data, Tmin=1e1, Tmax=1e4, cond='col', col=1e19, rho=1e-1):  # cond could be 'col' or 'rho' or 'T'.
    "This function makes a temperature cut for a given snapshot"
    "based on the temperature range, anything else would provide temperature"
    print("Making temperature and density cut for the dataset with Tmin = {} K and Tmax = {} K".format(Tmin, Tmax))
    ptype = "PartType0"
    data_ions = data # it should be corrected already
    if cond == 'col':
        data_ions = compute_density_gradient(args, data_ions)
    ad_ion_rot = data_ions.all_data()
    coords = ad_ion_rot[ptype, 'Coordinates'].in_units('cm')
    T = ad_ion_rot[ptype, 'grackle_temperature']
    ind_T = np.where((T > Tmin) & (T < Tmax))[0]
    if cond == 'col':
        coldens = ad_ion_rot[ptype, 'coldens']
        nHcol = ad_ion_rot[ptype, 'nHcol']
        ind_col = np.where(nHcol > col)[0]
        ind_T_col = np.intersect1d(ind_T, ind_col)
        return data_ions, ind_T_col
    elif cond == 'rho':
        ind_rho = np.where(ad_ion_rot[ptype, 'particle_number_density']>rho)[0]
        ind_T_rho = np.intersect1d(ind_T, ind_rho)
        return data_ions, ind_T_rho
    else:
        return data_ions, ind_T
    
def Correctne(args, data, Tmin=1e1, Tmax=1e4, cond='col', col=1e19, rho=1e-1):
    X = 0.73; Y = 0.25; Z = 0.02 # approximate mass densities
    ptype = "PartType0"
    data_ions, ind = make_temperature_density_cut(args, data, Tmin=Tmin, Tmax=Tmax, cond=cond, col=col, rho=rho)
    ad_ion_rot = data_ions.all_data()
    ind_nH = np.where((ad_ion_rot[ptype, 'hydrogen_number_density'] <= 1e3)
                       & (ad_ion_rot[ptype, 'hydrogen_number_density'] >= 1e-3))[0]
    ind_nH_cond = np.intersect1d(ind, ind_nH)
    ptype = "PartType0"
    if Tmax==1e4:
        density = np.logspace(-3, 3, 61); temperature = np.logspace(1, 4, 61)
    else:
        density = np.logspace(-3, 3, 61); temperature = np.logspace(1, 4.3, 71)
    print("Running CLOUDY to correct electron number density values...")
    ne_interp = cc.ion_density_temperature(density=density, temperature=temperature, extinguish=True, ext_den=21)
    # with open('/scratch/jh2/hs9158/results/data/ne_interp.pck', 'wb') as f:
    #     pickle.dump(ne_interp, f)

    with open('/scratch/jh2/hs9158/results/data/ne_interp.pck', 'rb') as file_handle:
        ne_interp_load = pickle.load(file_handle)

    print("CLOUDY run completed.")
    ne = ad_ion_rot[ptype, 'ne']
    ne_corrected = np.copy(ne)*unyt.cm**(-3)
    ne_corrected[ind_nH_cond] = ne_interp((ad_ion_rot[ptype,"hydrogen_number_density"][ind_nH_cond],
                                            ad_ion_rot[ptype,"grackle_temperature"][ind_nH_cond]))
    def _ne_corrected(field, data):
        return ne_corrected
    data.add_field(name=("PartType0","ne_corrected"), function=_ne_corrected, sampling_type = 'local', units="1/cm**3", force_override=True)
    print("ne_corrected field added to the dataset using CLOUDY.")
    return data

def create_ionizing_luminosity_data(args, data, mode, compile=False): # mode could be create or load, create calls SlugAnalysis, read reads the data from previously created files
    "This function calls the c++ function SlugAnalysis which extract ionizing luminosity values for all slug particles."
    "This function is called for a given snapshot and the output is saved in a file."
    # check if parttype4 exists in the dataset
    F = h5py.File(args.snappath+args.filename,"r")
    if 'PartType4' not in F.keys():
        print("PartType4 does not exist in the dataset.")
        return data
        F.close()
    else:
        if mode == 'create':
            os.chdir("/scratch/jh2/hs9158/results/codes/SlugAnalysis")
            if compile: os.system("make clean"); os.system("make") 
            os.system("./SlugAnalysis {} {} {} {}".format(args.resolution, args.scenario, args.snapnum, args.key))
            print("SlugAnalysis done for snapshot {}".format(args.snapnum))
        # read the data from a text file with a demilimiter of space character
        ionizing_luminosity = np.loadtxt(args.snappath+"ionizing_luminosities_{}.txt".format(str(args.snapnum).zfill(3)), delimiter=' ', dtype=float)
        # add the ionizing luminosity values to the yt dataset
        def _ionizing_luminosity(field, data):
            return ionizing_luminosity
        data.add_field(("PartType4", "ionizing_luminosity"), function=_ionizing_luminosity, sampling_type = 'local', units="", force_override=True)
        print("Ionizing luminosity field added to the dataset. Field name = (PartType4, ionizing_luminosity)")
        return data

def CorrectneCloudySlug(args, i, Tmin=1e1, Tmax=1e4, cond='col', col=1e19, rho=1e-1, ne_corrected=True, mode='load', compile=False): # ne_corrected controls if CLOUDY correction is applied or not
    "This function will correct the electron number density values based on both CLOUDY and Slug models in the ISM"
    data = yt.load(args.snappath+"snapshot_{}.hdf5".format(str(i).zfill(3)))
    # First we compute the recombination rates of the gas particles
    args.snapnum = i
    args.filename = "snapshot_{}.hdf5".format(str(i).zfill(3))
    X = 0.7524; Y = 0.2376 # this is extracted from summing the mass fractions of H and He from Grackle data
    def recom_coeff(T):
        return 2.54e-13*(T/1e4)**(-0.8163-0.0208*np.log(T/1e4))
    
    data = CorrectIons(args, data, i)
    if ne_corrected: data = Correctne(args, data, Tmin=Tmin, Tmax=Tmax, cond=cond, col=col, rho=rho)
    photo_Tmin = 1e1; photo_Tmax = 2e4
    data, ind_T = make_temperature_density_cut(args, data, Tmin=photo_Tmin, Tmax=photo_Tmax, cond='T', col=col, rho=rho)
    # Assuming a constant temperature of 1e4 K
    beta = recom_coeff(1e4)

    def _recombination_rate(field, data):
        return beta*unyt.cm**3/unyt.s*(data['PartType0', 'Density'].in_units('g/cm**3')/(constants.m_p*unyt.g))**2*(X+Y/4)*X*4/3*np.pi*data['PartType0', 'SmoothingLength'].in_units('cm')**3
    data.add_field(name=("PartType0","recombination_rate"), function=_recombination_rate, sampling_type = 'local', units="1/s", force_override=True)
    print("Recombination rates added to the dataset. Field name = (PartType0,recombination_rate)")
    data = create_ionizing_luminosity_data(args, data, mode=mode, compile=compile)
    print("Ionizing luminosity data added to the dataset.")
    

    # make arrays containing coordinates, particleID, Q/R
    
    ad_ion_rot = data.all_data()
    ionization_array = np.zeros((len(ad_ion_rot['PartType4', 'ParticleIDs']), 5))
    
    ionization_array[:,0:3] = ad_ion_rot['PartType4', 'Coordinates']; ionization_array[:,3:5] = ad_ion_rot['PartType4', 'ionizing_luminosity']
    sorted_indices = np.argsort(ionization_array[:, 4])[::-1]
    ionization_array_sorted = ionization_array[sorted_indices]
    print("Number of slug sources = ", len(ionization_array_sorted))
    # removing particles with ionizing luminosity below the threshold
    if args.resolution == 'lowres': ionizing_threshold = 1e46; batchsize = 100
    if args.resolution == 'medres': ionizing_threshold = 1e46; batchsize = 250
    if args.resolution == 'medhighres': ionizing_threshold = 0.3e46; batchsize = 500
    if args.resolution == 'highres': ionizing_threshold = 0.12e46; batchsize = 1000
    ind_Q = np.where(ionization_array_sorted[:, 4] > ionizing_threshold)[0]
    ionization_array_sorted = ionization_array_sorted[ind_Q]
    print("Number of ionizing slug sources = ", len(ionization_array_sorted))
    print("Ionizing threshold = ", ionizing_threshold)
    ne_nh = ad_ion_rot['PartType0', 'ne_corrected']/ad_ion_rot['PartType0', 'hydrogen_number_density']
    ind_ne_nh = np.where(ne_nh < 0.5)[0]
    ind_ne_nh_T = np.intersect1d(ind_ne_nh, ind_T)
    # find all gas particles within 0.5 kpc radius of the ionizing sources
    # stop()
    recombination_array = np.zeros((len(ad_ion_rot['PartType0', 'ParticleIDs'][ind_ne_nh_T]), 5))
    recombination_array[:,0:3] = ad_ion_rot['PartType0', 'Coordinates'][ind_ne_nh_T]
    recombination_array[:, 3] = ad_ion_rot['PartType0', 'ParticleIDs'][ind_ne_nh_T]
    recombination_array[:,4] = ad_ion_rot['PartType0', 'recombination_rate'][ind_ne_nh_T]
    Nbatch = np.ceil(len(ionization_array_sorted)/batchsize).astype(int)
    print("Per cent of gas particles below ionizing threshold = {} %".format(100*np.where(recombination_array[:,4]<ionizing_threshold)[0].shape[0]/len(recombination_array)))
    ad = data.all_data()
    age = ad['PartType4', 'SlugStateDouble'][:, 13]/1e6
    # Start a time test to see how long it takes to run the photoionization code
    start = time.time()
    recombination_array_sorted = recombination_array.copy()
    K = 40000
    # Initialize a set to store IDs of photoionized gas particles
    photoionized_ids = []
    photoionized_inds = []
    print("Running photoionization calculation...")
    for N in range(Nbatch):
        print("Batch number = ", N)
        star_ind = np.arange(N*batchsize, min((N+1)*batchsize, len(ionization_array_sorted)))
        gas_tree = cKDTree(recombination_array_sorted[:,0:3], leafsize=16)  # Experiment with different leaf sizes
        distances, indices = gas_tree.query(ionization_array_sorted[star_ind, 0:3], k=K, workers=4)
        next_indices = indices[0]
        next_keep_indices = np.ones(len(indices[0])).astype(bool)
        photoionized_inds_batch = []
        test = []
        for i in range(len(star_ind)):
            Q = ionization_array_sorted[star_ind[i], 4]
            
            sort_ind_gas_neighbours = next_indices[np.argsort(distances[i][next_keep_indices])]
            for j in sort_ind_gas_neighbours:
                if Q >= recombination_array_sorted[j, 4]:
                    photoionized_ids.append(recombination_array_sorted[j, 3])
                    photoionized_inds_batch.append(j)
                    Q -= recombination_array_sorted[j, 4]
                else:
                    test.append(Q)
                    recombination_array_sorted[j, 4] -= Q
                    break
            if i >= len(star_ind)-1:
                # print("Batch done.")
                break
            next_keep_indices = ~np.isin(indices[i+1], photoionized_inds_batch)
            next_indices = indices[i+1][next_keep_indices]
            
            # print("length of photoionized_ids = ", len(photoionized_inds))
            # print("length of next indices = ", len(next_indices))
        # remove the indices corresponding to photoionized_inds from recombination_array_sorted
        print("Batch ionization rate = {}".format(np.sum(ionization_array_sorted[star_ind,4])))
        print("Batch recombination rate = {}".format(np.sum(recombination_array_sorted[photoionized_inds_batch,4])))
        print("Batch last gas recombination rate = {}".format(np.sum(test)))
        recombination_array_sorted = recombination_array_sorted[~np.isin(np.arange(len(recombination_array_sorted)), photoionized_inds_batch)]

        print("length of recombination_array_sorted = ", len(recombination_array_sorted))
        print("Time taken for batch = ", time.time()-start)

    # Print time taken to run the photoionization code
    print("Time taken to run photoionization code: ", time.time()-start)
    # Convert set to list and print the result
    photoionized_particle_ids = np.array(list(photoionized_ids)).astype(int)
    photoionized_particle_inds = np.where(np.isin(ad_ion_rot['PartType0', 'ParticleIDs'], photoionized_particle_ids))[0]
    print(str(100*len(photoionized_particle_ids)/len(ind_T))+" % gas particles photoionized within T={}-{}K".format(str(photo_Tmin), str(photo_Tmax)))
    # The particle ids with a lot of collisional ionization (ne/nH) must be removed now.

    photoionized_flag = np.zeros(len(ad_ion_rot['PartType0', 'ParticleIDs']))
    photoionized_flag[photoionized_particle_inds] = 1

    
    # find these photoionized_particle_ids in ('PartType0', 'ParticleIDs') and add a field to the dataset setting a flag for photoionized particles
    def _photoionized_flag(field, data):
        return photoionized_flag
    data.add_field(name=("PartType0","photoionized_flag"), function=_photoionized_flag, sampling_type = 'local', units="", force_override=True)
    print("Photoionized flag added to the dataset. Field name = (PartType0,photoionized_flag)")

    ind_photo = np.where(ad_ion_rot['PartType0', 'photoionized_flag'] == 1)[0]
    print("Number of photoionized particles = ", len(ind_photo))
    ne_corrected_photoionized = np.copy(ad_ion_rot['PartType0', 'ne_corrected'])*unyt.cm**(-3)
    ne_corrected_photoionized[ind_photo] = ad_ion_rot['PartType0', 'Density'].in_units('g/cm**3')[ind_photo]/(constants.m_p*unyt.g)*(X+Y/4)  # assuming fully ionized hydrogen, singly ionized He
    T = ad_ion_rot['PartType0', 'grackle_temperature'].in_units('K')
    T_photo = np.copy(T)
    T_photo[ind_photo[np.where(T[ind_photo]<1e4)[0]]] = 1e4
    T_photo = T_photo*unyt.K    

    # add another function to compute the electron number density for photoionized particles
    def _ne_corrected_photoionized(field, data):
        return ne_corrected_photoionized
    
    # add a function to add the temperature for photoionized particles
    def _T_photo(field, data):
        return T_photo
    data.add_field(name=("PartType0","ne_corrected_photoionized"), function=_ne_corrected_photoionized, sampling_type = 'local', units="1/cm**3", force_override=True)
    data.add_field(name=("PartType0","T_photo"), function=_T_photo, sampling_type = 'local', units="K", force_override=True)
    print("ne_corrected_photoionized field added to the dataset. Field name = (PartType0,ne_corrected_photoionized)")
    print("T_photo field added to the dataset. Field name = (PartType0,T_photo)")
    processed_data = {}
    processed_data['ne_corrected_photoionized'] = ad_ion_rot['PartType0', 'ne_corrected_photoionized']
    processed_data['ne_corrected'] = ad_ion_rot['PartType0', 'ne_corrected']
    processed_data['ne'] = ad_ion_rot['PartType0', 'ne']
    processed_data['grackle_temperature_photo'] = ad_ion_rot['PartType0', 'T_photo']
    processed_data['grackle_temperature'] = ad_ion_rot['PartType0', 'grackle_temperature']
    processed_data['photoionized_flag'] = ad_ion_rot['PartType0', 'photoionized_flag']
    processed_data['ionizing_luminosity'] = ad_ion_rot['PartType4', 'ionizing_luminosity']
    processed_data['recombination_rate'] = ad_ion_rot['PartType0', 'recombination_rate']
    processed_data['hydrogen_number_density'] = ad_ion_rot['PartType0', 'hydrogen_number_density']
    processed_data['particle_number_density'] = ad_ion_rot['PartType0', 'particle_number_density']
    processed_data['molecular_weight'] = ad_ion_rot['PartType0', 'molecular_weight']
    # stop()
    # save the processed data as a numpy array in the snappath directory
    np.save(args.snappath+"snapshot_{}_processed_quantities_T{}.npy".format(str(args.snapnum).zfill(3), args.TRe_key), processed_data)


    # yt.save_as_dataset(data, args.snappath+"snapshot_{}_processed_quantities.h5".format(str(args.snapnum).zfill(3)), processed_data)
    # ps = yt.load(args.snappath+"snapshot_{}_processed_quantities.h5".format(str(args.snapnum).zfill(3)))
    print("Processed quantities saved to snapshot_{}_processed_quantities.npy".format(str(args.snapnum).zfill(3)))
    # sleep for 5 seconds to allow the system to save the file
    if args.resolution == 'lowres': time.sleep(60); print("Sleeping for 60 seconds")
    if args.resolution == 'medres': time.sleep(1500); print("Sleeping for 150 seconds")
    if args.resolution == 'medhighres': time.sleep(3000); print("Sleeping for 300 seconds")
    if args.resolution == 'highres': time.sleep(6000); print("Sleeping for 600 seconds")
    return data

def extended_interp(nH, T, interp_func):
    output = np.zeros_like(nH)
    
    # Condition 1: T > 19500 K → output = nH (any nH)
    mask_T_high = T > 19500
    output[mask_T_high] = nH[mask_T_high]
    
    # Condition 2: T between 10-19500 K
    mask_T_medium = (T >= 10) & (T <= 19500)
    
    # Subcondition 2a: nH < 1e-3 → output = nH
    mask_nH_low = (nH < 1e-3) & mask_T_medium
    output[mask_nH_low] = nH[mask_nH_low]
    
    # Subcondition 2b: nH > 1e3 → output = 0
    mask_nH_high = (nH > 1e3) & mask_T_medium
    output[mask_nH_high] = 0
    
    # Valid interpolation region (remaining cases)
    mask_valid = (~mask_T_high) & (~mask_nH_low) & (~mask_nH_high) & mask_T_medium
    output[mask_valid] = interp_func((nH[mask_valid], T[mask_valid]))
    
    return output

def make_spectrum_processed_data(args, snap):
    # this function creates another processed data file with the same name as the original one but with a different key and more processed data
    # read the processed quantities
    filename = 'snapshot_{}.hdf5'.format(str(snap).zfill(3))
    snappath = '/scratch/jh2/hs9158/results/LMC_run_{}_wind_scenario_{}{}/'.format(args.resolution, args.scenario, args.key)
    if os.path.exists(snappath+"snapshot_{}_processed_quantities_T{}.npy".format(str(snap).zfill(3), args.TRe_key)):
        print("Processed quantities already exist for this snapshot.")
        arr = np.load(snappath+"snapshot_{}_processed_quantities_T{}.npy".format(str(snap).zfill(3), args.TRe_key), allow_pickle=True)
        arrdata = arr.item()
        # stop()
        ne_corrected_photoionized = arrdata['ne_corrected_photoionized']
        T = arrdata['grackle_temperature_photo']
    # correct all electron densities where they are zero from the cloudy data
    ind_zero = np.where(ne_corrected_photoionized==0)[0]
    F = h5py.File(snappath + filename, 'r')
    XH = 0.752; XHe = 0.2376; XZ = 1-XH-XHe

    # Calculate number densities
    nH = (F['PartType0']['Density'][:] * 1e10 * constants.m_sol / 
        (1e3 * constants.pc)**3 / constants.m_p * XH)
    nHe = (F['PartType0']['Density'][:] * 1e10 * constants.m_sol / 
        (1e3 * constants.pc)**3 / (4 * constants.m_p) * XHe)

    ne_corrected_photoionized = np.asarray(ne_corrected_photoionized)
    with open('/scratch/jh2/hs9158/results/data/ne_interp.pck', 'rb') as file_handle:
        ne_interp_load = pickle.load(file_handle)
    ne_corrected_photoionized[ind_zero] = extended_interp(nH[ind_zero], T[ind_zero], ne_interp_load)

    # Initialize arrays
    nHI = np.zeros_like(ne_corrected_photoionized)
    nHII = np.zeros_like(ne_corrected_photoionized)
    nHeI = np.zeros_like(ne_corrected_photoionized)
    nHeII = np.zeros_like(ne_corrected_photoionized)
    nHeIII = np.zeros_like(ne_corrected_photoionized)

    # Create masks for different ionization regimes
    mask_h_only = (ne_corrected_photoionized > 0) & (ne_corrected_photoionized <= nH)
    mask_he_partial = (ne_corrected_photoionized > nH) & (ne_corrected_photoionized <= nH + 2*nHe)
    mask_he_full = ne_corrected_photoionized > nH + 2*nHe
    # Hydrogen-only ionization
    nHI[mask_h_only] = nH[mask_h_only] - ne_corrected_photoionized[mask_h_only]
    nHII[mask_h_only] = ne_corrected_photoionized[mask_h_only]
    nHeI[mask_h_only] = nHe[mask_h_only]

    # Helium partial/full ionization
    nHII[mask_he_partial | mask_he_full] = nH[mask_he_partial | mask_he_full]
    n_eHe = ne_corrected_photoionized - nH

    # Handle helium ionization states in single lines
    mask_he_single = mask_he_partial & (n_eHe <= nHe)
    mask_he_double = mask_he_partial & (n_eHe > nHe)

    nHeI[mask_he_single] = nHe[mask_he_single] - n_eHe[mask_he_single]
    nHeII[mask_he_single] = n_eHe[mask_he_single]

    nHeII[mask_he_double] = 2 * nHe[mask_he_double] - n_eHe[mask_he_double]
    nHeIII[mask_he_double] = n_eHe[mask_he_double] - nHe[mask_he_double]

    # Full helium ionization
    nHeIII[mask_he_full] = nHe[mask_he_full]

    # calculate the molecular weight
    mu = (nH+4*nHe)/(nH+nHe+ne_corrected_photoionized)

    # save the processed quantities in the same format as the original processed quantities but with ne_corrected_photoionized, hydrogen_number_density, and molecular_weight replaced with updated values and
    # add the new quantities for nHI, nHII, nHeI, nHeII, nHeIII
    arrdata['ne_corrected_photoionized'] = ne_corrected_photoionized
    arrdata['hydrogen_number_density'] = nH
    arrdata['molecular_weight'] = mu
    arrdata['nHI'] = nHI
    arrdata['nHII'] = nHII
    arrdata['nHeI'] = nHeI
    arrdata['nHeII'] = nHeII
    arrdata['nHeIII'] = nHeIII
    # save the new processed quantities
    np.save(snappath+"snapshot_{}_spectrum_processed_quantities_T{}.npy".format(str(snap).zfill(3), args.TRe_key), arrdata)
    print("Processed quantities saved to file.")

def test_yt_issues(args, data):
    data = Correctne(args, data, Tmin=1e1, Tmax=10**(4.3), cond='T', col=1e-19, rho=1)
    photo_Tmin = 1e1; photo_Tmax = 2e4
    data, ind_T = make_temperature_density_cut(args, data, Tmin=photo_Tmin, Tmax=photo_Tmax, cond='T', col=1e-19, rho=1)

    def _recombination_rate(field, data):
        return 1e-13*unyt.cm**3/unyt.s*(data['PartType0', 'Density'].in_units('g/cm**3')/(constants.m_p*unyt.g))**2*(0.7524+0.2376/4)*0.7524*4/3*np.pi*data['PartType0', 'SmoothingLength'].in_units('cm')**3
    data.add_field(name=("PartType0","recombination_rate"), function=_recombination_rate, sampling_type = 'local', units="1/s", force_override=True)
    print("Recombination rates added to the dataset. Field name = (PartType0,recombination_rate)")
    ad = data.all_data()
    # save this as a new processed dataset
    processed_data = {}
    processed_data['ne_corrected'] = ad['PartType0', 'ne_corrected']
    processed_data['recombination_rate'] = ad['PartType0', 'recombination_rate']
    # yt.save_as_dataset(data, args.snappath+"snapshot_{}_processed_test.h5".format(str(args.snapnum).zfill(3)), processed_data)
    # print("Processed quantities saved to snapshot_{}_processed_test.h5".format(str(args.snapnum).zfill(3)))
    return data

def get_plane_rotation(args, rotation_matrix):
    # define an equation of plane with the normal in the direction of the z-axis
    normal = np.array([0,0,1])
    node = np.array([0,1,0])
    # rotate the normal vector to the new frame
    normal_rot = np.dot(rotation_matrix, normal.T).T
    node_rot = np.dot(rotation_matrix, node.T).T
    args.normal_rot = normal_rot
    args.node_rot = node_rot
    def plane_rot(coord0, coord1, coord2): # two points on the plane and one point to compute based on the normal, the point to compute is supposed to be True
        # determine which two argments are float and which one is a flag
        if coord0: coord = -(normal_rot[1]*coord1+normal_rot[2]*coord2)/normal_rot[0]
        if coord1: coord = -(normal_rot[0]*coord0+normal_rot[2]*coord2)/normal_rot[1]
        if coord2: coord = -(normal_rot[0]*coord0+normal_rot[1]*coord1)/normal_rot[2]
        return coord
    args.plane_rot = plane_rot
    return print("Plane rotation function created and stored in args.")


def RotateGalaxy(args, ad_rot, rotation_matrix, center=np.array([0,0,0])):
    ad_rot['PartType0', 'Coordinates'] = (np.dot(rotation_matrix, (np.array(ad_rot['PartType0', 'Coordinates'])-center).T).T+center)*unyt.kpc
    ad_rot['PartType0', 'Velocities'] = np.dot(rotation_matrix, ad_rot['PartType0', 'Velocities'].T).T
    ad_rot['PartType0', 'MagneticField'] = np.dot(rotation_matrix, ad_rot['PartType0', 'MagneticField'].T).T
    ad_rot['PartType4', 'Coordinates'] = (np.dot(rotation_matrix, (np.array(ad_rot['PartType4', 'Coordinates'])-center).T).T+center)*unyt.kpc
    print("Output rotated galaxy data")
    return ad_rot

if __name__ == "__main__":
    stop()