#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Hilay Shah 2023-

import numpy as np
import h5py
import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import os
import argparse
import yt
from meshoid import Meshoid
import meshoid
from cfpack import print, hdfio, stop
import matplotlib.colors as colors
from matplotlib import cm
import imageio
import multiprocessing
from joblib import Parallel, delayed
from cfpack import constants
from multiprocessing import Pool
import rayMaker as rm
from functools import partial
from scipy.ndimage import convolve
import plotting as pl
import concurrent.futures as cofu
import cooling_rate as cr
import unyt
import cloudy_controls as cc
from scipy.interpolate import griddata
from scipy.spatial import distance_matrix
import export_to_stl as ex
import surf2stl
from scipy import integrate
import time
import dirty_parallel as dp
import RM_analysis as rma
import logging
import bootstrapping as boot
logging.basicConfig(format='%(message)s')  # Only show the message without prefixes

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def convert_to_cylindrical(pos_xyz, quantity_xyz):  # does not work for arrays with multiple rows
    r, phi = cart2pol(pos_xyz[:, 0], pos_xyz[:, 1])
    qr = quantity_xyz[:, 0]*np.cos(phi)+quantity_xyz[:, 1]*np.sin(phi)
    qphi = -quantity_xyz[:, 0]*np.sin(phi)+quantity_xyz[:, 1]*np.cos(phi)
    qz = quantity_xyz[:, 2]
    return np.transpose(np.array([qr, qphi, qz]))

def my_axis_field0(field, data):
    return data["PartType0", "Coordinates"][:, 0]

def my_axis_field1(field, data):
    return data["PartType0", "Coordinates"][:, 1]

def my_axis_field2(field, data):
    return data["PartType0", "Coordinates"][:, 2]

def gif_maker(args, plotpath, snapnumstart, snapnumend, plane='z', subplot=False, rotated=False, projected=False):
    images = []
    if projected: proj_arg = 'projected'
    else: proj_arg = 'sliced'
    for i in range(snapnumstart, snapnumend+1):
        if subplot=='movie':
            if rotated: filename = plotpath+'movie_{}_plane{}_rotated_{}_ccom.png'.format(i, plane, proj_arg)
            else: filename = plotpath+'movie_{}_plane{}_{}.png'.format(i, plane, proj_arg)
        elif subplot==True:
            if rotated: filename = plotpath+'{}_plane{}_rotated_{}_ccom.png'.format(i, plane, proj_arg)
            else: filename = plotpath+'{}_plane{}_{}.png'.format(i, plane, proj_arg)
        else: print("I don't know what to do.")
        images.append(imageio.imread(filename))
    # making the shapes of all arrays same
    shapes = []
    for i in range(snapnumend+1):
        shapes.append(np.array(images[i].shape))
    shapes = np.vstack(shapes)
    for i in range(shapes.shape[1]):
        if np.min(shapes[:, i]) == np.max(shapes[:, i]):
            continue
        else:
            print(i)
            min = np.min(shapes[:, i])
            for k in range(shapes.shape[0]):
                images[k] = np.delete(images[k], slice(0, shapes[k][i]-min), axis=i)
    if subplot=='movie':
        if rotated: imageio.mimsave("/scratch/jh2/hs9158/results/plots/gifs/movie_subplot_part0_plane{}_LMC_{}_wind_rotated_{}_scenario{}.gif".format(plane, args.resolution, proj_arg, args.scenario), images, duration=0.4,  # 10 FPS (1/10 = 0.1 seconds per frame)
                    loop=0)
        else: imageio.mimsave("/scratch/jh2/hs9158/results/plots/gifs/movie_subplot_part0_plane{}_LMC_{}_wind_scenario{}.gif".format(plane, args.resolution, args.scenario), images)
    elif subplot==True: 
        if rotated: imageio.mimsave("/scratch/jh2/hs9158/results/plots/gifs/gif_subplot_part0_plane{}_LMC_{}_wind_rotated_{}_scenario{}.gif".format(plane, args.resolution, proj_arg, args.scenario), images)
        else: imageio.mimsave("/scratch/jh2/hs9158/results/plots/gifs/gif_subplot_part0_plane{}_LMC_{}_wind_scenario{}.gif".format(plane, args.resolution, args.scenario), images)
    else: print("I don't know what to do.")

def new_gif_maker(args):
    images = []
    # Get all PNG files from the RM_movie directory
    for i in range(100):  # Assuming 100 frames from 0 to 99
        filename = args.plotpath + 'RM_movie/movie_plot_{}.png'.format(i)
        try:
            images.append(imageio.imread(filename))
        except FileNotFoundError:
            print(f"File not found: {filename}")
            continue

    # Verify we have images
    if not images:
        print("No images found!")
        return

    # Making the shapes of all arrays same
    shapes = []
    for i in range(len(images)):
        shapes.append(np.array(images[i].shape))
    shapes = np.vstack(shapes)
    
    # Normalize image sizes if needed
    for i in range(shapes.shape[1]):
        if np.min(shapes[:, i]) == np.max(shapes[:, i]):
            continue
        else:
            print(f"Normalizing dimension {i}")
            min_size = np.min(shapes[:, i])
            for k in range(shapes.shape[0]):
                images[k] = np.delete(images[k], slice(0, shapes[k][i]-min_size), axis=i)

    # Save the GIF
    output_path = args.plotpath + 'RM_movie/RM_animation.gif'
    imageio.mimsave(output_path, 
                    images, 
                    duration=0.1,  # 10 FPS (1/10 = 0.1 seconds per frame)
                    loop=0)        # 0 means loop forever
    
    print(f"GIF saved to: {output_path}")

def add_grackle_temperatures(args, ds):  # this function adds grackle temperatures to the dataset and saves it as hdf5 file in the snapshot directory
    # add grackle temperatures to the dataset
    ds = rm.create_ionizing_luminosity_data(args, ds, 'create')
    # save the dataset as hdf5 file
    ds.save_as_dataset(filename=args.snappath+args.filename)
    return ds


def calculate_SFR(args, snaprange):
    particletype = 'PartType4'
    stellarmass = []
    age_average = 5e6 # in yr
    # check if F.keys() contains the string particletype
    for i in range(snaprange+1):
        print(i)
        # check if snapshot file exists, else continue
        filename = 'snapshot_{}.hdf5'.format(str(i).zfill(3))
        if not os.path.exists(args.snappath + filename):
            print(f"File not found: {filename}")
            continue
        F = h5py.File(args.snappath+filename,"r+")
        if particletype in F.keys():
            # find partiles with age younger than age_average
            # print("Total Star mass = ", 1e10*F[particletype]["Masses"][:].sum())

            current_time = F['Header'].attrs['Time']
            # if StellarFormationTime is negative, assign it to zero when calculating real_age
            real_age = ((current_time - np.maximum(F['PartType4']['StellarFormationTime'][:], 0))*1e9)
            # real_age = ((current_time - F['PartType4']['StellarFormationTime'][:])*1e9)
            # check if RealAge already exists in the dataset
            if 'RealAge' in F['PartType4'].keys():
                print("RealAge already exists in the dataset.")
                # del F['PartType4']['RealAge']
                # add real age to the PartType4 dataset
            else:
                F['PartType4'].create_dataset('RealAge', data=real_age)
                stop()
            if i>20:
                F_prev = h5py.File(args.snappath+'snapshot_{}.hdf5'.format(str(i-1).zfill(3)),"r")
                # print("Difference in total star mass = ", 1e10*F[particletype]["Masses"][:].sum()-1e10*F_prev[particletype]["Masses"][:].sum())
                F_prev.close()
            # if i>=95:
            #     partID2046577age = F[particletype]['RealAge'][np.where(F[particletype]["ParticleIDs"][:]==4361)[0]]
            #     print("Age of particle 2046577 = ", partID2046577age)
            # print("oldest star age = ", real_age.max()/1e6)
            ind = np.where(real_age < age_average)[0]
            stellarmass.append(F[particletype]["Masses"][:][ind].sum())
            # find indices with common 'ParticleIDs' in F and F_prev
            # ind_common = np.where(np.isin(F[particletype]["ParticleIDs"][:], F_prev[particletype]["ParticleIDs"][:]))[0]
            # print("Total Young Star mass = ", 1e10*F[particletype]["Masses"][:][ind].sum())
            F.close()
        else:
            stellarmass.append(0)
    # particle ID = 867049
    SFR = np.array(stellarmass)/args.tstep*1e10 # in solar masses/yr?
    np.savetxt(args.snappath+'SFR.txt', SFR, delimiter=' ', fmt='%1.6e')
    return SFR

def calculate_temperature(args, snaprange):

    particletype = 'PartType0'

    temperature = []
    for i in range(snaprange):
        args.filename = 'snapshot_{}.hdf5'.format(str(i).zfill(3))
        F = h5py.File(args.snappath+args.filename,"r")
        ParticleIDs = F['PartType0']['ParticleIDs'][:]
        # ind where particleIDs are in the range of IDs of the disk particles
        ind_disk = np.where((ParticleIDs > 0) & (ParticleIDs < args.ngasdisk))[0]
        if particletype in F.keys():
            T = F[particletype]["InternalEnergy"][:][ind_disk]*1e10*1.3*constants.m_p*2/3/(constants.k_b)
            temperature.append(np.mean(T))
        else:
            temperature.append(0)
    np.savetxt(args.snappath+'temperature.txt', temperature, delimiter=' ', fmt='%1.6e')
    return temperature

def calculate_mass_outflow_rate(args, snaprange, radius, height, rotation_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
    # Rotate the relevant fields by the rotation matrix to ensure that the z-axis is aligned with the galaxy's angular momentum vector inside the loop
    dr = 0.05 # in kpc
    nz = 100
    nphi = 100
    dz = 2*height/nz # in kpc
    dphi = 2*np.pi/nphi
    dvol = radius*dphi*dz*dr
    particletype = 'PartType0'
    mass_flux_list = []
    for i in range(snaprange):
        print(i)
        args.filename = 'snapshot_{}.hdf5'.format(str(i).zfill(3))
        F = h5py.File(args.snappath+args.filename,"r")
        coord = F[particletype]["Coordinates"][:]
        mass = F[particletype]["Masses"][:]*1e10 # in solar masses
        hsml = F[particletype]["SmoothingLength"][:]/1e3  # in kpc
        v = F[particletype]["Velocities"][:]
        coord = np.dot(coord, rotation_matrix)
        v = np.dot(v, rotation_matrix)
        r_cyl, phi_cyl = cart2pol(coord[:, 0], coord[:, 1])
        # Find the particles close to a cylinder of radius and height
        
        ind1 = np.where((r_cyl < radius+dr) & (r_cyl >= radius) & (np.abs(coord[:, 2]) < height))[0]
        ind2 = np.where((r_cyl < radius) & (np.abs(coord[:, 2]) < height+dz) & (np.abs(coord[:, 2]) >= height))[0]
        ind = []
        # ind.append(ind1_out[np.where(r_cyl[ind1_out]-hsml[ind1_out]<=radius)[0]])
        # ind.append(ind1_in[np.where(r_cyl[ind1_in]+hsml[ind1_in]>=radius)[0]])
        # ind.append(ind2_out[np.where(np.abs(coord[ind2_out, 2])-hsml[ind2_out]<=height)[0]])
        # ind.append(ind2_in[np.where(np.abs(coord[ind2_in, 2])+hsml[ind2_in]>=height)[0]])
        # Run a loop over ind1 to find particles in small bins defined by dz and dphi
        mass_flux = 0
        for j in range(nz):
            for k in range(nphi):
                z_inst = -height+dz*(j+1/2)
                phi_inst = dphi*(k+1/2)
                area_inst = radius*dphi*dz*np.array([np.cos(phi_inst), np.sin(phi_inst), 0])
                ind1_inst = np.where((r_cyl[ind1] < radius+dr) & (r_cyl[ind1] >= radius) &
                                    (coord[ind1, 2] < z_inst+dz/2) & (coord[ind1, 2] > z_inst-dz/2) &
                                    (phi_cyl[ind1] > phi_inst-dphi/2) & (phi_inst < phi_inst+dphi/2))[0]
                if len(ind1_inst) > 0:
                    velocity_area = np.zeros(3)
                    mass_area = 0
                    for l in range(len(ind1_inst)):
                        velocity_area = velocity_area+v[ind1[ind1_inst[l]]]
                        mass_area = mass_area+mass[ind1[ind1_inst[l]]]
                    mass_flux_area = mass_area/dvol*np.dot(velocity_area*1e5/constants.pc/1e3*constants.year, area_inst)  # velocity in kpc/year, area in kpc^2, mass flux in Msol/year
                    # print(mass_flux_area, len(ind1_inst), j, k)
                    mass_flux = mass_flux+mass_flux_area
                else:continue
        mass_flux_list.append(mass_flux)        
                        
    # np.savetxt(args.snappath+'mass_outflow.txt', mass_outflow_list, delimiter=' ', fmt='%1.6e')
    # np.savetxt(args.snappath+'mass_inflow.txt', mass_inflow_list, delimiter=' ', fmt='%1.6e')
    np.savetxt(args.snappath+'mass_flux_rate.txt', mass_flux_list, delimiter=' ', fmt='%1.6e')
    return mass_flux_list

def make_temperature_cut(args, data, Tmin, Tmax):
    "This function makes a temperature cut for a given snapshot"
    "based on the temperature range"
    ptype = "PartType0"
    # check if the processed quantities already exist for this snapshot
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
    # yt_fields = np.array(data.field_list)
    # yt_parttype0_fields = yt_fields[np.where(yt_fields[:,0] == 'PartType0')]
    # for field in yt_parttype0_fields:
    #     print("Cutting temperature for field: ", field[1])
    #     ad_ion_rot[field[0], field[1]] = ad_ion_rot[field[0], field[1]][ind_T]

    # # making the cut for some important derived fields
    # ad_ion_rot[ptype, 'ne_corrected_photoionized'] = ad_ion_rot[ptype, 'ne_corrected_photoionized'][ind_T]
    # ad_ion_rot[ptype, 'ne_corrected'] = ad_ion_rot[ptype, 'ne_corrected'][ind_T]
    # ad_ion_rot[ptype, 'ne'] = ad_ion_rot[ptype, 'ne'][ind_T]
    
    # ad_ion_rot[ptype, 'grackle_temperature'] = ad_ion_rot[ptype, 'grackle_temperature'][ind_T]
    # ad_ion_rot[ptype, 'particle_number_density'] = ad_ion_rot[ptype, 'particle_number_density'][ind_T]
    # ad_ion_rot[ptype, 'hydrogen_number_density'] = ad_ion_rot[ptype, 'hydrogen_number_density'][ind_T]
    # print("Number of particles in the temperature range: ", len(ind_T))
    # print("Temperature range: ", np.min(T[ind_T]), np.max(T[ind_T]))
    # print("Temperature cut made for all fields in PartType0")
    # ne_n = ad_ion_rot[ptype, 'ne_corrected']/ad_ion_rot[ptype, 'particle_number_density']
    # nH = ad_ion_rot[ptype, 'hydrogen_number_density']
    # n_highi = ad_ion_rot[ptype, 'particle_number_density'][np.where(ne_n>0.1)[0]]
    # n_lowi = ad_ion_rot[ptype, 'particle_number_density'][np.where(ne_n<0.1)[0]]
    # plot a histogram of hydrogen number desity (nH) with logarithmic binning in the nH direction
    # plt.figure()
    # plt.hist(nH, bins=np.logspace(np.log10(np.min(nH)), np.log10(np.max(nH)), 100), label='nH')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('nH')
    # plt.ylabel('Counts')
    # plt.title('Temperature range: {}-{}'.format(Tmin, Tmax))
    # plt.savefig(args.plotpath+'nH_hist.png', dpi=300)
    # plt.close()

    # # plot the histogram of n_highi and n_lowi
    # y, x = rm.histogram(n_highi, 10000)
    # y_n, x_n = rm.histogram(n_lowi, 10000)
    # plt.figure()
    # plt.plot(x, y, label='ne/n > 0.1')
    # plt.plot(x_n, y_n, label='ne/n < 0.1')
    # plt.xlabel('Particle number density')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.ylabel('Counts')
    # plt.title('Temperature range: {}-{}'.format(Tmin, Tmax))
    # plt.savefig(args.plotpath+'ne_n_hist.png', dpi=300)

    # plt.figure(figsize=(5,5)); plt.scatter(ad_ion_rot[ptype, 'particle_number_density'], ne_n); plt.xscale('log'); plt.yscale('log')
    # plt.xlabel('Particle number density (cm**-3)'); plt.ylabel('ne/n')
    # plt.title('Temperature range: {}-{}'.format(Tmin, Tmax))
    # plt.savefig(args.plotpath+'ne_n_density.png', dpi=300)
    # # plt.show()

    # plt.figure(figsize=(5,5)); plt.scatter(ad_ion_rot[ptype, 'density'], ne_n); plt.xscale('log'); plt.yscale('log')
    # plt.xlabel('density (g*cm**-3)'); plt.ylabel('ne/n')
    # plt.title('Temperature range: {}-{}'.format(Tmin, Tmax))
    # plt.savefig(args.plotpath+'ne_n_massdensity.png', dpi=300)
    # plt.scatter(coords_highn[:,0], coords_highn[:,1], s=0.1, alpha=0.5, label='strange particles'); plt.scatter(coordsSF[:, 0], coordsSF[:, 1], s=0.1, alpha=0.5, label='stars'); plt.xlim([90, 110]); plt.ylim([90, 110]);plt.gca().set_aspect('equal'); plt.legend(); plt.show()

    return ind_T

def create_RM_map(args, ad_ion_rot, extent, resolution, R, R_obs, RMDM): # extent in kpc, resolution in sources/kpc^2
    "This function creates an RM map for a given snapshot based on the resolution and range and rotation matrix"
    "using the functions from the rayMaker.py code"
    # data_ions = rm.CorrectIons(args, data, i)
    # ad_ion_rot = data_ions.all_data()
    ind_T = make_temperature_cut(args, ad_ion_rot, args.Tmin, args.Tmax)
    print("Creating {} map for snapshot: ".format(RMDM), args.snapnum)
    ptype = "PartType0"
    # check if the processed quantities already exist for this snapshot
    if os.path.exists(args.snappath+"snapshot_{}_processed_quantities_T{}.npy".format(str(args.snapnum).zfill(3), args.TRe_key)):
        print("Processed quantities already exist for this snapshot.")
        arr = np.load(args.snappath+"snapshot_{}_processed_quantities_T{}.npy".format(str(args.snapnum).zfill(3), args.TRe_key), allow_pickle=True)
        arrdata = arr.item()
        # stop()
        ne_corrected_photoionzed = arrdata['ne_corrected_photoionized']
        ne_corrected = arrdata['ne_corrected']
        T = arrdata['grackle_temperature_photo']
    else:
        print("Process the quantities for this snapshot.", error=True)
        ne_corrected_photoionzed = ad_ion_rot[ptype, 'ne_corrected_photoionized']
        ne_corrected = ad_ion_rot[ptype, 'ne_corrected']

    # extract all indices except the ind_T
    ind_not_T = np.isin(np.arange(len(ne_corrected_photoionzed)), ind_T, invert=True)
    # make the relevant cuts
    ne_corrected_photoionzed[ind_not_T] = 0
    ne_corrected[ind_not_T] = 0
    # ne = ad_ion_rot[ptype, 'ne']
    ad_ion_rot = rm.RotateGalaxy(args, ad_ion_rot, R, center=args.center)
    ad_ion_rot = rm.RotateGalaxy(args, ad_ion_rot, R_obs, center=args.center)  # rotate to observer's frame
    coords = ad_ion_rot[ptype, 'Coordinates'].in_units('pc')-(args.center*1e3)*unyt.pc
    # ParticleID = ad_ion_rot[ptype, 'ParticleIDs']
    # ind_disk = np.where((ParticleID > 0) & (ParticleID < args.ngasdisk))[0]
    sml = ad_ion_rot[ptype, 'SmoothingLength'].in_units('pc')
    np.random.seed(0)
    points = np.column_stack((np.random.uniform(low=-extent*1e3/2, high=extent*1e3/2, size=int(resolution)),
                        np.random.uniform(low=-extent*1e3/2, high=extent*1e3/2, size=int(resolution)))) # in pc

    rayLength = 100000 # in parsec
    bins = 100000
    BLOS = ad_ion_rot[ptype, 'MagneticField'][:,2]*1e6 # in uG
    args_list = [(point) for point in points]
    # rm.rayInterpolate(points[0], coords, rayLength, bins, sml, ne_corrected_photoionzed, BLOS, 'RM')
    # Create a Pool object
    with Pool() as pool:
        results = pool.map(partial(rm.rayInterpolate, 
                                   coordinate=coords, 
                                   rayLength=rayLength, 
                                   bins=bins, 
                                   sml=sml,
                                   ne=ne_corrected_photoionzed, 
                                   BLOS=BLOS,
                                   RMDM=RMDM), args_list)

    # unpack the results
    points, RM, pointz, RMz = zip(*results)
    save_array = aa = np.column_stack((points, RM))

    # stack all arrays in RMz tuple
    stack_RMz = aa = np.column_stack((np.column_stack(RMz), pointz[0][:,2]))
    
    np.savez(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}.npz".format(RMDM, resolution, args.snapnum, args.Tmin, args.Tmax, args.TRe_key), data=save_array, column_names = ['x_pc', 'y_pc', 'RM'])
    np.savez(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}z.npz".format(RMDM, resolution, args.snapnum, args.Tmin, args.Tmax, args.TRe_key), data=stack_RMz)
    return points, RM

def create_RM_map_ion(args, ad_ion_rot, extent, resolution, R, R_obs, RMDM, imin, imax): # extent in kpc, resolution in sources/kpc^2
    "This function creates an RM map for a given snapshot based on the resolution and range and rotation matrix"
    "using the functions from the rayMaker.py code"
    # data_ions = rm.CorrectIons(args, data, i)
    # ad_ion_rot = data_ions.all_data()
    ind_T = make_temperature_cut(args, ad_ion_rot, args.Tmin, args.Tmax)
    print("Creating {} map for snapshot: ".format(RMDM), args.snapnum)
    ptype = "PartType0"
    # check if the processed quantities already exist for this snapshot
    if os.path.exists(args.snappath+"snapshot_{}_processed_quantities_T{}.npy".format(str(args.snapnum).zfill(3), args.TRe_key)):
        print("Processed quantities already exist for this snapshot.")
        arr = np.load(args.snappath+"snapshot_{}_processed_quantities_T{}.npy".format(str(args.snapnum).zfill(3), args.TRe_key), allow_pickle=True)
        arrdata = arr.item()
        # stop()
        ne_corrected_photoionzed = arrdata['ne_corrected_photoionized']
        ne_corrected = arrdata['ne_corrected']
        T = arrdata['grackle_temperature_photo']
        nH = arrdata['hydrogen_number_density']
    else:
        print("Process the quantities for this snapshot.", error=True)
        ne_corrected_photoionzed = ad_ion_rot[ptype, 'ne_corrected_photoionized']
        ne_corrected = ad_ion_rot[ptype, 'ne_corrected']

    istate = ne_corrected_photoionzed/nH
    ind_i = np.where((istate>imin/100) & (istate<imax/100))[0]
    ind_not_i = np.isin(np.arange(len(ne_corrected_photoionzed)), ind_i, invert=True)

    # extract all indices except the ind_T
    ind_not_T = np.isin(np.arange(len(ne_corrected_photoionzed)), ind_T, invert=True)
    # make the relevant cuts
    ne_corrected_photoionzed[ind_not_T] = 0
    ne_corrected[ind_not_T] = 0
    ne_corrected_photoionzed[ind_not_i] = 0
    ne_corrected[ind_not_i] = 0
    # ne = ad_ion_rot[ptype, 'ne']
    ad_ion_rot = rm.RotateGalaxy(args, ad_ion_rot, R, center=args.center)
    ad_ion_rot = rm.RotateGalaxy(args, ad_ion_rot, R_obs, center=args.center)  # rotate to observer's frame
    coords = ad_ion_rot[ptype, 'Coordinates'].in_units('pc')-(args.center*1e3)*unyt.pc
    # ParticleID = ad_ion_rot[ptype, 'ParticleIDs']
    # ind_disk = np.where((ParticleID > 0) & (ParticleID < args.ngasdisk))[0]
    sml = ad_ion_rot[ptype, 'SmoothingLength'].in_units('pc')
    np.random.seed(0)
    points = np.column_stack((np.random.uniform(low=-extent*1e3/2, high=extent*1e3/2, size=int(resolution)),
                        np.random.uniform(low=-extent*1e3/2, high=extent*1e3/2, size=int(resolution)))) # in pc

    rayLength = 100000 # in parsec
    bins = 100000
    BLOS = ad_ion_rot[ptype, 'MagneticField'][:,2]*1e6 # in uG
    args_list = [(point) for point in points]
    # rm.rayInterpolate(points[0], coords, rayLength, bins, sml, ne_corrected_photoionzed, BLOS, 'RM')
    # Create a Pool object
    with Pool() as pool:
        results = pool.map(partial(rm.rayInterpolate, 
                                   coordinate=coords, 
                                   rayLength=rayLength, 
                                   bins=bins, 
                                   sml=sml,
                                   ne=ne_corrected_photoionzed, 
                                   BLOS=BLOS,
                                   RMDM=RMDM), args_list)

    # unpack the results
    points, RM, pointz, RMz = zip(*results)
    save_array = aa = np.column_stack((points, RM))

    # stack all arrays in RMz tuple
    stack_RMz = aa = np.column_stack((np.column_stack(RMz), pointz[0][:,2]))
    
    np.savez(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}_i{}_{}.npz".format(RMDM, resolution, args.snapnum, args.Tmin, args.Tmax, args.TRe_key, imin, imax), data=save_array, column_names = ['x_pc', 'y_pc', 'RM'])
    np.savez(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}z_i{}_{}.npz".format(RMDM, resolution, args.snapnum, args.Tmin, args.Tmax, args.TRe_key, imin, imax), data=stack_RMz)
    return points, RM

def create_RM_map_nH(args, ad_ion_rot, extent, resolution, R, R_obs, RMDM, nHmin, nHmax): # extent in kpc, resolution in sources/kpc^2
    "This function creates an RM map for a given snapshot based on the resolution and range and rotation matrix"
    "using the functions from the rayMaker.py code"
    # data_ions = rm.CorrectIons(args, data, i)
    # ad_ion_rot = data_ions.all_data()
    ind_T = make_temperature_cut(args, ad_ion_rot, args.Tmin, args.Tmax)
    print("Creating {} map for snapshot: ".format(RMDM), args.snapnum)
    ptype = "PartType0"
    # check if the processed quantities already exist for this snapshot
    if os.path.exists(args.snappath+"snapshot_{}_processed_quantities_T{}.npy".format(str(args.snapnum).zfill(3), args.TRe_key)):
        print("Processed quantities already exist for this snapshot.")
        arr = np.load(args.snappath+"snapshot_{}_processed_quantities_T{}.npy".format(str(args.snapnum).zfill(3), args.TRe_key), allow_pickle=True)
        arrdata = arr.item()
        # stop()
        ne_corrected_photoionzed = arrdata['ne_corrected_photoionized']
        ne_corrected = arrdata['ne_corrected']
        T = arrdata['grackle_temperature_photo']
        nH = arrdata['hydrogen_number_density']
    else:
        print("Process the quantities for this snapshot.", error=True)
        ne_corrected_photoionzed = ad_ion_rot[ptype, 'ne_corrected_photoionized']
        ne_corrected = ad_ion_rot[ptype, 'ne_corrected']

    ind_nH = np.where((nH>nHmin) & (nH<nHmax))[0]
    ind_not_nH = np.isin(np.arange(len(ne_corrected_photoionzed)), ind_nH, invert=True)

    # extract all indices except the ind_T
    ind_not_T = np.isin(np.arange(len(ne_corrected_photoionzed)), ind_T, invert=True)
    # make the relevant cuts
    ne_corrected_photoionzed[ind_not_T] = 0
    ne_corrected[ind_not_T] = 0
    ne_corrected_photoionzed[ind_not_nH] = 0
    ne_corrected[ind_not_nH] = 0
    # ne = ad_ion_rot[ptype, 'ne']
    ad_ion_rot = rm.RotateGalaxy(args, ad_ion_rot, R, center=args.center)
    ad_ion_rot = rm.RotateGalaxy(args, ad_ion_rot, R_obs, center=args.center)  # rotate to observer's frame
    coords = ad_ion_rot[ptype, 'Coordinates'].in_units('pc')-(args.center*1e3)*unyt.pc
    # ParticleID = ad_ion_rot[ptype, 'ParticleIDs']
    # ind_disk = np.where((ParticleID > 0) & (ParticleID < args.ngasdisk))[0]
    sml = ad_ion_rot[ptype, 'SmoothingLength'].in_units('pc')
    np.random.seed(0)
    points = np.column_stack((np.random.uniform(low=-extent*1e3/2, high=extent*1e3/2, size=int(resolution)),
                        np.random.uniform(low=-extent*1e3/2, high=extent*1e3/2, size=int(resolution)))) # in pc

    rayLength = 100000 # in parsec
    bins = 100000
    BLOS = ad_ion_rot[ptype, 'MagneticField'][:,2]*1e6 # in uG
    args_list = [(point) for point in points]
    # rm.rayInterpolate(points[0], coords, rayLength, bins, sml, ne_corrected_photoionzed, BLOS, 'RM')
    # Create a Pool object
    with Pool() as pool:
        results = pool.map(partial(rm.rayInterpolate, 
                                   coordinate=coords, 
                                   rayLength=rayLength, 
                                   bins=bins, 
                                   sml=sml,
                                   ne=ne_corrected_photoionzed, 
                                   BLOS=BLOS,
                                   RMDM=RMDM), args_list)

    # unpack the results
    points, RM, pointz, RMz = zip(*results)
    save_array = aa = np.column_stack((points, RM))

    # stack all arrays in RMz tuple
    stack_RMz = aa = np.column_stack((np.column_stack(RMz), pointz[0][:,2]))
    
    np.savez(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}_nH{}_{}.npz".format(RMDM, resolution, args.snapnum, args.Tmin, args.Tmax, args.TRe_key, nHmin, nHmax), data=save_array, column_names = ['x_pc', 'y_pc', 'RM'])
    np.savez(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}z_nH{}_{}.npz".format(RMDM, resolution, args.snapnum, args.Tmin, args.Tmax, args.TRe_key, nHmin, nHmax), data=stack_RMz)
    return points, RM

def generate_random_pulsars(args, ad_ion_rot, resolution):
    SFR_coords = ad_ion_rot['PartType4', 'Coordinates'].in_units('pc')
    ClusterAge = ad_ion_rot['PartType4', 'SlugStateDouble'][:, 14]
    ind_SFR_age = np.where((ClusterAge > 5e6) & (ClusterAge<10e6))[0]
    if len(ind_SFR_age) < resolution:
        print("Number of SFR particles is less than the resolution.")
        resolution = len(ind_SFR_age)
        print("Setting resolution to: ", resolution)
    else: 
        np.random.seed(0)
        ind_SFR_age_res = np.random.choice(ind_SFR_age, size=int(resolution), replace=False)
    
    np.random.seed(0)
    SF_points = SFR_coords[ind_SFR_age_res]
    std_3d = 500; std_1d = std_3d/np.sqrt(3)
    # create a random sample from gaussian distribution with standard deviation 500, mean 0, and size resolution
    dx = np.random.normal(0, std_1d, resolution); dy = np.random.normal(0, std_1d, resolution); dz = np.random.normal(0, std_1d, resolution)
    pulsar_points = np.column_stack((SF_points[:, 0]+dx*unyt.pc, SF_points[:, 1]+dy*unyt.pc, SF_points[:, 2]+dz*unyt.pc))
    return SF_points, pulsar_points


def create_DM_map(args, ad_ion_rot, extent, resolution, R, R_obs): # extent in kpc, resolution in sources/kpc^2
    "This function creates an RM map for a given snapshot based on the resolution and range and rotation matrix"
    "using the functions from the rayMaker.py code"
    # data_ions = rm.CorrectIons(args, data, i)
    # ad_ion_rot = data_ions.all_data()
    ind_T = make_temperature_cut(args, ad_ion_rot, args.Tmin, args.Tmax)
    print("Creating DM pulsar map for snapshot:", args.snapnum)
    ptype = "PartType0"
    # check if the processed quantities already exist for this snapshot
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
        ne_corrected_photoionzed = ad_ion_rot[ptype, 'ne_corrected_photoionized']
        ne_corrected = ad_ion_rot[ptype, 'ne_corrected']

    
    # extract all indices except the ind_T
    ind_not_T = np.isin(np.arange(len(ne_corrected_photoionzed)), ind_T, invert=True)
    # make the relevant cuts
    ne_corrected_photoionzed[ind_not_T] = 0
    ne_corrected[ind_not_T] = 0
    ad_ion_rot = rm.RotateGalaxy(args, ad_ion_rot, R, center=args.center)
    ad_ion_rot = rm.RotateGalaxy(args, ad_ion_rot, R_obs, center=args.center)  # rotate to observer's frame
    BLOS = ad_ion_rot[ptype, 'MagneticField'][:,2]*1e6 # in uG
    coords = ad_ion_rot[ptype, 'Coordinates'].in_units('pc')
    
    np.random.seed(0)
    sml = ad_ion_rot[ptype, 'SmoothingLength'].in_units('pc')
    SF_points, pulsar_points = generate_random_pulsars(args, ad_ion_rot, resolution)

    bins = 100000
    rayLength = 50000
    # rm.rayInterpolate(pulsar_points[0][0:2], coords, rayLength, bins, sml, ne_corrected_photoionzed, BLOS, 'DM')
    # rm.PulsarRayInterpolate(pulsar_points[0], coords, rayLength, bins, sml, ne_corrected_photoionzed)

    args_list = [(point) for point in pulsar_points]
    # rm.rayInterpolate(points[0], coords, rayLength, bins, sml, ne, BLOS)
    # Create a Pool object
    with Pool() as pool:
        results = pool.map(partial(rm.PulsarRayInterpolate, 
                                   coordinate=coords,
                                   rayLength=rayLength,
                                   bins=bins, 
                                   sml=sml,
                                   ne=ne_corrected_photoionzed), args_list)

    # unpack the results
    points, DM, pointz, DMz = zip(*results)
    save_array = aa = np.column_stack((points, DM))
    # stack all arrays in RMz tuple
    stack_DMz = aa = np.column_stack((np.column_stack(DMz), pointz[0][:,2]))
    np.savez("/scratch/jh2/hs9158/results/data/PulsarDM{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T{}_raylos{}.npz".format(resolution, args.resolution, args.scenario, args.key, args.snapnum, args.Tmin, args.Tmax, args.key, args.TRe_key, rayLength), data=save_array, column_names = ['x_pc', 'y_pc', 'DM'])
    np.savez("/scratch/jh2/hs9158/results/data/PulsarDMz{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T{}_raylos{}.npz".format(resolution, args.resolution, args.scenario, args.key, args.snapnum, args.Tmin, args.Tmax, args.key, args.TRe_key, rayLength), data=stack_DMz)
    plt.figure(figsize=(7,14));
    plt.subplot(211)
    plt.scatter(coords[:, 0], coords[:, 1], s=0.05, alpha=0.02, label='gas particles');
    plt.scatter(SF_points[:, 0], SF_points[:, 1], s=1, alpha=0.8, label='SlugCluster Age 5-10 Myr');
    plt.scatter(pulsar_points[:, 0], pulsar_points[:, 1], s=0.8, alpha=0.2, label='pulsar particles');
    plt.scatter(np.array(points)[:, 0], np.array(points)[:,1], s=np.abs(np.array(DM)), alpha=0.5)
    plt.xlim([90*1e3, 110*1e3]); plt.ylim([90*1e3, 110*1e3]);
    plt.xlabel('x (pc)'); plt.ylabel('y (pc)');
    plt.gca().set_aspect('equal'); plt.legend()
    plt.subplot(212)
    y, x = rm.histogram(np.array(DM), 20)
    plt.plot(x, y)
    plt.xlabel(r'$\rm DM~(pc/cm^3)$')
    plt.ylabel('Number of sources')
    plt.gca().set_aspect('equal');
    plt.suptitle('Tmin = {}, Tmax = {}, sigmaRM = {}'.format(args.Tmin, args.Tmax, np.array(DM).std()))
    plt.savefig(args.plotpath+'PulsarMap_Tmin_{}_Tmax_{}_{}_{}_T{}.png'.format(args.Tmin, args.Tmax, args.key, args.snapnum, args.TRe_key), dpi=300)
    # plt.show()
    # stop()
    return points, DM

def create_prop_rays(args, ad_ion_rot, extent, resolution, R, R_obs, prop_key, init_cond):
    # ad_ion_rot = rm.CorrectIons(args, ad_ion_rot, args.snapnum)
    ind_T = make_temperature_cut(args, ad_ion_rot, args.Tmin, args.Tmax)
    print("Creating column maps for snapshot:", args.snapnum)
    ptype = "PartType0"
    # check if the processed quantities already exist for this snapshot
    if os.path.exists(args.snappath+'snapshot_spectrum_{}_processed_quantities_T{}.npy'.format(str(args.snapnum).zfill(3), args.TRe_key)):
        print("Processed quantities already exist for this snapshot.")
        arr = np.load(args.snappath+'snapshot_spectrum_{}_processed_quantities_T{}.npy'.format(str(args.snapnum).zfill(3), args.TRe_key), allow_pickle=True)
        arrdata = arr.item()
        # stop()
        ne_corrected_photoionzed = arrdata['ne_corrected_photoionized']
        ne_corrected = arrdata['ne_corrected']
        T = arrdata['grackle_temperature_photo']
    else:
        print("Process the quantities for this snapshot.", error=True)
        ne_corrected_photoionzed = ad_ion_rot[ptype, 'ne_corrected_photoionized']
        ne_corrected = ad_ion_rot[ptype, 'ne_corrected']

    
    # extract all indices except the ind_T
    ind_not_T = np.isin(np.arange(len(ne_corrected_photoionzed)), ind_T, invert=True)
    # make the relevant cuts
    ad_ion_rot = rm.RotateGalaxy(args, ad_ion_rot, R, center=args.center)
    ad_ion_rot = rm.RotateGalaxy(args, ad_ion_rot, R_obs, center=args.center)  # rotate to observer's frame
    np.random.seed(0)
    pulsar_points = np.column_stack((np.random.uniform(low=-extent*1e3/2, high=extent*1e3/2, size=int(resolution)),
                        np.random.uniform(low=-extent*1e3/2, high=extent*1e3/2, size=int(resolution)))) # in pc
    coords = ad_ion_rot[ptype, 'Coordinates'].in_units('pc')
    coords = coords-(args.center*1e3)*unyt.pc
    sml = ad_ion_rot[ptype, 'SmoothingLength'].in_units('pc')
    rayLength = 50000  # in pc
    bins = 100000
    # stop()
    if prop_key=='ne': prop = ne_corrected_photoionzed
    if prop_key=='n': prop = ad_ion_rot[ptype, 'particle_number_density']
    if prop_key=='nH': prop = ad_ion_rot[ptype, 'hydrogen_number_density']
    BLOS = ad_ion_rot[ptype, 'MagneticField'][:,2]*1e6 # in uG
    prop[ind_not_T] = 0
    ind_LOS = 0
    if init_cond=='data':
        args_list = [(point) for point in pulsar_points]
        # rm.rayInterpolate(points[0], coords, rayLength, bins, sml, ne, BLOS)
        # Create a Pool object
        with Pool() as pool:
            results = pool.map(partial(rm.PropRayInterpolate, 
                                    coordinate=coords,
                                    rayLength=rayLength,
                                    bins=bins, 
                                    sml=sml,
                                    prop=prop), args_list)
        points, nH, pointz, nHz = zip(*results)
        save_array = aa = np.column_stack((points, nH))
        # stack all arrays in RMz tuple
        np.savez("/scratch/jh2/hs9158/results/data/Paper_nH.npz", data=save_array, column_names = ['x_pc', 'y_pc', 'nH'])
        return points, nH
            

    # stop()
    # rm.rayInterpolate(pulsar_points[ind_LOS][0:2], coords, rayLength, bins, sml, prop, BLOS, 'DM')
    if init_cond=='all':
        for ind_LOS in range(len(pulsar_points)):
            print("ind_LOS: ", ind_LOS)
            z, propz = rm.PropRayInterpolate(pulsar_points[ind_LOS], coords, rayLength, bins, sml, prop)
            plt.figure(figsize=(7,14));
            plt.subplot(211)
            plt.scatter(coords[:, 0], coords[:, 2], s=0.05, alpha=0.01, label='gas particles');
            plt.vlines(pulsar_points[ind_LOS][0], z[:,2].min(), z[:,2].max(), color='r', label='Pulsar LOS')
            plt.xlim([pulsar_points[ind_LOS][0]-5e4, pulsar_points[ind_LOS][0]+5e4]); plt.ylim([250e3, 300e3]);
            plt.legend()
            plt.xlabel('x (pc)'); plt.ylabel('z (pc)');
            plt.subplot(212)
            plt.plot(z[:,2], propz)
            plt.yscale('log')
            plt.xlabel('z (pc)'); plt.ylabel(prop_key)
            plt.suptitle('{}'.format(np.round(pulsar_points[ind_LOS], 2)))
            plt.savefig(args.plotpath+'PulsarLOS_{}_{}_{}_T{}.png'.format(prop_key, args.snapnum, ind_LOS, args.TRe_key), dpi=300)
    else:
        z, propz = rm.PropRayInterpolate(pulsar_points[ind_LOS], coords, rayLength, bins, sml, prop)
        plt.figure(figsize=(7,14));
        plt.subplot(211)
        plt.scatter(coords[:, 0], coords[:, 2], s=0.05, alpha=0.01, label='gas particles');
        plt.vlines(pulsar_points[ind_LOS][0], z[:,2].min(), z[:,2].max(), color='r', label='Pulsar LOS')
        plt.xlim([pulsar_points[ind_LOS][0]-5e4, pulsar_points[ind_LOS][0]+5e4]); plt.ylim([250e3, 300e3]);
        plt.legend()
        plt.xlabel('x (pc)'); plt.ylabel('z (pc)');
        plt.subplot(212)
        plt.plot(z[:,2], propz)
        plt.yscale('log')
        plt.xlabel('z (pc)'); plt.ylabel(prop_key)
        plt.suptitle('{}'.format(np.round(pulsar_points[ind_LOS], 2)))
        plt.savefig(args.plotpath+'PulsarLOS_{}_{}_{}_T{}.png'.format(prop_key, args.snapnum, ind_LOS, args.TRe_key), dpi=300)
    return None
    # plt.show()
    # stop()

def read_RM_map(args, snapnum, resolution, RMDM, method): # method could be old or new depending on the file name
    "This function reads the RM map for a given snapshot and resolution"
    "from the data folder"
    if method=='old':
        data = np.load("/scratch/jh2/hs9158/results/data/{}{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T{}.npz".format(RMDM, resolution, args.resolution, args.scenario, args.key, snapnum, args.Tmin, args.Tmax, args.key, args.TRe_key))
    if method=='new':
        data = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}.npz".format(RMDM, resolution, snapnum, args.Tmin, args.Tmax, args.TRe_key))
        print("Reading RM map "+ args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}.npz".format(RMDM, resolution, snapnum, args.Tmin, args.Tmax, args.TRe_key))
    points = data['data'][:, 0:2]
    RM = data['data'][:, 2]
    return points, RM

def calculate_DM_CGM(n_0, beta, rc):
    "This function calculates the DM from the CGM between sun and LMC"
    "using the formula from the paper salem et al. 2015"
    def n(r): # r in kpc
        return n_0*rc**(3*beta)*r**(-3*beta)
    
    def r(r_LS): # r_LS in kpc
        return (72.25+13.26*r_LS+2453.3*r_LS**2)**(1/2)
    
    def DM(r_LS): # make sure r_LS is in kpc
        return n(r(r_LS))*1e3
    
    # integrate DM from r_LS = 0 to r_LS = 50.122 kpc
    DM_cgm = integrate.quad(DM, 0, 50.122)
    stop()
    return DM_cgm


def process_temperatures(args, i, save=True):
    ds = yt.load(args.snappath+'snapshot_{}.hdf5'.format(str(i).zfill(3)))
    # ds = yt.load(args.snappath+'snapdir_{}/snapshot_{}.0.hdf5'.format(str(i).zfill(3), str(i).zfill(3)))
    ds = rm.CorrectIons(args, ds, i)
    ad = ds.all_data()
    # stop()
    print("Calculating temperature for snapshot: ", i)
    T = cr.calculate_temperature(density = ad["PartType0", "particle_number_density"].in_units('1/cm**3'),
                                        temperature = ad["PartType0", "Temperature"].in_units('K'),
                                        internal_energy = ad["PartType0", "InternalEnergy"].in_units('km**2/s**2'))
    if save: np.save(args.snappath+'snapshot_{}_temperature.npy'.format(str(i).zfill(3)), T)
    print("Temperature calculated for snapshot: ", i)
    # sleep for 1000 seconds
    ds.close(); del T
    time.sleep(100); print("Sleeping for 100 seconds...")
    return print("Temperature calculated for snapshot: ", i)

def RM_frac(args, i, save=True):
    points, RM = read_RM_map(args, i, args.resolution, 'RM')
    rm.get_plane_rotation(args, args.rotation_matrix)
    
    return RM_frac

def main_temperature(args, args_list_num_start, args_list_num_end, workers):
    args_list = []
    print("printing the arguments for args_list...")
    for i in range(args_list_num_start, args_list_num_end, 1):
        print(i)
        args_list.append({'args': args,'i':i,'save':True})

    max_workers = workers
    with cofu.ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(
                process_temperatures,
                **args_num
            ) for args_num in args_list
        ]
        results = cofu.wait(futures)
        executor.shutdown(wait=True)

        # Wait for all futures to complete
        done, not_done = cofu.wait(futures, return_when=cofu.ALL_COMPLETED)

        # Check results and handle exceptions
        ## check if any tasks failed
        for future in cofu.as_completed(futures):
            future.result()


def main(args, args_list_num_start, args_list_num_end, workers, plane, rotation_matrix, zoom, slice):
    args_list = []
    print("printing the arguments for args_list...")
    for i in range(args_list_num_start, args_list_num_end, 1):
        print(i)
        args_list.append({'args': args,
                          'i':i,
                          'plane':plane,
                          'rotation_matrix': rotation_matrix,
                          'simtype': args.simtype,
                          'zoom': zoom})
    max_workers = workers
    with cofu.ProcessPoolExecutor(max_workers=workers) as executor:
        # Map the function over the executor
        ## loop over all simulation folders
        if slice:
            futures = [
                executor.submit(
                pl.subplot_slice_saver_wrapper,
                args_num
                ) for args_num in args_list]
            ## wait to ensure that all scheduled and running tasks have completed
            results = cofu.wait(futures)
            print(results)

            ## check if any tasks failed
            for future in cofu.as_completed(futures):
                future.result()
        else:
            futures = [
                executor.submit(
                pl.subplot_projection_saver_wrapper,
                args_num
                ) for args_num in args_list]
            ## wait to ensure that all scheduled and running tasks have completed
            results = cofu.wait(futures)
            print(results)

            ## check if any tasks failed
            for future in cofu.as_completed(futures):
                future.result()

# parallelize the call to rm.CorrectneCloudySlug
def main_ne(args, args_list_num_start, args_list_num_end, steps, workers, mode='create'):
    args_list = []
    print("printing the arguments for args_list...")
    # args, i, Tmin=1e1, Tmax=10**(4.3), cond='T', col=1e19, rho=1e-1, ne_corrected=True, mode='create'
    for n in range(args_list_num_start, args_list_num_end, steps):
        print(n)
        args_list.append({'args': args,
                          'i':n,
                          'Tmin':1e1,
                          'Tmax': args.TRe,
                          'cond': 'T',
                          'col': 1e19,
                          'rho': 1e-1,
                          'ne_corrected': True,
                          'mode':mode})

    max_workers = workers
    # stop()
    with cofu.ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(
                rm.CorrectneCloudySlug,
                **args_num
            ) for args_num in args_list
        ]
        results = cofu.wait(futures)
        executor.shutdown(wait=True)


        # Wait for all futures to complete
        done, not_done = cofu.wait(futures, return_when=cofu.ALL_COMPLETED)

        # Check results and handle exceptions
        for future in done:
            try:
                result = future.result()  # This will raise any exception caught during execution
                print(result)
            except Exception as e:
                print(f"An error occurred: {e}")

    print("All tasks completed.")

def main_plot(args, args_list_num_start, args_list_num_end, workers, plane, rotated, slice):
    args_list = []
    print("printing the arguments for args_list...")
    for i in range(args_list_num_start, args_list_num_end, 1):
        print(i)
        args_list.append({'args': args,
                          'i':i,
                          'plane':plane,
                          'rotated':rotated,
                          'slice': slice})

    max_workers = workers
    with cofu.ProcessPoolExecutor(max_workers=workers) as executor:
        # Map the function over the executor
        ## loop over all simulation folders
        futures = [
            executor.submit(
            pl.subplot_movie_plotter,
            **args_num
            ) for args_num in args_list]
        ## wait to ensure that all scheduled and running tasks have completed
        results = cofu.wait(futures)
        print(results)

        ## check if any tasks failed
        for future in cofu.as_completed(futures):
            future.result()

def main_particle_plot(args, args_list_num_start, args_list_num_end, workers):
    args_list = []
    print("printing the arguments for args_list...")
    for i in range(args_list_num_start, args_list_num_end, 1):
        print(i)
        args_list.append({'args': args,
                          'i':i})

    max_workers = workers
    with cofu.ProcessPoolExecutor(max_workers=workers) as executor:
        # Map the function over the executor
        ## loop over all simulation folders
        futures = [
            executor.submit(
            pl.make_particle_plot,
            **args_num
            ) for args_num in args_list]
        ## wait to ensure that all scheduled and running tasks have completed
        results = cofu.wait(futures)
        print(results)

        ## check if any tasks failed
        for future in cofu.as_completed(futures):
            future.result()


def sigmoid(x):
    return 1/(1+np.exp(-x/1000))

def combine_hdf5(args, i):
    # Write a function to combine the hdf5 files for a given snapshot
    # the hdf5 files contain the same fields but for different particles
    # the function should combine the fields for all particles and save them in a single hdf5 file
    # the function should also combine the fields for all particles in the same order
    print("Combining hdf5 files for snapshot: ", i)
    # open the hdf5 files
    F = h5py.File(args.snappath+'snapdir_{}/snapshot_{}.0.hdf5'.format(str(i).zfill(3), str(i).zfill(3)),"r")
    F1 = h5py.File(args.snappath+'snapdir_{}/snapshot_{}.1.hdf5'.format(str(i).zfill(3), str(i).zfill(3)),"r")
    F2 = h5py.File(args.snappath+'snapdir_{}/snapshot_{}.2.hdf5'.format(str(i).zfill(3), str(i).zfill(3)),"r")
    F3 = h5py.File(args.snappath+'snapdir_{}/snapshot_{}.3.hdf5'.format(str(i).zfill(3), str(i).zfill(3)),"r")
    F4 = h5py.File(args.snappath+'snapdir_{}/snapshot_{}.4.hdf5'.format(str(i).zfill(3), str(i).zfill(3)),"r")
    F5 = h5py.File(args.snappath+'snapdir_{}/snapshot_{}.5.hdf5'.format(str(i).zfill(3), str(i).zfill(3)),"r")
    F6 = h5py.File(args.snappath+'snapdir_{}/snapshot_{}.6.hdf5'.format(str(i).zfill(3), str(i).zfill(3)),"r")
    F7 = h5py.File(args.snappath+'snapdir_{}/snapshot_{}.7.hdf5'.format(str(i).zfill(3), str(i).zfill(3)),"r")

    # create a new hdf5 file to save the combined fields
    F_combined = h5py.File(args.snappath+'snapshot_{}.hdf5'.format(str(i).zfill(3)),"w")

    # loop over all the fields in the hdf5 files and combine them
    for field in F.keys():
        print(field)
        # combine the fields for different PartTypes
        F_combined.create_group(field)
        for field1 in F[field].keys():
            print(field1)
            # stop()
            F_combined[field].create_dataset(field1, data=np.concatenate((F[field][field1][:], F1[field][field1][:], F2[field][field1][:], F3[field][field1][:], F4[field][field1][:], F5[field][field1][:], F6[field][field1][:], F7[field][field1][:])))

    # Copy the attributes of the header from the first hdf5 file
    for attr in F["Header"].attrs.keys():
        F_combined["Header"].attrs[attr] = F["Header"].attrs[attr]
    # change the attribute 'NumPart_ThisFile' in the header to the total number of particles in the combined hdf5 file and change "NumFilesPerSnapshot" to 1
    F_combined["Header"].attrs["NumPart_ThisFile"] = F["Header"].attrs["NumPart_Total"]
    F_combined["Header"].attrs["NumFilesPerSnapshot"] = 1

    # close the hdf5 files
    F.close(); F1.close(); F2.close(); F3.close(); F4.close(); F5.close(); F6.close(); F7.close()
    F_combined.close()

def DM_LMC(args):
    DM_list = [65, 45, 94, 103, 100, 91, 97, 68, 119,
               136, 126, 103, 69, 124, 94, 75, 89, 273, 146,
               97, 131, 73]
    DM_list_johnston = [16, 44, 51, 19, 62, 69, 45, 11, 35, 84, 71, 6]
    DM_gal = 25  # should be either 25 or 25/sin|b| depending on manchester et al or ridley et al
    DM_LMC_corrected = np.array(DM_list)-DM_gal
    DM_LMC_corrected_johnston = np.array(DM_list_johnston)
    DM_LMC_overall = np.concatenate((DM_LMC_corrected, DM_LMC_corrected_johnston))
    # make a CDF of the DM_LMC_corrected
    DM_LMC_corrected_sorted = np.sort(DM_LMC_overall)
    y = np.cumsum(DM_LMC_corrected_sorted)
    y_norm = y/np.max(y)
    # plt.plot(DM_LMC_corrected_sorted, y_norm); plt.show()
    DM_T37 = np.load("/scratch/jh2/hs9158/results/data/{}{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T37_raylos50000.npz".format('PulsarDM', 200, args.resolution, args.scenario, args.key, 100, 10.0, 10000000.0, args.key))
    DM_T39 = np.load("/scratch/jh2/hs9158/results/data/{}{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T39_raylos50000.npz".format('PulsarDM', 200, args.resolution, args.scenario, args.key, 100, 10.0, 10000000.0, args.key))
    DM_T40 = np.load("/scratch/jh2/hs9158/results/data/{}{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T40_raylos50000.npz".format('PulsarDM', 200, args.resolution, args.scenario, args.key, 100, 10.0, 10000000.0, args.key))
    DM_T41 = np.load("/scratch/jh2/hs9158/results/data/{}{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T41_raylos50000.npz".format('PulsarDM', 200, args.resolution, args.scenario, args.key, 100, 10.0, 10000000.0, args.key))
    DM_T42 = np.load("/scratch/jh2/hs9158/results/data/{}{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T42_raylos50000.npz".format('PulsarDM', 200, args.resolution, args.scenario, args.key, 100, 10.0, 10000000.0, args.key))
    # stop()
    # DM_T39 = np.load("/scratch/jh2/hs9158/results/data/{}{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T39.npz".format('DM', 50, args.resolution, args.scenario, args.key, 100, 10.0, 10000000.0, args.key))
    DM_T37 = DM_T37['data'][:, 3]
    DM_T39 = DM_T39['data'][:, 3]
    DM_T40 = DM_T40['data'][:, 3]
    DM_T41 = DM_T41['data'][:, 3]
    DM_T42 = DM_T42['data'][:, 3]
    # DM_T39 = DM_T39['data'][:, 2]
    # create CDFs of the DM_T37 and DM_T39
    DM_T37_sorted = np.sort(DM_T37)
    DM_T39_sorted = np.sort(DM_T39)
    DM_T40_sorted = np.sort(DM_T40)
    DM_T41_sorted = np.sort(DM_T41)
    DM_T42_sorted = np.sort(DM_T42)
    # DM_T39_sorted = np.sort(DM_T39)
    y_T37 = np.cumsum(DM_T37_sorted)
    y_T39 = np.cumsum(DM_T39_sorted)
    y_T40 = np.cumsum(DM_T40_sorted)
    y_T41 = np.cumsum(DM_T41_sorted)
    y_T42 = np.cumsum(DM_T42_sorted)

    y_T37_norm = y_T37/np.max(y_T37)
    y_T39_norm = y_T39/np.max(y_T39)
    y_T40_norm = y_T40/np.max(y_T40)
    y_T41_norm = y_T41/np.max(y_T41)
    y_T42_norm = y_T42/np.max(y_T42)

    scale = 1
    plt.figure(figsize=(7*scale, 5*scale))
    plt.plot(DM_T37_sorted, y_T37_norm, label=r'${\rm Mock~DM}~(T_{\rm Re} {\rm = 5000~ K})$', linewidth=3)
    plt.plot(DM_T39_sorted, y_T39_norm, label=r'${\rm Mock~DM}~(T_{\rm Re} {\rm = 8000~ K})$', linewidth=3)
    plt.plot(DM_T40_sorted, y_T40_norm, label=r'${\rm Mock~DM}~(T_{\rm Re} {\rm = 10000~ K})$', linewidth=3)
    plt.plot(DM_T41_sorted, y_T41_norm, label=r'${\rm Mock~DM}~(T_{\rm Re} {\rm = 12500~ K})$', linewidth=3)
    plt.plot(DM_T42_sorted, y_T42_norm, label=r'${\rm Mock~DM}~(T_{\rm Re} {\rm = 16000~ K})$', linewidth=3)
    plt.plot(DM_LMC_corrected_sorted, y_norm, label='Observed MW-corrected DM', linewidth=3)
    plt.xlabel(r'$\rm DM~(pc/cm^3)$')
    plt.ylabel('CDF')
    plt.legend(loc='best')
    plt.xlim([0, 250])
    plt.ylim([0, 1])
    plt.show()
    plt.savefig("/scratch/jh2/hs9158/results/paper_plots/DM_LMC.pdf", dpi=100)
    stop()

def process_everything(args, dirty_args, args_list_num_start, args_list_num_end, workers, rotation_matrix, zoom, slice, mode):
    # This function processes everything for a given snapshot
    print("Starting parallel processing to get the temperatures.")
    # dp.dirty_main(args, dirty_args, dirty_args.key1)
    # memory requirements for temperature calc
    # medres, scenario3 - 30 GB per snapshot
    if args.resolution == 'medres':
        # main_temperature(args, args_list_num_start, args_list_num_end+1, workers)
        # print("Finished parallel processing to get the temperatures.")
        # # main_temperature(args, args_list_num_start, args_list_num_end+1, workers)
        # main(args, args_list_num_start, args_list_num_end+1, workers, 'z', rotation_matrix, True, slice)
        # main_plot(args, args_list_num_start, args_list_num_end+1, workers*5, 'z', True, slice)
        # main(args, args_list_num_start, args_list_num_end+1, workers, 'y', rotation_matrix, True, slice)
        # main_plot(args, args_list_num_start, args_list_num_end+1, workers*5, 'y', True, slice)
        # main(args, args_list_num_start, args_list_num_end+1, workers, 'z', rotation_matrix, False, slice)
        # main(args, args_list_num_start, args_list_num_end+1, workers, 'y', rotation_matrix, False, slice)
        
        main_ne(args, 91, 101, 1, 10, mode)
    if args.resolution == 'medhighres':
        main_temperature(args, args_list_num_start, args_list_num_end+1, 5)
        # print("Finished parallel processing to get the temperatures.")
        # main_temperature(args, args_list_num_start, args_list_num_end+1, workers)
        # main(args, args_list_num_start, args_list_num_end+1, 5, 'z', rotation_matrix, True, slice)
        # main_plot(args, args_list_num_start, args_list_num_end+1, 16, 'z', True, slice)
        # main(args, args_list_num_start, args_list_num_end+1, 6, 'y', rotation_matrix, True, slice)
        # main_plot(args, args_list_num_start, args_list_num_end+1, 16, 'y', True, slice)
        # main(args, args_list_num_start, args_list_num_end+1, workers, 'z', rotation_matrix, False, slice)
        # main(args, args_list_num_start, args_list_num_end+1, workers, 'y', rotation_matrix, False, slice)
        
        main_ne(args, 100, 101, 1, 1, mode)
    stop()

def clean_process_everything(args, args_list_num_start, args_list_num_end, workers, rotation_matrix, zoom, slice, mode):
    main_temperature(args, args_list_num_start, args_list_num_end+1, workers)
    stop()

def make_simulation_grid(args, i, R, plane_extent, los_extent, resolution):
    particletype = 'PartType0'
    # check if the processed quantities already exist for this snapshot
    if os.path.exists(args.snappath+'snapshot_{}_processed_quantities_T{}.npy'.format(str(args.snapnum).zfill(3), args.TRe_key)):
        print("Processed quantities already exist for this snapshot.")
        arr = np.load(args.snappath+'snapshot_{}_processed_quantities_T{}.npy'.format(str(args.snapnum).zfill(3), args.TRe_key), allow_pickle=True)
        arrdata = arr.item()
        # stop()
        ne_corrected_photoionzed = arrdata['ne_corrected_photoionized']
        ne_corrected = arrdata['ne_corrected']
        T = arrdata['grackle_temperature_photo']
        molecular_weight = arrdata['molecular_weight']*constants.m_p
    else:
        print("Process the quantities for this snapshot.", error=True)
    # stop()
    ds = yt.load(args.snappath+'snapshot_{}.hdf5'.format(str(i).zfill(3)))
    ad = ds.all_data()
    # T = ad['PartType0', 'grackle_temperature']
    # stop()
    coord = np.array(ad["PartType0", "Coordinates"])
    coord = coord - args.center
    mass = np.array(ad["PartType0", "Masses"][:])
    B = np.array(ad["PartType0", "MagneticField"][:])*1e6
    if args.scenario==0:
        V = np.array(ad["PartType0", "Velocities"][:])
    else:
        V = np.array(ad["PartType0", "Velocities"][:]) - np.array([0, 0, np.linalg.norm(args.V_LMC)]) # in the frame of the center of the LMC if it is translating in the wind scenario
    Z = np.array(ad["PartType0", "Metallicity"][:])
    hsml = np.array(ad["PartType0", "SmoothingLength"][:])
    rho = np.array(ad["PartType0", "Density"][:].in_units('g/cm**3'))
    # rotate coord, B, V
    coord = np.dot(R, coord.T).T
    B = np.dot(R, B.T).T
    V = np.dot(R, V.T).T
    ind_proj = np.where((np.abs(coord[:, 0]) <= plane_extent/2+2) & (np.abs(coord[:, 1]) <= plane_extent/2+2) & (np.abs(coord[:, 2])<=plane_extent/2+2))[0]
    coord = coord[ind_proj]; B = B[ind_proj]; V = V[ind_proj]; Z = Z[ind_proj]; rho = rho[ind_proj]; hsml = hsml[ind_proj]; mass = mass[ind_proj]; T = T[ind_proj]
    print("Creating the grid for snapshot: ", i)
    X, Y, Z = np.meshgrid(np.linspace(-plane_extent/2, plane_extent/2, resolution), np.linspace(-plane_extent/2, plane_extent/2, resolution), np.linspace(-plane_extent/2, plane_extent/2, resolution), indexing='ij')
    master_data_list = []
    T_grid = meshoid.WeightedGridInterp3D(T, np.ones(len(T)), coord, hsml, np.array([0, 0, 0]), plane_extent, res=resolution)
    master_data_list.append(T_grid)
    np.savez(args.snappath+'Gridres{}_snap{}_{}_scen{}_T_grid.npz'.format(resolution, str(i).zfill(3), args.resolution, args.scenario), X=X, Y=Y, Z=Z, T_grid=T_grid)
    del T_grid, T
    print("T grid created and deleted.")
    Density_grid = meshoid.WeightedGridInterp3D(rho, mass, coord, hsml, np.array([0, 0, 0]), plane_extent, res=resolution)
    master_data_list.append(Density_grid)
    np.savez(args.snappath+'Gridres{}_snap{}_{}_scen{}_Density_grid.npz'.format(resolution, str(i).zfill(3), args.resolution, args.scenario), Density_grid=Density_grid)
    del Density_grid, rho
    print("Density grid created and deleted.")
    BX_grid = meshoid.WeightedGridInterp3D(B[:, 0], np.ones(len(B[:, 0])), coord, hsml, np.array([0, 0, 0]), plane_extent, res=resolution)
    master_data_list.append(BX_grid)
    np.savez(args.snappath+'Gridres{}_snap{}_{}_scen{}_BX_grid.npz'.format(resolution, str(i).zfill(3), args.resolution, args.scenario), BX_grid=BX_grid)
    del BX_grid
    BY_grid = meshoid.WeightedGridInterp3D(B[:, 1], np.ones(len(B[:, 1])), coord, hsml, np.array([0, 0, 0]), plane_extent, res=resolution)
    master_data_list.append(BY_grid)
    np.savez(args.snappath+'Gridres{}_snap{}_{}_scen{}_BY_grid.npz'.format(resolution, str(i).zfill(3), args.resolution, args.scenario), BY_grid=BY_grid)
    del BY_grid
    BZ_grid = meshoid.WeightedGridInterp3D(B[:, 2], np.ones(len(B[:, 2])), coord, hsml, np.array([0, 0, 0]), plane_extent, res=resolution)
    master_data_list.append(BZ_grid)
    np.savez(args.snappath+'Gridres{}_snap{}_{}_scen{}_BZ_grid.npz'.format(resolution, str(i).zfill(3), args.resolution, args.scenario), BZ_grid=BZ_grid)
    del BZ_grid, B
    print("Magnetic field grid created and deleted.")
    Vx_grid = meshoid.WeightedGridInterp3D(V[:, 0], np.ones(len(V[:, 0])), coord, hsml, np.array([0, 0, 0]), plane_extent, res=resolution)
    master_data_list.append(Vx_grid)
    np.savez(args.snappath+'Gridres{}_snap{}_{}_scen{}_Vx_grid.npz'.format(resolution, str(i).zfill(3), args.resolution, args.scenario), Vx_grid=Vx_grid)
    del Vx_grid
    Vy_grid = meshoid.WeightedGridInterp3D(V[:, 1], np.ones(len(V[:, 1])), coord, hsml, np.array([0, 0, 0]), plane_extent, res=resolution)
    master_data_list.append(Vy_grid)
    np.savez(args.snappath+'Gridres{}_snap{}_{}_scen{}_Vy_grid.npz'.format(resolution, str(i).zfill(3), args.resolution, args.scenario), Vy_grid=Vy_grid)
    del Vy_grid
    Vz_grid = meshoid.WeightedGridInterp3D(V[:, 2], np.ones(len(V[:, 2])), coord, hsml, np.array([0, 0, 0]), plane_extent, res=resolution)
    master_data_list.append(Vz_grid)
    np.savez(args.snappath+'Gridres{}_snap{}_{}_scen{}_Vz_grid.npz'.format(resolution, str(i).zfill(3), args.resolution, args.scenario), Vz_grid=Vz_grid)
    del Vz_grid, V
    print("Velocity grid created and deleted.")   
    # plt.imshow(Density_grid[:, :, 50], origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()]); plt.show()
    # save the X, Y, Z, Density_grid, Z_grid, BX_grid, BY_grid, BZ_grid, Vx_grid, Vy_grid, Vz_grid in a numpy zip array
    print("Saving the overall grid for snapshot: ", i)
    np.savez(args.snappath+'Gridres{}_snap{}_{}_scen{}.npz'.format(resolution, str(i).zfill(3), args.resolution, args.scenario), X=X, Y=Y, Z=Z, T_grid=master_data_list[0], Density_grid=master_data_list[1], BX_grid=master_data_list[2], BY_grid=master_data_list[3], BZ_grid=master_data_list[4], Vx_grid=master_data_list[5], Vy_grid=master_data_list[6], Vz_grid=master_data_list[7])
    # delete all the previous zip arrays created as they are not needed anymore
    print("Deleting the individual grid files for snapshot: ", i)
    # stop()
    os.remove(args.snappath+'Gridres{}_snap{}_{}_scen{}_Vy_grid.npz'.format(resolution, str(i).zfill(3), args.resolution, args.scenario))
    os.remove(args.snappath+'Gridres{}_snap{}_{}_scen{}_Vz_grid.npz'.format(resolution, str(i).zfill(3), args.resolution, args.scenario))
    os.remove(args.snappath+'Gridres{}_snap{}_{}_scen{}_Vx_grid.npz'.format(resolution, str(i).zfill(3), args.resolution, args.scenario))
    os.remove(args.snappath+'Gridres{}_snap{}_{}_scen{}_BZ_grid.npz'.format(resolution, str(i).zfill(3), args.resolution, args.scenario))
    os.remove(args.snappath+'Gridres{}_snap{}_{}_scen{}_BY_grid.npz'.format(resolution, str(i).zfill(3), args.resolution, args.scenario))
    os.remove(args.snappath+'Gridres{}_snap{}_{}_scen{}_BX_grid.npz'.format(resolution, str(i).zfill(3), args.resolution, args.scenario))
    os.remove(args.snappath+'Gridres{}_snap{}_{}_scen{}_Density_grid.npz'.format(resolution, str(i).zfill(3), args.resolution, args.scenario))
    os.remove(args.snappath+'Gridres{}_snap{}_{}_scen{}_T_grid.npz'.format(resolution, str(i).zfill(3), args.resolution, args.scenario))
    print("Overall grid saved for snapshot: ", i)

    stop()

def read_Grid(gridres, snapnum, resolution, scenario):
    data = np.load(args.snappath+"Gridres{}_snap{}_{}_scen{}.npz".format(gridres, str(snapnum).zfill(3), resolution, scenario))
    X = data['X']; Y = data['Y']; Z = data['Z']
    Density_grid = data['Density_grid']
    T_grid = data['T_grid']
    BX_grid = data['BX_grid']
    BY_grid = data['BY_grid']
    BZ_grid = data['BZ_grid']
    Vx_grid = data['Vx_grid']
    Vy_grid = data['Vy_grid']
    Vz_grid = data['Vz_grid']
    stop()
    # # Visualization
    # plt.pcolormesh(X[:,:,int(gridres/2)], Y[:,:,int(gridres/2)], Density_grid[:,:,int(gridres/2)], shading='auto'); plt.show() # face-on
    # plt.pcolormesh(X[:,int(gridres/2),:], Z[:,int(gridres/2),:], Density_grid[:,int(gridres/2),:], shading='auto'); plt.show() # edge-on
    return X, Y, Z, Density_grid, T_grid, BX_grid, BY_grid, BZ_grid, Vx_grid, Vy_grid, Vz_grid

def RM_temperature_ion_analysis(args, key='plot'): # key could be 'plot' or 'table'
    args.Tmin = 1e1; args.Tmax = 1e7
    EM = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}.npz".format('EM', 200, 100, args.Tmin, args.Tmax, args.TRe_key))
    EM = EM['data'][:, 2]
    points, RM_tot = read_RM_map(args, 100, 200, 'RM', 'new')
    args.Tmin = 1e1; args.Tmax = 5e3
    points, RM_CNM = read_RM_map(args, 100, 200, 'RM', 'new')
    args.Tmin = 5e3; args.Tmax = 8e3
    points, RM_UNM = read_RM_map(args, 100, 200, 'RM', 'new')
    args.Tmin = 8e3; args.Tmax = 2e4
    points, RM_WIM = read_RM_map(args, 100, 200, 'RM', 'new')
    args.Tmin = 2e4; args.Tmax = 1e7
    points, RM_HIM = read_RM_map(args, 100, 200, 'RM', 'new')
    args.Tmin = 1e1; args.Tmax = 1e7
    points2, RM200 = read_RM_map(args, 100, 200, 'RM', 'new')
    imin = 0; imax = 1
    args.Tmin = 1e1; args.Tmax = 1e7
    data = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}_i{}_{}.npz".format('RM', 200, 100, args.Tmin, args.Tmax, args.TRe_key, imin, imax))
    points2 = data['data'][:, 0:2]
    RM_i01 = data['data'][:, 2]
    imin = 1; imax = 10
    data = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}_i{}_{}.npz".format('RM', 200, 100, args.Tmin, args.Tmax, args.TRe_key, imin, imax))
    RM_i110 = data['data'][:, 2]
    imin = 10; imax = 90
    data = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}_i{}_{}.npz".format('RM', 200, 100, args.Tmin, args.Tmax, args.TRe_key, imin, imax))
    RM_i1090 = data['data'][:, 2]
    imin = 90; imax = 120
    data = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}_i{}_{}.npz".format('RM', 200, 100, args.Tmin, args.Tmax, args.TRe_key, imin, imax))
    RM_i90100 = data['data'][:, 2]
    # reading the density bin data
    data = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}_nH{}_{}.npz".format('RM', 200, 100, args.Tmin, args.Tmax, args.TRe_key, 1e-16, 0.1))
    RM_nH_level1 = data['data'][:, 2]
    data = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}_nH{}_{}.npz".format('RM', 200, 100, args.Tmin, args.Tmax, args.TRe_key, 0.1, 1.0))
    RM_nH_level2 = data['data'][:, 2]
    data = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}_nH{}_{}.npz".format('RM', 200, 100, args.Tmin, args.Tmax, args.TRe_key, 1.0, 10.0))
    RM_nH_level3 = data['data'][:, 2]
    data = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}_nH{}_{}.npz".format('RM', 200, 100, args.Tmin, args.Tmax, args.TRe_key, 10.0, 1e16))
    RM_nH_level4 = data['data'][:, 2]
    # stop()
    ind = np.where(np.abs(EM)<100)[0]
    ind2 = np.where(np.abs(EM)<100)[0]
    # stop()
    points = points[ind]; RM_tot = RM_tot[ind]; RM_CNM = RM_CNM[ind]; RM_UNM = RM_UNM[ind]; RM_WIM = RM_WIM[ind]
    points2 = points2[ind2]; RM200 = RM200[ind2]; RM_i01 = RM_i01[ind2]; RM_i110 = RM_i110[ind2]; RM_i1090 = RM_i1090[ind2]; RM_i90100 = RM_i90100[ind2]
    scale = 1
    if key=='plot':
        fig = plt.figure(figsize=(18*scale, 9.5*scale), constrained_layout=False)
        gs = fig.add_gridspec(2, 5)
        gs.update(wspace=0, hspace=0)

        # Create subplots using the gridspec
        axs = []
        for i in range(2):
            for j in range(5):
                axs.append(fig.add_subplot(gs[i, j]))
        
        RM_scale = 30
        color = np.repeat('r', len(RM_tot))
        color[np.where(RM_tot>0)[0]] = 'b'
        axs[0].scatter(points[:, 0], points[:, 1], s=np.sqrt(np.abs(RM_tot)*RM_scale), c=color, alpha=0.5)
        axs[0].set_xlabel('X (pc)'); axs[0].set_ylabel('Y (pc)'); axs[0].set_title('Total RM')
        axs[0].set_aspect('equal', adjustable='box')

        color = np.repeat('r', len(RM_CNM))
        color[np.where(RM_CNM>0)[0]] = 'b'
        axs[1].scatter(points[:, 0], points[:, 1], s=np.sqrt(np.abs(RM_CNM)*RM_scale), c=color, alpha=0.5)
        axs[1].set_xlabel('X (pc)');axs[1].set_title(r'$T$(K)=$10-5\times 10^3$')
        axs[1].set_aspect('equal', adjustable='box')

        color = np.repeat('r', len(RM_UNM))
        color[np.where(RM_UNM>0)[0]] = 'b'
        axs[2].scatter(points[:, 0], points[:, 1], s=np.sqrt(np.abs(RM_UNM)*RM_scale), c=color, alpha=0.5)
        axs[2].set_xlabel('X (pc)'); axs[2].set_title(r'$T$(K)=5$\times 10^3-8\times 10^3$')
        axs[2].set_aspect('equal', adjustable='box')

        color = np.repeat('r', len(RM_WIM))
        color[np.where(RM_WIM>0)[0]] = 'b'
        axs[3].scatter(points[:, 0], points[:, 1], s=np.sqrt(np.abs(RM_WIM)*RM_scale), c=color, alpha=0.5)
        axs[3].set_xlabel('X (pc)'); axs[3].set_title(r'$T$(K)=8$\times 10^3-2\times 10^4$')
        axs[3].set_aspect('equal', adjustable='box')

        color = np.repeat('r', len(RM_HIM))
        color[np.where(RM_HIM>0)[0]] = 'b'
        axs[4].scatter(points[:, 0], points[:, 1], s=np.sqrt(np.abs(RM_HIM)*RM_scale), c=color, alpha=0.5)
        axs[4].set_xlabel('X (pc)'); axs[4].set_title(r'$T$(K) $\geq 2\times 10^4$')
        axs[4].set_aspect('equal', adjustable='box')

        color = np.repeat('r', len(RM_tot))
        color[np.where(RM_tot>0)[0]] = 'b'
        axs[5].scatter(points[:, 0], points[:, 1], s=np.sqrt(np.abs(RM_tot)*RM_scale), c=color, alpha=0.5)
        axs[5].set_xlabel('X (pc)'); axs[5].set_title('Total RM')
        axs[5].set_aspect('equal', adjustable='box')

        color = np.repeat('r', len(RM_i01))
        color[np.where(RM_i01>0)[0]] = 'b'
        axs[6].scatter(points2[:, 0], points2[:, 1], s=np.sqrt(np.abs(RM_i01)*RM_scale), c=color, alpha=0.5)
        axs[6].set_xlabel('X (pc)'); axs[6].set_title(r'$\rm \mathit{x}_\mathrm{e}=0-0.01$')
        axs[6].set_aspect('equal', adjustable='box')

        color = np.repeat('r', len(RM_i110))
        color[np.where(RM_i110>0)[0]] = 'b'
        axs[7].scatter(points2[:, 0], points2[:, 1], s=np.sqrt(np.abs(RM_i110)*RM_scale), c=color, alpha=0.5)
        axs[7].set_xlabel('X (pc)'); axs[7].set_title(r'$\rm \mathit{x}_\mathrm{e}=0.01-0.1$')
        axs[7].set_aspect('equal', adjustable='box')

        color = np.repeat('r', len(RM_i1090))
        color[np.where(RM_i1090>0)[0]] = 'b'
        axs[8].scatter(points2[:, 0], points2[:, 1], s=np.sqrt(np.abs(RM_i1090)*RM_scale), c=color, alpha=0.5)
        axs[8].set_xlabel('X (pc)'); axs[8].set_title(r'$\rm \mathit{x}_\mathrm{e}=0.1-0.9$')
        axs[8].set_aspect('equal', adjustable='box')

        color = np.repeat('r', len(RM_i90100))
        color[np.where(RM_i90100>0)[0]] = 'b'
        axs[9].scatter(points2[:, 0], points2[:, 1], s=np.sqrt(np.abs(RM_i90100)*RM_scale), c=color, alpha=0.5)
        axs[9].set_xlabel('X (pc)');axs[9].set_title(r'$\rm \mathit{x}_\mathrm{e} \geq 0.9$')
        axs[9].set_aspect('equal', adjustable='box')

        # color = np.repeat('r', len(RM_nH_level1))
        # color[np.where(RM_nH_level1>0)[0]] = 'b'
        # axs[8].scatter(points2[:, 0], points2[:, 1], s=np.sqrt(np.abs(RM_nH_level1)*RM_scale), c=color, alpha=0.5)
        # axs[8].set_xlabel('X (pc)'); axs[8].set_ylabel('Y (pc)'); axs[8].set_title(r'$\rm RM~from~\mathit{n}_{\rm H}{\rm (cm^{-3})} \leq 10^{-1}$')
        # axs[8].set_aspect('equal', adjustable='box')

        # color = np.repeat('r', len(RM_nH_level2))
        # color[np.where(RM_nH_level2>0)[0]] = 'b'
        # axs[9].scatter(points2[:, 0], points2[:, 1], s=np.sqrt(np.abs(RM_nH_level2)*RM_scale), c=color, alpha=0.5)
        # axs[9].set_xlabel('X (pc)'); axs[9].set_title(r'$\rm RM~from~\mathit{n}_{\rm H}{\rm (cm^{-3})}=10^{-1}-1$')
        # axs[9].set_aspect('equal', adjustable='box')

        # color = np.repeat('r', len(RM_nH_level3))
        # color[np.where(RM_nH_level3>0)[0]] = 'b'
        # axs[10].scatter(points2[:, 0], points2[:, 1], s=np.sqrt(np.abs(RM_nH_level3)*RM_scale), c=color, alpha=0.5)
        # axs[10].set_xlabel('X (pc)'); axs[10].set_title(r'$\rm RM~from~\mathit{n}_{\rm H}{\rm (cm^{-3})}=1-10$')
        # axs[10].set_aspect('equal', adjustable='box')

        # color = np.repeat('r', len(RM_nH_level4))
        # color[np.where(RM_nH_level4>0)[0]] = 'b'
        # axs[11].scatter(points2[:, 0], points2[:, 1], s=np.sqrt(np.abs(RM_nH_level4)*RM_scale), c=color, alpha=0.5)
        # axs[11].set_xlabel('X (pc)'); axs[11].set_title(r'$\rm RM~from~\mathit{n}_{\rm H}{\rm (cm^{-3})} \geq 10$')
        # axs[11].set_aspect('equal', adjustable='box')
        # remove yticklabels for subplots 2, 3, 4, 6, 7, 8, 10, 11, 12
        for i in [1, 2, 3, 4, 6, 7, 8, 9]:
            axs[i].set_yticklabels([])
        # set both x and y limits to -4999 to 4999
        for i in range(10):
            axs[i].set_xlim(-4999, 4999)
            axs[i].set_ylim(-4999, 4999)


        plt.savefig("/scratch/jh2/hs9158/results/paper_plots/RM_temperature_ion_nH_analysis.pdf", dpi=100)
        plt.show()
    if key=='table':
        

        # ignore all indices where (np.abs(RM_CNM)+np.abs(RM_UNM)+np.abs(RM_WIM)) is zero
        ind = np.where((np.abs(RM_CNM)+np.abs(RM_UNM)+np.abs(RM_WIM)+np.abs(RM_HIM))>0)[0]
        fRM_T = np.zeros((len(ind), 5))
        fRM_T[:, 0] = np.abs(RM_tot[ind])/(np.abs(RM_CNM[ind])+np.abs(RM_UNM[ind])+np.abs(RM_WIM[ind])+np.abs(RM_HIM[ind]))
        fRM_T[:, 1] = np.abs(RM_CNM[ind])/(np.abs(RM_CNM[ind])+np.abs(RM_UNM[ind])+np.abs(RM_WIM[ind])+np.abs(RM_HIM[ind]))
        fRM_T[:, 2] = np.abs(RM_UNM[ind])/(np.abs(RM_CNM[ind])+np.abs(RM_UNM[ind])+np.abs(RM_WIM[ind])+np.abs(RM_HIM[ind]))
        fRM_T[:, 3] = np.abs(RM_WIM[ind])/(np.abs(RM_CNM[ind])+np.abs(RM_UNM[ind])+np.abs(RM_WIM[ind])+np.abs(RM_HIM[ind]))
        fRM_T[:, 4] = np.abs(RM_HIM[ind])/(np.abs(RM_CNM[ind])+np.abs(RM_UNM[ind])+np.abs(RM_WIM[ind])+np.abs(RM_HIM[ind]))
        # take a weighted average of fRM_T values with the absolute RM values as weights
        fRM_T_tot = np.zeros(5)
        fRM_T_tot[0] = np.sum(fRM_T[:, 0]*np.abs(RM_tot[ind]))/np.sum(np.abs(RM_tot[ind]))
        fRM_T_tot[1] = np.sum(fRM_T[:, 1]*np.abs(RM_tot[ind]))/np.sum(np.abs(RM_tot[ind]))
        fRM_T_tot[2] = np.sum(fRM_T[:, 2]*np.abs(RM_tot[ind]))/np.sum(np.abs(RM_tot[ind]))
        fRM_T_tot[3] = np.sum(fRM_T[:, 3]*np.abs(RM_tot[ind]))/np.sum(np.abs(RM_tot[ind]))
        fRM_T_tot[4] = np.sum(fRM_T[:, 4]*np.abs(RM_tot[ind]))/np.sum(np.abs(RM_tot[ind]))

        ind = np.where((np.abs(RM_i01)+np.abs(RM_i110)+np.abs(RM_i1090)+np.abs(RM_i90100))>0)[0]
        fRM_i = np.zeros((len(ind), 5))
        fRM_i[:, 0] = np.abs(RM_tot[ind])/(np.abs(RM_i01[ind])+np.abs(RM_i110[ind])+np.abs(RM_i1090[ind])+np.abs(RM_i90100[ind]))
        fRM_i[:, 1] = np.abs(RM_i01[ind])/(np.abs(RM_i01[ind])+np.abs(RM_i110[ind])+np.abs(RM_i1090[ind])+np.abs(RM_i90100[ind]))
        fRM_i[:, 2] = np.abs(RM_i110[ind])/(np.abs(RM_i01[ind])+np.abs(RM_i110[ind])+np.abs(RM_i1090[ind])+np.abs(RM_i90100[ind]))
        fRM_i[:, 3] = np.abs(RM_i1090[ind])/(np.abs(RM_i01[ind])+np.abs(RM_i110[ind])+np.abs(RM_i1090[ind])+np.abs(RM_i90100[ind]))
        fRM_i[:, 4] = np.abs(RM_i90100[ind])/(np.abs(RM_i01[ind])+np.abs(RM_i110[ind])+np.abs(RM_i1090[ind])+np.abs(RM_i90100[ind]))
        fRM_i_tot = np.zeros(5)
        fRM_i_tot[0] = np.sum(fRM_i[:, 0]*np.abs(RM_tot[ind]))/np.sum(np.abs(RM_tot[ind]))
        fRM_i_tot[1] = np.sum(fRM_i[:, 1]*np.abs(RM_tot[ind]))/np.sum(np.abs(RM_tot[ind]))
        fRM_i_tot[2] = np.sum(fRM_i[:, 2]*np.abs(RM_tot[ind]))/np.sum(np.abs(RM_tot[ind]))
        fRM_i_tot[3] = np.sum(fRM_i[:, 3]*np.abs(RM_tot[ind]))/np.sum(np.abs(RM_tot[ind]))
        fRM_i_tot[4] = np.sum(fRM_i[:, 4]*np.abs(RM_tot[ind]))/np.sum(np.abs(RM_tot[ind]))
        
        sign_RM_T = np.zeros(5)
        # percentage of signs same as total RM from each categories
        sign_RM_T[0] = len(np.where(np.sign(RM_tot)*np.sign(RM_tot)==1)[0])/len(RM_tot)
        sign_RM_T[1] = len(np.where(np.sign(RM_tot)*np.sign(RM_CNM)==1)[0])/len(RM_tot)
        sign_RM_T[2] = len(np.where(np.sign(RM_tot)*np.sign(RM_UNM)==1)[0])/len(RM_tot)
        sign_RM_T[3] = len(np.where(np.sign(RM_tot)*np.sign(RM_WIM)==1)[0])/len(RM_tot)
        sign_RM_T[4] = len(np.where(np.sign(RM_tot)*np.sign(RM_HIM)==1)[0])/len(RM_tot)
        
        sign_RM_i = np.zeros(5)
        sign_RM_i[0] = len(np.where(np.sign(RM_tot)*np.sign(RM_tot)==1)[0])/len(RM_tot)
        sign_RM_i[1] = len(np.where(np.sign(RM_tot)*np.sign(RM_i01)==1)[0])/len(RM_tot)
        sign_RM_i[2] = len(np.where(np.sign(RM_tot)*np.sign(RM_i110)==1)[0])/len(RM_tot)
        sign_RM_i[3] = len(np.where(np.sign(RM_tot)*np.sign(RM_i1090)==1)[0])/len(RM_tot)
        sign_RM_i[4] = len(np.where(np.sign(RM_tot)*np.sign(RM_i90100)==1)[0])/len(RM_tot)

        print(r"\begin{table*}")
        print(r"\centering")
        print(r"\caption{Mean RM contribution and same sign fractions by gas phase.}")
        print(r"\begin{tabular}{cccccc}")
        print(r"\hline\hline")
        print(r"\\[0.1ex]")
        print(r"Temperature (K) & All $T$ & $10\leq T<5000$ & $5000 \leq T<8000$ & $8000 \leq T<20000$ & $T \geq 20000$ \\")
        print(r"[0.1ex]\\")
        print(r"\hline")
        print(r"\\[0ex]")
        print(r"$\langle f_{{\mathrm{{RM}},k}} \rangle~\%$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ \\".format(np.round(fRM_T_tot[0]*100,2), np.round(fRM_T_tot[1]*100,2), np.round(fRM_T_tot[2]*100,2), np.round(fRM_T_tot[3]*100,2), np.round(fRM_T_tot[4]*100,2)))
        print(r"[0.7ex]")
        print(r"$f_{{\mathrm{{RM-same}},k}}$ \% & ${}$ & ${}$ & ${}$ & ${}$ & ${}$\\".format(np.round(sign_RM_T[0]*100,2), np.round(100*sign_RM_T[1],2), np.round(100*sign_RM_T[2],2), np.round(100*sign_RM_T[3],2), np.round(sign_RM_T[4]*100,2)))
        print(r"\\[-3ex]")
        print(r"\\ \hline\hline")
        print(r"\\[0.1ex]")
        print(r"$x_\mathrm{{e}}$ ($n_{{\rm e}}/n_{{\rm H}}$) & All $x_\mathrm{{e}}$ & $0\leq x_\mathrm{{e}}<0.01$ & $0.01\leq x_\mathrm{{e}}<0.1$ & $0.1 \leq x_\mathrm{e}<0.9$ & $x_\mathrm{e} \geq 0.9$\\")
        print(r"[0.1ex]\\")
        print(r"\hline")
        print(r"\\[0ex]")
        print(r"$\langle f_{{\mathrm{{RM}},k}} \rangle~\%$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$\\".format(np.round(fRM_i_tot[0]*100,2), np.round(fRM_i_tot[1]*100,2), np.round(fRM_i_tot[2]*100,2), np.round(fRM_i_tot[3]*100,2), np.round(fRM_i_tot[4]*100,2)))
        print(r"[0.7ex]")
        print(r"$f_{{\mathrm{{RM-same}},k}}$ \% & ${}$ & ${}$ & ${}$ & ${}$ & ${}$\\".format(np.round(sign_RM_i[0]*100,2), np.round(100*sign_RM_i[1],2), np.round(100*sign_RM_i[2],2), np.round(100*sign_RM_i[3],2), np.round(100*sign_RM_i[4],2)))
        print(r"\\[-3ex]")
        print(r"\\ \hline\hline")
        print(r"\end{tabular}\\[1.5ex]")
        print(r"\footnotesize{Columns: (1) Physical and numerical quantities; (2), (3), (4), (5), (6) bin ranges of different physical quantities.")
        print(r"\\\vspace{0.1in}")
        print(r"}")
        print(r"\label{tab:RM_frac_table}")
        print(r"\end{table*}")
        stop()

def calculate_master_table_statistics(args, statistic, key, resolution, scenario, RM_resolution):


    snap = 100
    args.snappath = '/scratch/jh2/hs9158/results/LMC_run_{}_wind_scenario_{}{}/'.format(resolution, scenario, key)
    args.plotpath = "/scratch/jh2/hs9158/results/plots/LMC_run_{}_wind_scenario_{}{}/".format(resolution, scenario, key)
    print("working on resolution {} scenario {} with key {}".format(resolution, scenario, key))
    print("working on snapshot {}".format(snap))
    args.snapnum = snap; args.resolution = resolution; args.key = key; args.scenario = scenario; args.Tmin = 1e1; args.Tmax = 1e7
    points, RM = read_RM_map(args, args.snapnum, RM_resolution, 'RM', method='new')
    points, EM = read_RM_map(args, args.snapnum, RM_resolution, 'EM', method='new')
    # ignore points with absolute RM greater than 250
    ind = np.where(np.abs(EM)<100)[0]
    points = points[ind]; RM = RM[ind]
    if statistic=='alignment':
        p_pos, p_neg = boot.bootstrap_simulation(args,points, RM, statistic)
        return p_pos, p_neg
    else:
        stat = boot.bootstrap_simulation(args, points, RM, statistic)
        return stat

def print_master_table_statistics(args):
    keys = ['_turb2', '', '_turb2_order2', '_turb6_order2', '', '_turb2_order2', '_turb6_order2', '_turb12', '_turb6_order2', '_turb6_order2', '_turb6_order2']
    scenarios = [0, 0, 0, 0, 3, 3, 3, 0, 0, 3, 0]
    legend = ['turb2_s0', 'order2_s0', 'turb2_order2_s0', 'turb6_order2_s0', 'order2_s3', 'turb2_order2_s3', 'turb6_order2_s3', 'turb12_s0', 'turb6_order2_s0', 'turb6_order2_s3', 'turb6_order2_s0']
    resolutions = ['medres', 'medres', 'medres', 'medres', 'medres', 'medres', 'medres', 'medhighres', 'medhighres', 'medhighres', 'highres']
    I_RM = boot.bootstrap_observations(args, 'percentile')
    p_pos_obs, p_neg_obs = boot.bootstrap_observations(args, 'alignment')
    stat_obs = boot.bootstrap_observations(args, 'std')
    print("parameters for observations are ${} \pm {}$ & ${} \pm {}$ & ${} \pm {}$ & ${} \pm {}$".format(np.round(stat_obs[0], 2), np.round(stat_obs[1]/2, 2), np.round(I_RM[0], 2), np.round(I_RM[1]/2, 2), np.round(p_pos_obs[0], 2), np.round(p_pos_obs[1]/2, 2), np.round(p_neg_obs[0], 2), np.round(p_neg_obs[1]/2, 2)))
    for i in range(len(keys)):
        p_pos, p_neg = calculate_master_table_statistics(args, 'alignment', keys[i], resolutions[i], scenarios[i], 200)
        std = calculate_master_table_statistics(args, 'std', keys[i], resolutions[i], scenarios[i], 200)
        I_RM = calculate_master_table_statistics(args, 'percentile', keys[i], resolutions[i], scenarios[i], 200)
        print("table parameters for key {} resolution {} scenario {} are".format(keys[i], resolutions[i], scenarios[i]))
        print('${} \pm {}$ & ${} \pm {}$ & ${} \pm {}$ & ${} \pm {}$'.format(np.round(std[0], 2), np.round(std[1]/2, 2), np.round(I_RM[0], 2), np.round(I_RM[1]/2, 2), np.round(p_pos[0], 2), np.round(p_pos[1]/2, 2), np.round(p_neg[0], 2), np.round(p_neg[1]/2, 2)))

def print_master_table_time_statistics(args):
    I_RM = boot.bootstrap_observations(args, 'percentile')
    p_pos_obs, p_neg_obs = boot.bootstrap_observations(args, 'alignment')
    stat_obs = boot.bootstrap_observations(args, 'std')
    print("parameters for observations are ${} \pm {}$ & ${} \pm {}$ & ${} \pm {}$ & ${} \pm {}$".format(np.round(stat_obs[0], 2), np.round(stat_obs[1]/2, 2), np.round(I_RM[0], 2), np.round(I_RM[1]/2, 2), np.round(p_pos_obs[0], 2), np.round(p_pos_obs[1]/2, 2), np.round(p_neg_obs[0], 2), np.round(p_neg_obs[1]/2, 2)))
    for RM_resolution in [100, 200, 300, 400, 500]:
        p_pos, p_neg = calculate_master_table_statistics(args, 'alignment', '_turb6_order2', 'medhighres', 3, RM_resolution)
        std = calculate_master_table_statistics(args, 'std', '_turb6_order2', 'medhighres', 3,  RM_resolution)
        I_RM = calculate_master_table_statistics(args, 'percentile', '_turb6_order2', 'medhighres', 3, RM_resolution)
        print("table parameters for RM resolution {} are".format(RM_resolution))
        print("${} \pm {}$ & ${} \pm {}$ & ${} \pm {}$ & ${} \pm {}$".format(np.round(std[0], 2), np.round(std[1]/2, 2), np.round(I_RM[0], 2), np.round(I_RM[1]/2, 2), np.round(p_pos[0], 2), np.round(p_pos[1]/2, 2), np.round(p_neg[0], 2), np.round(p_neg[1]/2, 2)))
              

        
# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":


    # parse command line arguments
    parser = argparse.ArgumentParser(description='Setting up simulation for LMC in GIZMO...')
    parser.add_argument('-o', '--process_temperatures', type=int, help='optional argument to process temperatures for a given snapshot')
    parser.add_argument('-s', '--process_ne', type=int, help='optional argument to process ne for a given snapshot')
    parser.add_argument('-w', '--workers', type=int, help='number of workers to use for parallel processing')
    parser.add_argument('-e', '--process_everything', type=int, help='optional argument to process everything for a given snapshot')
    parser.add_argument('-a', '--scenario', type=int, help='optional scenario argument')
    parser.add_argument('-b', '--key', type=str, help='optional key argument')
    parser.add_argument('-r', '--RM_analysis', type=int, help='optional RM analysis argument')
    parser.add_argument('-f', '--SFR_analysis', type=int, help='optional SFR analysis argument')
    parser.add_argument('-z', '--table_B', type=int, help='optional argument to obtain B values for the table.')
    parser.add_argument('resolution', type=str, help='The resolution to be used for the simulation analysis. It can be lowres or medres...')
    args = parser.parse_args()
    args.center_method = 'com'  # 'density' or 'velocity' or 'com': density: follows the densest gas cell; velocity: follows the velocity vector of the LMC; com follows the center of mass of the LMC disk
    args.snapnum = 100
    args.filename = 'snapshot_{}.hdf5'.format(str(args.snapnum).zfill(3))
    args.dirfilename = 'snapdir_{}/snapshot_{}.0.hdf5'.format(str(args.snapnum).zfill(3), str(args.snapnum).zfill(3))
    if args.scenario is None: args.scenario = 0
    args.TRe_key = int(37)
    args.TRe = 10**(args.TRe_key/10)
    if args.key is None: args.key = ''  # '_field_reversals' or '_B_reduce4' or '' or '_turb2' or '_turb2_order2' or '_turb6_order2'
    key = args.key
    # args.resolution = 'medres'
    ## For jh2 output
    # args.snappath = '/scratch/jh2/hs9158/results/LMC_run_{}_wind_scenario_{}{}/'.format(args.resolution, args.scenario, key)
    args.snappath = '/scratch/jh2/hs9158/results/MW_run_89Msun/'
    args.plotpath = "/scratch/jh2/hs9158/results/plots/LMC_run_{}_wind_scenario_{}{}/".format(args.resolution, args.scenario, key)
    args.V_LMC = np.array([-314.8, 29.03, -53.951])  # in km/s
    args.tstep = 5e6 # in yr
    if args.resolution == 'medres': args.ngasdisk = int(22e5)
    if args.resolution == 'lowres': args.ngasdisk = int(1e5) # number of gas particles initialized in the disk
    if args.resolution == 'medhighres': args.ngasdisk = int(88e5)
    if args.resolution == 'highres': args.ngasdisk = int(22e6)
    if args.scenario<2: args.simtype = 'isolated'; args.center = np.array([100, 100, 100])
    else:
        args.simtype = 'wind'
        if args.center_method == 'density':
            with h5py.File(args.snappath+args.filename, 'r') as F:
                coords = F['PartType0']['Coordinates'][:]
                rho = F['PartType0']['Density'][:]
            args.center = coords[np.argmax(rho)]
        if args.center_method == 'velocity': args.center = np.array([100, 100, 100+np.linalg.norm(args.V_LMC)*1e5*constants.year/constants.pc/1e3*args.snapnum*args.tstep])
        if args.center_method == 'com':
            with h5py.File(args.snappath+args.filename, 'r') as F:
                coords = F['PartType2']['Coordinates'][0:args.ngasdisk][:]
                fake_center = np.array([100, 100, 100+np.linalg.norm(args.V_LMC)*1e5*constants.year/constants.pc/1e3*args.snapnum*args.tstep])
                # find indices within 20 kpc of the fake center
                ind = np.where(np.linalg.norm(coords-fake_center, axis=1)<5)[0]
                # find indices common between ind and np.arange(args.ngasdisk)
                ind = np.intersect1d(ind, np.arange(args.ngasdisk))
                # args.center = np.average(coords[ind], axis=0)
                args.center = np.average(coords, axis=0)
            
    print("Some important parameters for this run are: snappath={}, plotpath={}; simtype={}".format(args.snappath, args.plotpath, args.simtype))
    print("Number of CPU cores/hyperthreads:", multiprocessing.cpu_count())
    # For mk27
    # args.snappath = '/scratch/mk27/hs9158/LMC_run_lowres_wind/'
    # args.plotpath = '/scratch/mk27/hs9158/plots/test3/'
    
    args.n_LMC = np.array([0.43, -0.373, 0.822])
    # compute angle between args.n_LMC and args.V_LMC
    args.theta_LMC = np.radians(180-np.degrees(np.arccos(np.dot(args.n_LMC, args.V_LMC/np.linalg.norm(args.V_LMC)))))

    R = np.array([[np.cos(args.theta_LMC), 0, np.sin(args.theta_LMC)], [0, 1, 0], [-np.sin(args.theta_LMC), 0, np.cos(args.theta_LMC)]])  # negative of the making of LMC rotation_matrix
    # R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    alpha = np.radians(-34.7); beta = np.radians(0); gamma = np.radians(-147.5)
    # alpha about x-axis, beta about y-axis, gamma seems to rotate about z-axis
    a = np.cos(alpha); b = np.cos(beta); c = np.cos(gamma)
    d = np.sin(alpha); e = np.sin(beta); f = np.sin(gamma)
    R_obs = np.array([[b*c, d*e*c-a*f, a*e*c+d*f], [b*f, d*e*f+a*c, a*e*f-d*c], [-e, d*b, a*b]])

    alpha = np.radians(0); beta = args.theta_LMC; gamma = np.radians(225)
    a = np.cos(alpha); b = np.cos(beta); c = np.cos(gamma)
    d = np.sin(alpha); e = np.sin(beta); f = np.sin(gamma)    
    R_new = np.array([[b*c, d*e*c-a*f, a*e*c+d*f], [b*f, d*e*f+a*c, a*e*f-d*c], [-e, d*b, a*b]])

    alpha = np.radians(+34.7-180); beta = np.radians(0); gamma = np.radians(90-(180-139.9))
    a = np.cos(alpha); b = np.cos(beta); c = np.cos(gamma)
    d = np.sin(alpha); e = np.sin(beta); f = np.sin(gamma)    
    R_obs_new = np.array([[b*c, d*e*c-a*f, a*e*c+d*f], [b*f, d*e*f+a*c, a*e*f-d*c], [-e, d*b, a*b]])
    # R_obs = np.array([[-0.527, 0.767, -0.365], [0.631, -0.641, 0.437], [-0.569, 0, 0.822]])  # From angular momentum aligned with z-axis to observer's frame
    R_sim_obs = np.matmul(R_obs, R)  # multiply the rotation matrices to get the final rotation matrix
    R_sim_obs_new = np.matmul(R_obs_new, R_new)  # multiply the rotation matrices to get the final rotation matrix
    args.rotation_matrix = R_sim_obs_new
    ## dirty arguments now
    dirty_args = parser.parse_args()
    dirty_args.resolution = args.resolution
    dirty_args.key1 = 'T'
    dirty_args.key2 = 'ne'
    dirty_args.range_min = 5
    dirty_args.range_max = 100
    args.node_vector = np.array([0.62887557, -0.5295624, -0.56927952])
    # Normal analysis functions start from here
    if args.process_temperatures is not None:
        process_temperatures(args, args.process_temperatures, save=True)
        stop()
    if args.process_ne is not None:
        workers = args.workers
        steps = 1  # can be either 10 r 1
        # steps = 1
        ds = rm.CorrectneCloudySlug(args, args.process_ne, Tmin=1e1, Tmax=args.TRe, cond='T', col=1e19, rho=1e-1, ne_corrected=True, mode='create')  # here Tmin and Tmax are ranges to correct the electron number density
        # main_ne(args, args.process_ne, args.process_ne+workers*steps, steps, workers, mode='create')
        stop()
    if args.process_everything is not None:
        process_everything(args, dirty_args, 80, 90, 8, R, True, False, 'create')
        stop()
    if args.RM_analysis is not None:
        RM_key = 'RM'
        for i in range(100, 101, 1):
            args.snapnum = i; filename = 'snapshot_{}.hdf5'.format(str(i).zfill(3))
            if args.scenario<2: args.simtype = 'isolated'; args.center = np.array([100, 100, 100])
            else:
                args.simtype = 'wind'
                if args.center_method == 'density':
                    with h5py.File(args.snappath+filename, 'r') as F:
                        coords = F['PartType0']['Coordinates'][:]
                        rho = F['PartType0']['Density'][:]
                    args.center = coords[np.argmax(rho)]
                if args.center_method == 'velocity': args.center = np.array([100, 100, 100+np.linalg.norm(args.V_LMC)*1e5*constants.year/constants.pc/1e3*args.snapnum*args.tstep])
                if args.center_method == 'com':
                    with h5py.File(args.snappath+filename, 'r') as F:
                        coords = F['PartType2']['Coordinates'][0:args.ngasdisk][:]
                        fake_center = np.array([100, 100, 100+np.linalg.norm(args.V_LMC)*1e5*constants.year/constants.pc/1e3*args.snapnum*args.tstep])
                        # find indices within 20 kpc of the fake center
                        ind = np.where(np.linalg.norm(coords-fake_center, axis=1)<5)[0]
                        # find indices common between ind and np.arange(args.ngasdisk)
                        ind = np.intersect1d(ind, np.arange(args.ngasdisk))
                        # args.center = np.average(coords[ind], axis=0)
                        args.center = np.average(coords, axis=0)
            print("The center is at: ", args.center)
            
            ds = yt.load(args.snappath+filename)
            ds = rm.CorrectIons(args, ds, i)
            ad = ds.all_data()
            args.Tmin = 1e1; args.Tmax = 1e7
            if i==100:
                for res in range(200, 300, 100):
                    if res==200:
                        ds = yt.load(args.snappath+filename)
                        ds = rm.CorrectIons(args, ds, i)
                        ad = ds.all_data()
                        args.Tmin = 1e1; args.Tmax = 1e7
                        points, RM = create_RM_map(args, ad, 10, res, R_new, R_obs_new, RM_key)
                        pl.plot_RM_map(args, points, RM, 'RM')
                        ds = yt.load(args.snappath+filename)
                        ds = rm.CorrectIons(args, ds, i)
                        ad = ds.all_data()
                        args.Tmin = 1e1; args.Tmax = 5e3
                        points, RM = create_RM_map(args, ad, 10, res, R_new, R_obs_new, RM_key)
                        pl.plot_RM_map(args, points, RM, 'RM')
                        ds = yt.load(args.snappath+filename)
                        ds = rm.CorrectIons(args, ds, i)
                        ad = ds.all_data()
                        args.Tmin = 5e3; args.Tmax = 8e3
                        points, RM = create_RM_map(args, ad, 10, res, R_new, R_obs_new, RM_key)
                        pl.plot_RM_map(args, points, RM, 'RM')
                        ds = yt.load(args.snappath+filename)
                        ds = rm.CorrectIons(args, ds, i)
                        ad = ds.all_data()
                        args.Tmin = 8e3; args.Tmax = 2e4
                        points, RM = create_RM_map(args, ad, 10, res, R_new, R_obs_new, RM_key)
                        pl.plot_RM_map(args, points, RM, 'RM')
                        ds = yt.load(args.snappath+filename)
                        ds = rm.CorrectIons(args, ds, i)
                        ad = ds.all_data()
                        args.Tmin = 2e4; args.Tmax = 1e7
                        points, RM = create_RM_map(args, ad, 10, res, R_new, R_obs_new, RM_key)
                        pl.plot_RM_map(args, points, RM, 'RM')
                    else:
                        ds = yt.load(args.snappath+filename)
                        ds = rm.CorrectIons(args, ds, i)
                        ad = ds.all_data()
                        args.Tmin = 1e1; args.Tmax = 1e7
                        points, RM = create_RM_map(args, ad, 10, res, R_new, R_obs_new, RM_key)
                        pl.plot_RM_map(args, points, RM, 'RM')
            else: 
                points, RM = create_RM_map(args, ad, 10, 200, R_new, R_obs_new, RM_key)
                pl.plot_RM_map(args, points, RM, 'RM')
        stop()

    if args.SFR_analysis is not None:
        calculate_SFR(args, 100)
        stop()
    if args.table_B is not None:
        # keys = ['', '_turb2', '_turb2_order2', '_turb6_order2', '', '_turb2_order2', '_turb6_order2', '_turb6_order2', '_turb12', '_turb6_order2', '_turb6_order2']
        # scenarios = [0, 0, 0, 0, 3, 3, 3, 0, 0, 3, 0]
        # legend = ['order2_s0', 'turb2_s0', 'turb2_order2_s0', 'turb6_order2_s0', 'order2_s3', 'turb2_order2_s3', 'turb6_order2_s3', 'turb6_order2_s0', 'turb12_s0', 'turb6_order2_s3', 'turb6_order2_s0']
        snap = 100
        args.snappath = '/scratch/jh2/hs9158/results/LMC_run_{}_wind_scenario_{}{}/'.format(args.resolution, args.scenario, args.key)
        args.plotpath = "/scratch/jh2/hs9158/results/plots/LMC_run_{}_wind_scenario_{}{}/".format(args.resolution, args.scenario, args.key)
        print("working on resolution {} scenario {} with key {}".format(args.resolution, args.scenario, args.key))
        print("working on snapshot {}".format(snap))
        ds = yt.load(args.snappath+'snapshot_{:03d}.hdf5'.format(args.snapnum))
        r, bturb = pl.plot_rcyl_profile(args, ds, zscale = 0.5, rscale=10, weight='Volume', property='MagneticField', component = 'turb', rotation_matrix=R_new, plot=False)
        r, baz = pl.plot_rcyl_profile(args, ds, zscale = 0.5, rscale=10, weight='Volume', property='MagneticField', component = 'azimuthal', rotation_matrix=R_new, plot=False)
        r, Btot = pl.plot_rcyl_profile(args, ds, zscale = 0.5, rscale=10, weight='Volume', property='MagneticField', component = 'total', rotation_matrix=R_new, plot=False)
        # np.save(args.plotpath+"Btot.npy", Btot)
        # np.save(args.plotpath+"r.npy", r)
        r = np.load(args.plotpath+"r.npy")
        plt.figure(figsize=(7*0.8, 5*0.8)); plt.plot(r, Btot, label=r'$<|B_{\rm total}|>~(\mu{\rm G})$', linewidth=3); plt.plot(r, bturb, label=r'$<|B_{\rm turb}|>~(\mu{\rm G})$', linewidth=3); plt.plot(r, baz, label=r'$<B_{\rm azimuthal}>~(\mu{\rm G})$', linewidth=3); plt.xlabel(r'$r~({\rm kpc})$'); plt.ylabel(r'$B~(\mu{\rm G})$'); plt.xlim([0, 9]); plt.legend(loc='best'); plt.show()
        B_tot_wind = np.load("/scratch/jh2/hs9158/results/plots/LMC_run_medhighres_wind_scenario_3_turb6_order2/Btot.npy")
        B_tot_isolated = np.load("/scratch/jh2/hs9158/results/plots/LMC_run_medhighres_wind_scenario_0_turb6_order2/Btot.npy")
        scale=1.2
        plt.figure(figsize=(7*scale, 5*scale)); plt.plot(r, B_tot_wind, label=r'$<|B_{\rm total}|>~(\mu{\rm G})+{\rm wind}$', linewidth=3); plt.plot(r, B_tot_isolated, label=r'$<|B_{\rm total}|>~(\mu{\rm G})+{\rm isolated}$', linewidth=3); plt.xlabel(r'$r~({\rm kpc})$'); plt.ylabel(r'$B~(\mu{\rm G})$'); plt.xlim([0, 9]); plt.legend(loc='best'); plt.show()
        density_wind = np.load("/scratch/jh2/hs9158/results/plots/LMC_run_medhighres_wind_scenario_3_turb6_order2/density.npy")
        density_isolated = np.load("/scratch/jh2/hs9158/results/plots/LMC_run_medhighres_wind_scenario_0_turb6_order2/density.npy")
        plt.figure(figsize=(7*scale, 5*scale)); plt.plot(r, density_wind/density_isolated, label=r'$<\rho>+{\rm wind}/<\rho>+{\rm isolated}$', linewidth=3); plt.plot(r, B_tot_wind/B_tot_isolated, label=r'$<|B_{\rm total}|>~(\mu{\rm G})+{\rm wind}/<|B_{\rm total}|>~(\mu{\rm G})+{\rm isolated}$', linewidth=3); plt.xlabel(r'$r~({\rm kpc})$'); plt.ylabel('Ratio'); plt.xlim([0, 9]); plt.legend(loc='best'); plt.show()
        #  plt.xlabel(r'$r~({\rm kpc})$'); plt.ylabel(r'$B~(\mu{\rm G})~{\rm ratio}$'); plt.xlim([0, 9]); plt.legend(loc='best'); plt.show()
        stop()
        # find the index closest to 2.5 in r
        ind = np.where(np.abs(r-2.5)==np.min(np.abs(r-2.5)))[0]
        print("Bturb and Baz at 2.5 kpc from center are {} & {}".format(np.round(bturb[ind][0], 2), np.round(baz[ind][0], 2)))
        # stop()
    # process_temperatures(args, 23, save=True)
    # stop()
    # read_Grid(128, 100, 'medhighres', 0)
    # stop()
    # make_simulation_grid(args, 100, R, 20, 20, 128)  # enter the resolution in Ngrid
    # make_simulation_grid(args, 100, R, 20, 20, 1024)  # enter the resolution in Ngrid
    # make_simulation_grid(args, 100, R, 10, 10, 2048)  # enter the resolution in Ngrid
    # stop()
    # pl.make_paper_RMstats(args, 100)
    # stop()
    # pl.make_paper_plot_Brad(args, 100)
    # stop()
    # pl.make_paper_plot_ne_B(args, 100)
    # stop()
    # pl.make_paper_SFR_plot()
    # pl.make_paper_plot_Bedge(args, 100, key='histogram')
    # stop()
    # pl.slice_align(args, 100, 'z', R, simtype='wind', zoom=True)
    # stop()
    # args.Tmin = 1e1; args.Tmax = 1e7
    # stop()
    # points, EM = read_RM_map(args, 100, 500, 'EM', 'new')
    # points, RM = read_RM_map(args, 100, 500, 'RM', 'new')
    # stop()
    # data = np.load("/scratch/jh2/hs9158/results/data/Paper_nH.npz")
    # nH = data['nH']
    # pl.paper_plot_RMaz(args, key='mag_average')
    # pl.make_RM_plot_paper(args, R_sim_obs_new)
    # print_master_table_time_statistics(args)
    # print_master_table_statistics(args)
    # stop()
    # rm.get_plane_rotation(args, R_obs_new)
    # stop()
    # RM_temperature_ion_analysis(args, key='plot') # key could be plot or table
    # pl.table_plot_sigmaRM(args)
    # stop()
    # rma.table_B(args)
    # new_gif_maker(args)
    # stop()
    # pl.make_RM_movie(args, R_sim_obs_new)
    # stop()
    # rma.positive_negative_RM_fraction(args)
    # pl.make_RMmaps_with_sigmaRM(args)
    # pl.make_RM_analysis_plots_res(args)
    # pl.make_paper_plot_rho(args, 100, rotated=True)
    # pl.make_paper_plot_B(args, rotated=True)
    # stop()
    # pl.make_phase_plots(args, 100)
    # stop()
    # calculate_DM_CGM(0.46, 0.35, 0.71)
    # pl.particle_plotter(args, 0, 'z', rotated=True)
    # stop()
    # stop()
    # DM_LMC(args)
    # stop()
    # for i in range(0, 67, 1):
    #     combine_hdf5(args, i)
    # stop()

    # clean_process_everything(args, 4, 100, 8, R, True, False, 'create')
    
    # pl.plotRMz_T(args, 10, 200, 'RM', ind_cond='all')
    # stop()
    # pl.PlotDiagnostics(args, 84)
    # pl.subplot_slice_saver(args=args, i=10, plane='y', rotation_matrix=R, simtype=args.simtype, zoom=False)
    # pl.subplot_slice_saver(args=args, i=200, plane='z', rotation_matrix=R, simtype=args.simtype, zoom=True)
    # stop()
    # pl.subplot_plotter(args, 1, 'y', rotated=True, slice=False)
    # for i in range(90, 101, 1):
    #     args.snapnum = i
    #     process_temperatures(args, i)
    # stop()
    # pl.subplot_plotter(args, 0, 'y', rotated=True)
    # pl.subplot_projection_saver(args=args, i=0, plane='z', rotation_matrix=R, simtype=args.simtype, zoom=True)
    # main_temperature(args, args_list_num_start=0, args_list_num_end=26, workers=1)
    # main_plot(args, args_list_num_start=80, args_list_num_end=101, workers=32, plane='z', rotated=True, slice=False)
    # main_plot(args, args_list_num_start=80, args_list_num_end=101, workers=32, plane='y', rotated=True, slice=False)
    # stop()
    # for i in range(20):
    #     pl.subplot_movie_plotter(args, i, 'z', rotated=True, slice=False)
    #     pl.subplot_movie_plotter(args, i, 'y', rotated=True, slice=False)
    # stop()
    # main(args, args_list_num_start=0, args_list_num_end=26, workers=4, plane='y', rotation_matrix=R, zoom=True, slice=False)
    # main(args, args_list_num_start=0, args_list_num_end=26, workers=4, plane='z', rotation_matrix=R, zoom=True, slice=False)
    # main_plot(args, args_list_num_start=0, args_list_num_end=26, workers=26, plane='y', rotated=True, slice=False)
    # main_plot(args, args_list_num_start=0, args_list_num_end=26, workers=26, plane='z', rotated=True, slice=False)
    # main(args, args_list_num_start=170, args_list_num_end=200, workers=30, plane='z', rotation_matrix=R, zoom=True, slice=True)
    # main(args, args_list_num_start=0, args_list_num_end=100, workers=32, plane='z', rotation_matrix=R, zoom=False, slice=True)
    # args.Tmin = 1e1; args.Tmax = 1e7
    # pl.make_pretty_plots(args, rotation_matrix=R_sim_obs_new)
    # stop()
    # main_ne(args, args_list_num_start=89, args_list_num_end=101, steps=1, workers=12, mode='create')
    # # pl.reynolds_test(args)
    # stop()
    # T_range = np.logspace(1, 4.3, 71)
    # den_range = np.logspace(-3, 3, 61)
    # # # cc.compute_quantities_from_ion(den_range, T_range[0], extinguish=True, ext_den=21)
    # cc.ion_density_temperature(den_range, T_range, extinguish=True, ext_den=21)
    # for T in T_range:
    #     # cc.make_different_cloudy_runs(T, extinguish=False)
        # cc.make_different_cloudy_runs(T, extinguish=True, ext_den=21)
    #     cc.make_different_cloudy_runs(T, extinguish=True, ext_den=21.5)
    #     cc.make_different_cloudy_runs(T, extinguish=True, ext_den=20.5)
    # stop()

    # ds = yt.load(args.snappath+args.filename)
    # ds = rm.compute_density_gradient(args, ds, 10, 200)
    # ad = ds.all_data()
    # stop()
    # ds = rm.CorrectIons(args, ds, args.snapnum)
    # ad = ds.all_data()
    # arr = np.load(args.snappath+'snapshot_{}_processed_quantities_T{}.npy'.format(str(args.snapnum).zfill(3), args.TRe_key), allow_pickle=True)
    # arrdata = arr.item()
    # ind0 = np.where(arrdata['hydrogen_number_density']==0)[0]
    # # indices apart from the ones where hydrogen number density is zero
    # ind1 = np.where(arrdata['hydrogen_number_density']!=0)[0]
    # stop()
    
    # # pl.test_rotation_matrix(args, ds, R=R_new, R_obs=R_obs_new, center=np.array([100, 100, 100]))
    # stop()
    
    # main_particle_plot(args, 0, 79, 16)
    # stop()
    # pl.plot_stars(args, plane='x')
    # stop()
    
    
    # calculate_temperature(args, 84)
    # mass_flux = calculate_mass_outflow_rate(args, 59, 9.6, 2)
    # stop()
    # PlotDiagnostics(args, 59)
    
    # print("Starting the plotting loop...")
    # for i in range(0, 99, 1):
    # #     print(i)
    # # #     pl.subplot_plotter(args, i, 'z', rotated=True, slice=True)
    # # #     pl.subplot_plotter(args, i, 'y', rotated=True, slice=True)
    #     pl.subplot_plotter(args, i, 'z', rotated=True, slice=False)
    #     pl.subplot_plotter(args, i, 'y', rotated=True, slice=False)

    # stop()
    # gif_maker(args, args.plotpath, 0, 100, plane='z', subplot='movie', rotated=True, projected=True)
    # # gif_maker(args.plotpath, 0, 99, plane='z', subplot=True, rotated=True, projected=False)
    # gif_maker(args, args.plotpath, 0, 100, plane='y', subplot='movie', rotated=True, projected=True)
    # # gif_maker(args.plotpath, 0, 99, plane='y', subplot=True, rotated=True, projected=False)    
    # stop()
    # # load SFR data from data folder
    # SFR = np.loadtxt(args.snappath+'SFR.txt', delimiter=' ', dtype=float)
    # plt.figure()
    # plt.plot(np.arange(150, 200)*args.tstep/1e6, SFR[150:200])
    # plt.xlabel('Time (Myr)')
    # plt.ylabel('SFR (Msol/yr)')
    # plt.yscale('log')
    # plt.ylim([1e-2, 1e1])
    # plt.title("key={}".format(args.key))
    # plt.show()
    # stop()
    # # same plot as above for temperatures
    # T = np.loadtxt(args.snappath+'temperature.txt', delimiter=' ', dtype=float)
    # plt.figure()
    # plt.plot(np.arange(len(SFR))*args.tstep/1e6, T)
    # plt.xlabel('Snapshot number')
    # plt.ylabel('Temperature (K)')
    # plt.legend()
    # plt.yscale('log')
    # plt.show()
    # stop()

    # # parallelize the following loop
    # # num_cores = multiprocessing.cpu_count(); print("Number of cores=", num_cores)
    # # Parallel(n_jobs=num_cores)(delayed(VrPlot_maker)(args) for args.snapnum in range(91))
    # # Parallel(n_jobs=num_cores)(delayed(subplot_maker)(args) for args.snapnum in range(91))

    # subplot_maker('z'); subplot_maker('y'); subplot_maker('x')

    # for i in range(99):
    #     args.snapnum = i
    #     args.filename = 'snapshot_{}.hdf5'.format(str(args.snapnum).zfill(3))
    #     ds = yt.load(args.snappath+args.filename)
    #     ds = rm.create_ionizing_luminosity_data(args, ds, 'create')
    # stop()
    # ds = yt.load(args.snappath+args.filename)
    # # ds = rm.CorrectIons(args, ds, args.snapnum)
    # # stop()
    # ad = ds.all_data()
    args.Tmin = 1e1; args.Tmax = 1e7
    # # # create_prop_rays(args, ad, 10, 200, R_new, R_obs_new, 'ne', 'all')
    # # points, nH = create_prop_rays(args, ad, 10, 200, R_new, R_obs_new, 'nH', 'data')
    # ds = yt.load(args.snappath+args.filename)
    # ad = ds.all_data()
    # points, RM = create_RM_map_ion(args, ad, 10, 200, R_new, R_obs_new, 'RM', 90, 120)
    # stop()
    # ds = yt.load(args.snappath+args.filename)
    # points, RM = create_RM_map_nH(args, ad, 10, 200, R_new, R_obs_new, 'RM', 1e1, 1e16)
    # ds = yt.load(args.snappath+args.filename)
    # ad = ds.all_data()
    # points, RM = create_RM_map_nH(args, ad, 10, 200, R_new, R_obs_new, 'RM', 1e0, 1e1)
    # ds = yt.load(args.snappath+args.filename)
    # ad = ds.all_data()
    # points, RM = create_RM_map_nH(args, ad, 10, 200, R_new, R_obs_new, 'RM', 1e-1, 1e0)
    # ds = yt.load(args.snappath+args.filename)
    # ad = ds.all_data()
    # points, RM = create_RM_map_nH(args, ad, 10, 200, R_new, R_obs_new, 'RM', 1e-16, 1e-1)
    # stop()
    # create_DM_map(args, ad, 10, 200, R_new, R_obs_new)
    # points, RM = create_RM_map(args, ad, 10, 2, R_new, R_obs_new, 'RM')
    # # # pl.plot_RM_map(args, points, RM, 'DM')
    # stop()

    # ds = yt.load(args.snappath+args.filename)
    # ad = ds.all_data()
    # args.Tmin = 1e1; args.Tmax = 8e3
    # create_DM_map(args, ad, 10, 200, R_new, R_obs_new)
    # # points, RM = create_RM_map(args, ad, 10, 200, R_new, R_obs_new, 'DM')
    # # pl.plot_RM_map(args, points, RM, 'DM')

    # ds = yt.load(args.snappath+args.filename)
    # ad = ds.all_data()
    # args.Tmin = 1e5; args.Tmax = 1e7
    # create_DM_map(args, ad, 10, 200, R_new, R_obs_new)
    # # points, RM = create_RM_map(args, ad, 10, 200, R_new, R_obs_new, 'DM')
    # # pl.plot_RM_map(args, points, RM, 'DM')
    # # stop()
    # ds = yt.load(args.snappath+args.filename)
    # ad = ds.all_data()
    # args.Tmin = 8e3; args.Tmax = 20e3
    # create_DM_map(args, ad, 10, 200, R_new, R_obs_new)
    # points, RM = create_RM_map(args, ad, 10, 200, R_new, R_obs_new, 'DM')
    # pl.plot_RM_map(args, points, RM, 'DM')
    # ds = yt.load(args.snappath+args.filename)
    # ad = ds.all_data()
    # args.Tmin = 20e3; args.Tmax = 1e5
    # create_DM_map(args, ad, 10, 200, R_new, R_obs_new)
    # points, RM = create_RM_map(args, ad, 10, 200, R_new, R_obs_new, 'RM')
    # pl.plot_RM_map(args, points, RM, 'DM')
    # stop()
    # r, prop = pl.plot_rcyl_profile(args, ds, zscale = 0.5, rscale=10, weight='Volume', property='Density', component = 'Total', rotation_matrix=R_new, plot=True)
    # np.save(args.plotpath+"density.npy", prop)


    # ds = rm.test_yt_issues(args, ds)
    # stop()
    # arr = np.load(args.snappath+'snapshot_{}_processed_quantities.npy'.format(str(4940).zfill(3)), allow_pickle=True)
    # extract the data from the array which is a set
    # data = arr.item()
    # args.Tmin = 1e1; args.Tmax = 1e7
    # points, RM = read_RM_map(args, 93, 200, 'RM')
    # pl.plot_RM_map(args, points, RM, 'RM')
    # stop()
    ds = rm.CorrectneCloudySlug(args, 23, Tmin=1e1, Tmax=args.TRe, cond='T', col=1e19, rho=1e-1, ne_corrected=True, mode='create', compile=True)  # here Tmin and Tmax are ranges to correct the electron number density
    stop()
    ds = yt.load(args.snappath+args.filename)
    ad = ds.all_data()
    args.Tmin = 1e1; args.Tmax = 1e7
    create_DM_map(args, ad, 10, 200, R_new, R_obs_new)
    stop()
    # pl.plot_time_rcyl_profile(args, 100, timesteps = 5, zscale = 0.25, rscale=10, weight='Volume', property='MagneticField', component='total', rotation_matrix=R, save=False)
    # pl.plot_last_rcyl_profile(args, 100, zscale = 0.5, rscale=10, weight='Volume', property='MagneticField', component='azimuthal', rotation_matrix=R_new)
    # stop()
    # ad = ds.all_data()
    # ad = make_temperature_cut(args, ds, 10, 20000)

    # stop()
    # results_uncorrected = np.load("/scratch/jh2/hs9158/results/data/point_RM300_lowres_scenario3_snap_99_Tmin_10.0_Tmax_10000000.0_uncorrected.npz")
    # results_corrected = np.load("/scratch/jh2/hs9158/results/data/point_RM300_lowres_scenario3_snap_99_Tmin_10.0_Tmax_10000000.0.npz")
    args.Tmin = 1e1; args.Tmax = 1e7
    # sigma_RM = []
    # for i in range(50, 100):
    #     points, RM = read_RM_map(args, i, 100)
    #     sigma_RM.append(np.std(RM))

    # plt.plot(np.arange(50, 100)*5, sigma_RM)
    # plt.xlabel('Time (Myr)')
    # plt.ylabel('sigmaRM')
    # plt.ylim([0, 20])
    # plt.title("key = {}".format(key))
    # plt.show()
    # stop()
    # ad = make_temperature_cut(args, ds, args.Tmin, args.Tmax)
    for i in range(50, 100, 1):
        print(i)
        args.snapnum = i
        ds = yt.load(args.snappath+"snapshot_{}.hdf5".format(str(i).zfill(3)))
        # ds = rm.CorrectIons(args, ds, i)
        ad = ds.all_data()
        points, RM = create_RM_map(args, ad, 10, 100, R, R_obs)
        pl.plot_RM_map(args, points, RM)
    stop()
    # RMdata = np.load("/scratch/jh2/hs9158/results/data/point_RM{}_{}_scenario{}_snap_{}_Tmin_{}_Tmax_{}.npz".format(300, args.resolution, args.scenario, args.snapnum, args.Tmin, args.Tmax), allow_pickle=True)
    # points = RMdata['data'][:, :2]
    # RM = RMdata['data'][:, 2]
    # pl.plot_RM_map(args, points, RM)

    args.Tmin = 1e1; args.Tmax = 8e3
    ad = ds.all_data()
    ad = make_temperature_cut(args, ds, args.Tmin, args.Tmax)
    points, RM = create_RM_map(args, ad, 10, 100, R, R_obs)
    pl.plot_RM_map(args, points, RM)

    args.Tmin = 8e3; args.Tmax = 20e3
    ad = ds.all_data()
    ad = make_temperature_cut(args, ds, args.Tmin, args.Tmax)
    points, RM = create_RM_map(args, ad, 10, 100, R, R_obs)
    pl.plot_RM_map(args, points, RM)

    args.Tmin = 20e3; args.Tmax = 5e5
    ad = ds.all_data()
    ad = make_temperature_cut(args, ds, args.Tmin, args.Tmax)
    points, RM = create_RM_map(args, ad, 10, 100, R, R_obs)
    pl.plot_RM_map(args, points, RM)

    args.Tmin = 5e5; args.Tmax = 5e7
    ad = ds.all_data()
    ad = make_temperature_cut(args, ds, args.Tmin, args.Tmax)
    points, RM = create_RM_map(args, ad, 10, 100, R, R_obs)
    pl.plot_RM_map(args, points, RM)
    stop()
    # points, RM_values = create_RM_map(args, ds, 10, 200, R, R_obs)
    pl.create_scatterRM_plot(args, ds, 10, 20, R, R_obs)
    stop()
    # pl.make_particle_plot(args, ds)
    # stop()
    # pl.create_scatterRM_plot(args, 58, ds, 10, 10, rotation_matrix=np.array([[-0.527, 0.767, -0.365], [0.631, -0.641, 0.437], [-0.569, 0, 0.822]]))
    # create_scatterRM_plot(args, 48, ds, 10, 10, rotation_matrix=np.array([[-0.527, 0.767, -0.365], [0.631, -0.641, 0.437], [-0.569, 0, 0.822]]))
    # create_scatterRM_plot(args, 38, ds, 10, 10, rotation_matrix=np.array([[-0.527, 0.767, -0.365], [0.631, -0.641, 0.437], [-0.569, 0, 0.822]]))
    # create_scatterRM_plot(args, 28, ds, 10, 10, rotation_matrix=np.array([[-0.527, 0.767, -0.365], [0.631, -0.641, 0.437], [-0.569, 0, 0.822]]))
    stop()
    # points, RM_values = create_RM_map(args, ds, 10, 96, np.array([[-0.527, 0.767, -0.365], [0.631, -0.641, 0.437], [-0.569, 0, 0.822]]))
    RM_array = np.load("/scratch/jh2/hs9158/results/data/point_RM_96.npz")
    points = RM_array['data'][:, :2]
    RM_values = RM_array['data'][:, 2]
    
    stop()
    file0 = h5py.File(args.snappath+args.filename,'r')
    k0 = list(file0['PartType{}'.format(parttype)].keys())
    ad = ds.all_data()
    coord_x = ad['PartType{}'.format(parttype),'Coordinates'][:, 0]
    coord = ad['PartType{}'.format(parttype),'Coordinates'].in_units('kpc')
    v = ad[('PartType{}'.format(parttype), 'Velocities')]
    v = v.in_units('km/s')
    print("number of particles={} with PartType {}".format(len(coord), parttype))
    # plot radius from coord on x-axis and v on y-axis
    # fig, ax = plt.subplots()
    # v_norm = np.linalg.norm(v, axis=1); r = np.linalg.norm(coord, axis=1)
    # ax.scatter(r, v_norm, s=0.1)
    # ax.set_xlabel('r (kpc)')
    # ax.set_ylabel('V (km/s)')
    # plt.savefig('/scratch/jh2/hs9158/results/plots/V_r.png', dpi=300)

    ds.add_field(("PartType0", "coord_x"), function=my_axis_field0, units='kpc', sampling_type='local')
    ds.add_field(("PartType0", "coord_y"), function=my_axis_field1, units='kpc', sampling_type='local')
    ds.add_field(("PartType0", "coord_z"), function=my_axis_field2, units='kpc', sampling_type='local')
    # L = 7
    # cg = ds.covering_grid(level=L, left_edge=ds.domain_left_edge, dims=[2**L]*3)
    # fn = cg.save_as_dataset(fields=[("PartType0", "Masses")])
    # ds_grid = yt.load(fn)
    plot = yt.ParticlePlot(ds, ("PartType0", "coord_x"),("PartType0", "coord_z"), ("PartType0", "Masses"), figure_size=5)
    plot.set_log(("PartType0", "Masses"), True)
    plot.set_log("coord_x", False); plot.set_log("coord_z", False)
    plot.set_xlim(0, 10000); plot.set_ylim(0, 10000)
    plot.save(args.plotpath+'particle_{}_part0_yz.pdf'.format(args.snapnum))

    # using Meshoid
    stop()

