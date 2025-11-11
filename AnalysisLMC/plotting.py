#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Hilay Shah 2023-

import numpy as np
import h5py
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import yt
import yt.units as u
import unyt
from meshoid import Meshoid
from cfpack import print, hdfio, stop, constants
from multiprocessing import Pool
import pickle
import logging
from time import sleep
from random import randint
import os
from LMC_analysis import *
import cmasher as cmr
import matplotlib.gridspec as gridspec
import LMC_analysis as la
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.animation import FuncAnimation
from moviepy.editor import ImageSequenceClip
import glob
import re
import gaensler_analysis as ga
from mpl_toolkits.axes_grid1 import make_axes_locatable
import RM_analysis as rma

# plt.rcParams.update({'font.size': 10})  # Adjust font size here

def test_cofu(args, i):
    ans = np.array(i**2+2)
    path = args.plotpath
    np.savez(path+'test_{}.npz'.format(i), ans)
    print("data saved for {}".format(i))

def test_cofu_wrapper(args_dict):
    return test_cofu(**args_dict)

def subplot_slice_saver(args, i, plane='x', rotation_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), simtype='isolated', zoom=True):
    sleep_sec = randint(1, 10)
    snappath = args.snappath; plotpath = args.plotpath; print(i, snappath, plotpath)
    filename = 'snapshot_{}.hdf5'.format(str(i).zfill(3))
    if args.center_method == 'density':
        with h5py.File(args.snappath+args.filename, 'r') as F:
            coords = F['PartType0']['Coordinates'][:]
            rho = F['PartType0']['Density'][:]
        Mcenter = coords[np.argmax(rho)]
    if args.center_method == 'velocity': 
        if args.scenario<2: Mcenter = np.array([100, 100, 100])
        else: Mcenter = np.array([100, 100, 100+np.linalg.norm(args.V_LMC)*1e5*constants.year/constants.pc/1e3*i*args.tstep])
    particletype = 'PartType0'
    ds = yt.load(args.snappath+filename)
    rm.CorrectIons(args,ds)
    ad = ds.all_data()
    T = ad['PartType0', 'grackle_temperature']
    molecular_weight = (ad['PartType0', 'molecular_weight']*u.mh).in_units('g')
    # close the yt file
    ds.close()
    print("file open now.")
    with h5py.File(snappath+filename, "r") as F:
        coord = F[particletype]["Coordinates"][:]
        mass = F[particletype]["Masses"][:]
        B = F[particletype]["MagneticField"][:]
        V = F[particletype]["Velocities"][:]
        Z = F[particletype]["Metallicity"][:]
        # T = F[particletype]["InternalEnergy"][:]*1e10*1.3*constants.m_p*2/3/(constants.k_b)
        ParticleIDs = F[particletype]['ParticleIDs'][:]
        ind_disk = np.where((ParticleIDs > 0) & (ParticleIDs < args.ngasdisk))[0]
        if simtype=='wind': V[ind_disk] = V[ind_disk] - np.array([0, 0, np.linalg.norm(args.V_LMC)])  # subtracting the translating component
        hsml = F[particletype]["SmoothingLength"][:]
        SFflag = False
        if "PartType4" in F.keys():
            SFtime = F["PartType4"]["StellarFormationTime"][:]*1e3 # in Myr
            SFtime_recent_ind = np.where(((i-1)*args.tstep/1e6<SFtime) & (SFtime<i*args.tstep/1e6))[0] # SFR in the last 5 Myr
            SFmass = F["PartType4"]["Masses"][:][SFtime_recent_ind]*1e10 # in Msol
            SFcoords = F["PartType4"]["Coordinates"][:][SFtime_recent_ind]
            SFflag = True
    print("file read now.")
    # rotate vector quantities by rotation matrix
    B = np.dot(rotation_matrix, B.T).T
    V = np.dot(rotation_matrix, V.T).T
    # create an array of dot product between coord and V

    print("Meshoid start now.")
    if zoom: Msize=20
    else: Msize=60
    if simtype=='isolated':
        coord = coord-Mcenter; coord = np.dot(rotation_matrix, coord.T).T
        if SFflag==True: SFcoords = SFcoords-Mcenter; SFcoords = np.dot(rotation_matrix, SFcoords.T).T
        new_Mcenter = np.array([0, 0, 0])
        res = 200
        if plane=='x': Xcenter=new_Mcenter[1]; Ycenter=new_Mcenter[2]; ax1=1; ax2=2; axLOS=0
        if plane=='y': Xcenter=new_Mcenter[0]; Ycenter=new_Mcenter[2]; ax1=0; ax2=2; axLOS=1
        if plane=='z': Xcenter=new_Mcenter[0]; Ycenter=new_Mcenter[1]; ax1=0; ax2=1; axLOS=2
        X = np.linspace(Xcenter-Msize/2, Xcenter+Msize/2, res); Y = np.linspace(Ycenter-Msize/2, Ycenter+Msize/2, res)
        X, Y = np.meshgrid(X, Y)
    # rmax = 10; res = 200; X = Y = np.linspace(-rmax, rmax, res)
    if simtype=='wind':
        new_Mcenter = np.array([0, 0, 0])
        coord = coord-Mcenter; coord = np.dot(rotation_matrix, coord.T).T
        if SFflag==True: SFcoords = SFcoords-Mcenter; SFcoords = np.dot(rotation_matrix, SFcoords.T).T
        res = 50
        if plane=='x': Xcenter=new_Mcenter[1]; Ycenter=new_Mcenter[2]; ax1=1; ax2=2; axLOS=0
        if plane=='y': Xcenter=new_Mcenter[0]; Ycenter=new_Mcenter[2]; ax1=0; ax2=2; axLOS=1
        if plane=='z': Xcenter=new_Mcenter[0]; Ycenter=new_Mcenter[1]; ax1=0; ax2=1; axLOS=2
        X = np.linspace(Xcenter-Msize/2, Xcenter+Msize/2, res); Y = np.linspace(Ycenter-Msize/2, Ycenter+Msize/2, res)
        X, Y = np.meshgrid(X, Y)

    Vr = np.sum((coord)*V, axis=1)/np.linalg.norm((coord), axis=1) # should the translating component of the velocity be subtracted for this calc?!
    quantities = []
    quantities.append(X); quantities.append(Y)
    if SFflag:
        hist, xedges, yedges = np.histogram2d(SFcoords[:,ax1], SFcoords[:,ax2], bins=res, weights=SFmass, range=[[X.min(), X.max()], [Y.min(), Y.max()]])
        SFR_dens_proj = hist/args.tstep/(Msize/res)**2 # SFR density in Msol/yr/kpc^2
        quantities.append(SFR_dens_proj.T)
        del SFR_dens_proj, hist, xedges, yedges, SFcoords, SFmass, SFtime, SFtime_recent_ind
    # plt.scatter(coord[:,0], coord[:,2], s=0.05); plt.xlim([-50, 50]); plt.ylim([-50, 50]); plt.show()
    M = Meshoid(coord, mass, hsml)
    dens_slice = M.Slice(M.Density()/molecular_weight,
                         center=new_Mcenter, size=Msize, res=res, plane=plane)
    dens_slice = dens_slice*constants.m_sol*1e10/(1e3*constants.pc)**3
    quantities.append(dens_slice)
    del M, dens_slice
    Mbx = Meshoid(coord, B[:, ax1], hsml)
    BX_slice = Mbx.Slice(Mbx.m, center=new_Mcenter, size=Msize, res=res, plane=plane)
    quantities.append(BX_slice)
    del Mbx, BX_slice
    Mby = Meshoid(coord, B[:, ax2], hsml)
    BY_slice = Mby.Slice(Mby.m, center=new_Mcenter, size=Msize, res=res, plane=plane)
    quantities.append(BY_slice)
    del Mby, BY_slice
    Mb = Meshoid(coord, np.linalg.norm(B, axis=1), hsml)
    B_slice = Mb.Slice(Mb.m, center=new_Mcenter, size=Msize, res=res, plane=plane)
    quantities.append(B_slice)
    del Mb, B_slice
    Mvr = Meshoid(coord, Vr, hsml)
    V_slice = Mvr.Slice(Mvr.m, center=new_Mcenter, size=Msize, res=res, plane=plane)
    quantities.append(V_slice)
    del Mvr, V_slice
    Mz = Meshoid(coord, Z, hsml)
    Z_slice = Mz.Slice(Mz.m, center=new_Mcenter, size=Msize, res=res, plane=plane)
    quantities.append(Z_slice)
    del Mz, Z_slice
    Mtemp = Meshoid(coord, T, hsml)
    T_slice = Mtemp.Slice(Mtemp.m, center=new_Mcenter, size=Msize, res=res, plane=plane)
    quantities.append(T_slice)
    del Mtemp, T_slice
    MVx = Meshoid(coord, V[:, ax1], hsml)
    VX_slice = MVx.Slice(MVx.m, center=new_Mcenter, size=Msize, res=res, plane=plane)
    quantities.append(VX_slice)
    del MVx, VX_slice
    MVy = Meshoid(coord, V[:, ax2], hsml)
    VY_slice = MVy.Slice(MVy.m, center=new_Mcenter, size=Msize, res=res, plane=plane)
    quantities.append(VY_slice)
    del MVy, VY_slice
    # append the center of the domain
    quantities.append(Mcenter)
    print("Meshoid done and deleted now.")
    # save the list using np.savez
    if (rotation_matrix==np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).all():
        np.savez(plotpath+'quantities_plane{}_{}_sliced_c{}.npz'.format(plane, i, args.center_method), np.array(quantities))
    else:
        np.savez(plotpath+'quantities_plane{}_{}_rotated_sliced_c{}.npz'.format(plane, i, args.center_method), np.array(quantities))
    del quantities, coord, mass, hsml, B, V, Z, T, Vr, X, Y
    print("saved and deleted quantities for {}".format(filename))
    print('sleep: %d sec.' % (sleep_sec))
    sleep(sleep_sec)

def slice_align(args, i, plane='z', rotation_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), simtype='isolated', zoom=True):
    snappath = args.snappath; plotpath = args.plotpath; print(i, snappath, plotpath)
    filename = 'snapshot_{}.hdf5'.format(str(i).zfill(3))
    if args.center_method == 'density':
        with h5py.File(args.snappath+args.filename, 'r') as F:
            coords = F['PartType0']['Coordinates'][:]
            rho = F['PartType0']['Density'][:]
        Mcenter = coords[np.argmax(rho)]
    if args.center_method == 'velocity': 
        if args.scenario<2: Mcenter = np.array([100, 100, 100])
        else: Mcenter = np.array([100, 100, 100+np.linalg.norm(args.V_LMC)*1e5*constants.year/constants.pc/1e3*i*args.tstep])
    if args.center_method == 'com':
        with h5py.File(args.snappath+filename, 'r') as F:
            coords = F['PartType2']['Coordinates'][0:args.ngasdisk][:]
            fake_center = np.array([100, 100, 100+np.linalg.norm(args.V_LMC)*1e5*constants.year/constants.pc/1e3*i*args.tstep])
            # find indices within 20 kpc of the fake center
            ind = np.where(np.linalg.norm(coords-fake_center, axis=1)<5)[0]
            # find indices common between ind and np.arange(args.ngasdisk)
            ind = np.intersect1d(ind, np.arange(args.ngasdisk))
            # args.center = np.average(coords[ind], axis=0)
            Mcenter = np.average(coords, axis=0)
    particletype = 'PartType0'
    ds = yt.load(args.snappath+filename)
    # close the yt file
    ds.close()
    print("file open now.")
    with h5py.File(snappath+filename, "r") as F:
        coord = F[particletype]["Coordinates"][:]
        mass = F[particletype]["Masses"][:]
        B = F[particletype]["MagneticField"][:]
        V = F[particletype]["Velocities"][:]
        Z = F[particletype]["Metallicity"][:]
        # T = F[particletype]["InternalEnergy"][:]*1e10*1.3*constants.m_p*2/3/(constants.k_b)
        ParticleIDs = F[particletype]['ParticleIDs'][:]
        ind_disk = np.where((ParticleIDs > 0) & (ParticleIDs < args.ngasdisk))[0]
        if simtype=='wind': V[ind_disk] = V[ind_disk] - np.array([0, 0, np.linalg.norm(args.V_LMC)])  # subtracting the translating component
        hsml = F[particletype]["SmoothingLength"][:]
        SFflag = False
        if "PartType4" in F.keys():
            SFtime = F["PartType4"]["StellarFormationTime"][:]*1e3 # in Myr
            SFtime_recent_ind = np.where(((i-1)*args.tstep/1e6<SFtime) & (SFtime<i*args.tstep/1e6))[0] # SFR in the last 5 Myr
            SFmass = F["PartType4"]["Masses"][:][SFtime_recent_ind]*1e10 # in Msol
            SFcoords = F["PartType4"]["Coordinates"][:][SFtime_recent_ind]
            SFflag = True
    print("file read now.")
    # rotate vector quantities by rotation matrix
    B = np.dot(rotation_matrix, B.T).T
    V = np.dot(rotation_matrix, V.T).T
    # create an array of dot product between coord and V

    print("Meshoid start now.")
    if zoom: Msize=20
    else: Msize=60
    if simtype=='isolated':
        coord = coord-Mcenter; coord = np.dot(rotation_matrix, coord.T).T
        if SFflag==True: SFcoords = SFcoords-Mcenter; SFcoords = np.dot(rotation_matrix, SFcoords.T).T
        new_Mcenter = np.array([0, 0, 0])
        res = 200
        if plane=='x': Xcenter=new_Mcenter[1]; Ycenter=new_Mcenter[2]; ax1=1; ax2=2; axLOS=0
        if plane=='y': Xcenter=new_Mcenter[0]; Ycenter=new_Mcenter[2]; ax1=0; ax2=2; axLOS=1
        if plane=='z': Xcenter=new_Mcenter[0]; Ycenter=new_Mcenter[1]; ax1=0; ax2=1; axLOS=2
        X = np.linspace(Xcenter-Msize/2, Xcenter+Msize/2, res); Y = np.linspace(Ycenter-Msize/2, Ycenter+Msize/2, res)
        X, Y = np.meshgrid(X, Y)
    # rmax = 10; res = 200; X = Y = np.linspace(-rmax, rmax, res)
    if simtype=='wind':
        new_Mcenter = np.array([0, 0, 0])
        coord = coord-Mcenter; coord = np.dot(rotation_matrix, coord.T).T
        if SFflag==True: SFcoords = SFcoords-Mcenter; SFcoords = np.dot(rotation_matrix, SFcoords.T).T
        res = 400
        if plane=='x': Xcenter=new_Mcenter[1]; Ycenter=new_Mcenter[2]; ax1=1; ax2=2; axLOS=0
        if plane=='y': Xcenter=new_Mcenter[0]; Ycenter=new_Mcenter[2]; ax1=0; ax2=2; axLOS=1
        if plane=='z': Xcenter=new_Mcenter[0]; Ycenter=new_Mcenter[1]; ax1=0; ax2=1; axLOS=2
        X = np.linspace(Xcenter-Msize/2, Xcenter+Msize/2, res); Y = np.linspace(Ycenter-Msize/2, Ycenter+Msize/2, res)
        X, Y = np.meshgrid(X, Y)

    Vr = np.sum((coord)*V, axis=1)/np.linalg.norm((coord), axis=1) # should the translating component of the velocity be subtracted for this calc?!
    quantities = []
    quantities.append(X); quantities.append(Y)
    if SFflag:
        hist, xedges, yedges = np.histogram2d(SFcoords[:,ax1], SFcoords[:,ax2], bins=res, weights=SFmass, range=[[X.min(), X.max()], [Y.min(), Y.max()]])
        SFR_dens_proj = hist/args.tstep/(Msize/res)**2 # SFR density in Msol/yr/kpc^2
        quantities.append(SFR_dens_proj.T)
        del SFR_dens_proj, hist, xedges, yedges, SFcoords, SFmass, SFtime, SFtime_recent_ind
    # plt.scatter(coord[:,0], coord[:,2], s=0.05); plt.xlim([-50, 50]); plt.ylim([-50, 50]); plt.show()
    # normalise B and V so that they are unit vectors
    B = B/np.linalg.norm(B, axis=1)[:, np.newaxis]
    V = V/np.linalg.norm(V, axis=1)[:, np.newaxis]
    # take a dot product between B and V to get the cos of the angle between them
    cos_theta = np.sum(B*V, axis=1)

    ds = yt.load(args.snappath+filename)
    rm.CorrectIons(args,ds, 100)
    ad = ds.all_data()
    T = ad['PartType0', 'grackle_temperature']
    molecular_weight = (ad['PartType0', 'molecular_weight']*u.mh).in_units('g')
    M = Meshoid(coord, cos_theta, hsml)
    cos_slice = M.Slice(M.m, center=new_Mcenter, size=Msize, res=res, plane=plane)
    Mdens = Meshoid(coord, mass, hsml)
    dens_slice = M.Slice(Mdens.Density()/molecular_weight,
                         center=new_Mcenter, size=Msize, res=res, plane=plane)
    dens_slice = dens_slice*constants.m_sol*1e10/(1e3*constants.pc)**3
    stop()
    # make a subplot of the cos_slice and dens_slice
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(cos_slice, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()], cmap='seismic')
    plt.colorbar()
    plt.title('cos(theta) between B and V')
    plt.subplot(122)
    plt.imshow(np.log10(dens_slice), origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()], cmap='CMRmap')
    plt.colorbar()
    plt.title('Density')
    plt.savefig(plotpath+'cos_theta_density_{}_{}.png'.format(plane, i))



def subplot_projection_saver(args, i, plane='x', rotation_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), simtype='isolated', zoom=True):
    sleep_sec = 6
    snappath = args.snappath; plotpath = args.plotpath; print(i, snappath, plotpath)
    filename = 'snapshot_{}.hdf5'.format(str(i).zfill(3))
    # filename = 'snapdir_{}/snapshot_{}.0.hdf5'.format(str(i).zfill(3), str(i).zfill(3))
    if args.center_method == 'density':
        with h5py.File(args.snappath+filename, 'r') as F:
            coords = F['PartType0']['Coordinates'][:]
            rho = F['PartType0']['Density'][:]
        Mcenter = coords[np.argmax(rho)]
    if args.center_method == 'velocity': 
        if args.scenario<2: Mcenter = np.array([100, 100, 100])
        else: Mcenter = np.array([100, 100, 100+np.linalg.norm(args.V_LMC)*1e5*constants.year/constants.pc/1e3*i*args.tstep])
    if args.center_method == 'com':
        with h5py.File(args.snappath+filename, 'r') as F:
            coords = F['PartType2']['Coordinates'][0:args.ngasdisk][:]
            fake_center = np.array([100, 100, 100+np.linalg.norm(args.V_LMC)*1e5*constants.year/constants.pc/1e3*i*args.tstep])
            # find indices within 20 kpc of the fake center
            ind = np.where(np.linalg.norm(coords-fake_center, axis=1)<5)[0]
            # find indices common between ind and np.arange(args.ngasdisk)
            ind = np.intersect1d(ind, np.arange(args.ngasdisk))
            # args.center = np.average(coords[ind], axis=0)
            Mcenter = np.average(coords, axis=0)
    particletype = 'PartType0'
    ds = yt.load(args.snappath+filename)
    # stop()
    rm.CorrectIons(args, ds, i)
    ad = ds.all_data()
    T = ad['PartType0', 'grackle_temperature']
    molecular_weight = (ad['PartType0', 'molecular_weight']*u.mh).in_units('g')
    # stop()
    coord = np.array(ad["PartType0", "Coordinates"])
    mass = np.array(ad["PartType0", "Masses"][:])
    B = np.array(ad["PartType0", "MagneticField"][:])
    V = np.array(ad["PartType0", "Velocities"][:])
    Z = np.array(ad["PartType0", "Metallicity"][:])
    # T = F[particletype]["InternalEnergy"][:]*1e10*1.3*constants.m_p*2/3/(constants.k_b)
    ParticleIDs = np.array(ad["PartType0", 'ParticleIDs'][:])
    ind_disk = np.where((ParticleIDs > 0) & (ParticleIDs < args.ngasdisk))[0]
    # stop()
    if simtype=='wind': V[ind_disk] = V[ind_disk] - np.array([0, 0, np.linalg.norm(args.V_LMC)])  # subtracting the translating component
    hsml = np.array(ad["PartType0", "SmoothingLength"][:])
    SFflag = False
    # if "PartType4" in F.keys():
    if i>10:
        SFtime = np.array(ad["PartType4", "StellarFormationTime"][:])*1e3 # in Myr
        SFtime_recent_ind = np.where(((i-1)*args.tstep/1e6<SFtime) & (SFtime<i*args.tstep/1e6))[0] # SFR in the last 5 Myr
        SFmass = np.array(ad["PartType4","Masses"][:])[SFtime_recent_ind]*1e10 # in Msol
        SFcoords = np.array(ad["PartType4","Coordinates"][:])[SFtime_recent_ind]
        SFflag = True

    # close the yt file
    ds.close()
    # print("file open now.")
    # with h5py.File(snappath+filename, "r") as F:
    #     coord = F[particletype]["Coordinates"][:]
    #     mass = F[particletype]["Masses"][:]
    #     B = F[particletype]["MagneticField"][:]
    #     V = F[particletype]["Velocities"][:]
    #     Z = F[particletype]["Metallicity"][:]
    #     # T = F[particletype]["InternalEnergy"][:]*1e10*1.3*constants.m_p*2/3/(constants.k_b)
    #     ParticleIDs = F[particletype]['ParticleIDs'][:]
    #     ind_disk = np.where((ParticleIDs > 0) & (ParticleIDs < args.ngasdisk))[0]
    #     # stop()
    #     if simtype=='wind': V[ind_disk] = V[ind_disk] - np.array([0, 0, np.linalg.norm(args.V_LMC)])  # subtracting the translating component
    #     hsml = F[particletype]["SmoothingLength"][:]
    #     SFflag = False
    #     if "PartType4" in F.keys():
    #         SFtime = F["PartType4"]["StellarFormationTime"][:]*1e3 # in Myr
    #         SFtime_recent_ind = np.where(((i-1)*args.tstep/1e6<SFtime) & (SFtime<i*args.tstep/1e6))[0] # SFR in the last 5 Myr
    #         SFmass = F["PartType4"]["Masses"][:][SFtime_recent_ind]*1e10 # in Msol
    #         SFcoords = F["PartType4"]["Coordinates"][:][SFtime_recent_ind]
    #         SFflag = True
    # print("file read now.")
    # rotate vector quantities by rotation matrix
    B = np.dot(rotation_matrix, B.T).T
    V = np.dot(rotation_matrix, V.T).T
    # create an array of dot product between coord and V
    # stop()
    print("Meshoid start now.")
    if zoom: Msize=20
    else: Msize=60
    if simtype=='isolated':
        coord = coord-Mcenter; coord = np.dot(rotation_matrix, coord.T).T
        if SFflag==True: SFcoords = SFcoords-Mcenter; SFcoords = np.dot(rotation_matrix, SFcoords.T).T
        new_Mcenter = np.array([0, 0, 0])
        if args.resolution == 'medres' or args.resolution == 'lowres': res = 200
        elif args.resolution == 'highres': res = 600
        else: res = 400

        if plane=='x': Xcenter=new_Mcenter[1]; Ycenter=new_Mcenter[2]; ax1=1; ax2=2; axLOS=0
        if plane=='y': Xcenter=new_Mcenter[0]; Ycenter=new_Mcenter[2]; ax1=0; ax2=2; axLOS=1
        if plane=='z': Xcenter=new_Mcenter[0]; Ycenter=new_Mcenter[1]; ax1=0; ax2=1; axLOS=2
        X = np.linspace(Xcenter-Msize/2, Xcenter+Msize/2, res); Y = np.linspace(Ycenter-Msize/2, Ycenter+Msize/2, res)
        X, Y = np.meshgrid(X, Y)
    # rmax = 10; res = 200; X = Y = np.linspace(-rmax, rmax, res)
    if simtype=='wind':
        new_Mcenter = np.array([0, 0, 0])
        coord = coord-Mcenter; coord = np.dot(rotation_matrix, coord.T).T
        if SFflag==True: SFcoords = SFcoords-Mcenter; SFcoords = np.dot(rotation_matrix, SFcoords.T).T
        if args.resolution == 'medres': res = 200
        elif args.resolution == 'lowres': res = 50
        elif args.resolution == 'medhighres': res = 400
        else: res = 100; print("Resolution not specified, setting to 100")
        if plane=='x': Xcenter=new_Mcenter[1]; Ycenter=new_Mcenter[2]; ax1=1; ax2=2; axLOS=0
        if plane=='y': Xcenter=new_Mcenter[0]; Ycenter=new_Mcenter[2]; ax1=0; ax2=2; axLOS=1
        if plane=='z': Xcenter=new_Mcenter[0]; Ycenter=new_Mcenter[1]; ax1=0; ax2=1; axLOS=2
        X = np.linspace(Xcenter-Msize/2, Xcenter+Msize/2, res); Y = np.linspace(Ycenter-Msize/2, Ycenter+Msize/2, res)
        X, Y = np.meshgrid(X, Y)

    Vr = np.sum((coord)*V, axis=1)/np.linalg.norm((coord), axis=1) # should the translating component of the velocity be subtracted for this calc?!
    quantities = []
    quantities.append(X); quantities.append(Y)
    if SFflag:
        hist, xedges, yedges = np.histogram2d(SFcoords[:,ax1], SFcoords[:,ax2], bins=res, weights=SFmass, range=[[X.min(), X.max()], [Y.min(), Y.max()]])
        SFR_dens_proj = hist/args.tstep/(Msize/res)**2 # SFR density in Msol/yr/kpc^2
        quantities.append(SFR_dens_proj.T)
        del SFR_dens_proj, hist, xedges, yedges, SFcoords, SFmass, SFtime, SFtime_recent_ind
    # plt.scatter(coord[:,0], coord[:,2], s=0.05); plt.xlim([-50, 50]); plt.ylim([-50, 50]); plt.show()
    # Keeping only the data along the LOS that is required for the projection as Meshoid does not allow to select the LOS length
    LOS = 30
    ind_proj = np.where((np.abs(coord[:, ax1]) <= X.max()) & (np.abs(coord[:, ax2]) <= Y.max()) & (np.abs(coord[:, axLOS])<=LOS/2))[0]
    M = Meshoid(coord[ind_proj], mass[ind_proj], hsml[ind_proj])
    molecular_weight = molecular_weight[ind_proj]
    rho = M.Density()*constants.m_sol*1e10/(1e3*constants.pc)**3/molecular_weight  # in particles per cubic cm
    dens_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,ax1], coord[ind_proj][:,ax2], bins=res, weights=M.Density(), range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    dens_proj_pp, xedges, yedges = np.histogram2d(coord[ind_proj][:,ax1], coord[ind_proj][:,ax2], bins=res, weights=rho, range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    num_counts, xedges, yedges = np.histogram2d(coord[ind_proj][:,ax1], coord[ind_proj][:,ax2], bins=res, range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    num_counts[num_counts==0] = 1
    dens_proj = dens_proj/num_counts
    dens_proj_pp = dens_proj_pp/num_counts
    # plt.figure(); plt.imshow(dens_proj); plt.show()
    quantities.append(dens_proj_pp.T) # saved in the  units of free particles per cc

    # make dens_proj = 0 to dens_proj = 1 to avoid division by zero
    dens_proj[dens_proj==0] = 1
    BXdens_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,ax1], coord[ind_proj][:,ax2], bins=res, weights=M.Density()*B[ind_proj][:, ax1], range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    BX_proj = BXdens_proj/num_counts/dens_proj
    quantities.append(BX_proj.T)

    BYdens_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,ax1], coord[ind_proj][:,ax2], bins=res, weights=M.Density()*B[ind_proj][:, ax2], range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    BY_proj = BYdens_proj/num_counts/dens_proj
    quantities.append(BY_proj.T)
    
    Bdens_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,ax1], coord[ind_proj][:,ax2], bins=res, weights=M.Density()*np.linalg.norm(B[ind_proj], axis=1), range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    B_proj = Bdens_proj/num_counts/dens_proj
    quantities.append(B_proj.T)

    Vrdens_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,ax1], coord[ind_proj][:,ax2], bins=res, weights=M.Density()*Vr[ind_proj], range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    Vr_proj = Vrdens_proj/num_counts/dens_proj
    quantities.append(Vr_proj.T)    
    
    Zdens_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,ax1], coord[ind_proj][:,ax2], bins=res, weights=M.Density()*Z[ind_proj], range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    Z_proj = Zdens_proj/num_counts/dens_proj
    quantities.append(Z_proj.T)
    Tdens_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,ax1], coord[ind_proj][:,ax2], bins=res, weights=M.Density()*T[ind_proj], range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    T_proj = Tdens_proj/num_counts/dens_proj
    quantities.append(T_proj.T)
    print("Tmin = {}".format(T_proj[25, 25]))

    VXdens_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,ax1], coord[ind_proj][:,ax2], bins=res, weights=M.Density()*V[ind_proj][:, ax1], range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    VX_proj = VXdens_proj/num_counts/dens_proj
    quantities.append(VX_proj.T)

    VYdens_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,ax1], coord[ind_proj][:,ax2], bins=res, weights=M.Density()*V[ind_proj][:, ax2], range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    VY_proj = VYdens_proj/num_counts/dens_proj
    quantities.append(VY_proj.T)
    del M, dens_proj, BXdens_proj, BX_proj, BYdens_proj, BY_proj, Bdens_proj, B_proj, Vrdens_proj, Vr_proj, Zdens_proj, Z_proj, Tdens_proj, T_proj, VXdens_proj, VX_proj, VYdens_proj, VY_proj
    
    # append the center of the domain
    quantities.append(Mcenter)
    print("Meshoid done and deleted now.")
    # save the list using np.savez
    if (rotation_matrix==np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).all():
        np.savez(plotpath+'quantities_plane{}_{}_projected_c{}.npz'.format(plane, i, args.center_method), np.array(quantities))
        print("saved unrotated projected quantities for {}".format(filename))
    else:
        np.savez(plotpath+'quantities_plane{}_{}_rotated_projected_c{}.npz'.format(plane, i, args.center_method), np.array(quantities))
        print("saved rotated projected quantities for {}".format(filename))
    del quantities, coord, mass, hsml, B, V, Z, T, Vr, X, Y
    print("saved and deleted quantities for {}".format(filename))
    print('sleep: %d sec.' % (sleep_sec))
    sleep(sleep_sec)

def subplot_slice_saver_wrapper(args_dict):
    try:
        return subplot_slice_saver(**args_dict)
    except Exception as e:
        # Log the exception instead of raising it
        print("Exception in worker process:", e)

def subplot_projection_saver_wrapper(args_dict):
    try:
        return subplot_projection_saver(**args_dict)
    except Exception as e:
        # Log the exception instead of raising it
        print("Exception in worker process:", e)

def subplot_plotter(args, i, plane='x', rotated=False, slice=True):
    print(i)
    plotpath = args.plotpath
    plt.switch_backend('agg')
    # read the saved numpy zip array
    if slice:
        if rotated: quantities = np.load(plotpath+'quantities_plane{}_{}_rotated_sliced_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
        else: quantities = np.load(plotpath+'quantities_plane{}_{}_sliced_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
    else:
        if rotated: quantities = np.load(plotpath+'quantities_plane{}_{}_rotated_projected_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
        else: quantities = np.load(plotpath+'quantities_plane{}_{}_projected_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
    if quantities['arr_0'].shape[0]==12:
        X, Y, dens_slice, BX_slice, BY_slice, B_slice, V_slice, Z_slice, T_slice, VX_slice, VY_slice, Mcenter = quantities['arr_0']
        SFR_proj = np.zeros_like(dens_slice)
    else:
        X, Y, SFR_proj, dens_slice, BX_slice, BY_slice, B_slice, V_slice, Z_slice, T_slice, VX_slice, VY_slice, Mcenter = quantities['arr_0']

    # make 4 subplots in grid of 3 by 2
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    cmapdens = cmr.rainforest_r
    cmapB = cmr.cosmic
    cmapVr = cmr.prinsenvlag_r
    cmapZ = cmr.cosmic
    cmapT = cmr.ember
    # dens_slice[dens_slice==0] = 1e-10
    p0 = axs[0, 0].pcolormesh(X, Y, dens_slice, norm=colors.LogNorm(vmin=1e-4,vmax=1e1), cmap=cmapdens)
    axs[0, 0].set_aspect('equal')
    axs[0, 0].set_title(r'$\rm Density~(m_p~cm^{-3})$')
    p1 = axs[0, 1].pcolormesh(X, Y, B_slice*1e6, norm=colors.LogNorm(vmin=1e-1,vmax=10), cmap=cmapB)
    axs[0, 1].streamplot(X, Y, BX_slice, BY_slice, color='y', linewidth=0.5, density=2, arrowsize=1.0, arrowstyle='->')
    axs[0, 1].set_aspect('equal', adjustable='box')
    axs[0, 1].set_xlim([X.min(), X.max()])
    axs[0, 1].set_ylim([Y.min(), Y.max()])
    axs[0, 1].set_title(r'$\rm B~(\mathit{\mu}G)$')
    p2 = axs[1, 0].pcolormesh(X, Y, V_slice, norm=colors.Normalize(vmin=-100,vmax=100), cmap=cmapVr)
    axs[1, 0].set_aspect('equal')
    axs[1, 0].set_title('Vr (km/s)')
    axs[1, 0].streamplot(X, Y, VX_slice, VY_slice, color='k', linewidth=0.5, density=2, arrowsize=1.0, arrowstyle='->')
    axs[1, 0].set_aspect('equal', adjustable='box')
    axs[1, 0].set_xlim([X.min(), X.max()])
    axs[1, 0].set_ylim([Y.min(), Y.max()])
    p3 = axs[1, 1].pcolormesh(X, Y, Z_slice, norm=colors.LogNorm(vmin=10**(-2.2),vmax=10**(-1.8)), cmap=cmapZ)
    axs[1, 1].set_aspect('equal')
    axs[1, 1].set_title('Z')
    p4 = axs[2, 0].pcolormesh(X, Y, T_slice, norm=colors.LogNorm(vmin=1e2,vmax=1e5), cmap=cmapT)
    axs[2, 0].set_aspect('equal')
    axs[2, 0].set_title('T (K)')
    p5 = axs[2, 1].pcolormesh(X, Y, SFR_proj, norm=colors.LogNorm(vmin=1e-5,vmax=1e0))
    axs[2, 1].set_aspect('equal')
    axs[2, 1].set_title('SFR (Msol/yr)')
    fig.colorbar(p0,label="Density"); fig.colorbar(p1,label="B (uG)"); fig.colorbar(p2,label="V"); fig.colorbar(p3,label="Z")
    fig.colorbar(p4,label="T (K)"); fig.colorbar(p5,label="SFR Density (Msol/yr/kpc**2)")
    fig.suptitle("Center={} kpc, Time={} Myr".format(np.round(Mcenter, 2), i*args.tstep/1e6))
    for ax in axs.flat:
        ax.set(xlabel='X (kpc)', ylabel='Y (kpc)')
        # ax.label_outer()
    if slice:
        if rotated: plt.savefig(plotpath+'{}_plane{}_rotated_sliced_c{}.png'.format(i, plane, args.center_method), dpi=300)
        else: plt.savefig(plotpath+'{}_plane{}_sliced_c{}.png'.format(i, plane, args.center_method), dpi=300)
    else:
        if rotated: plt.savefig(plotpath+'{}_plane{}_rotated_projected_c{}.png'.format(i, plane, args.center_method), dpi=300)
        else: plt.savefig(plotpath+'{}_plane{}_projected_c{}.png'.format(i, plane, args.center_method), dpi=300)
    plt.close()

def subplot_movie_plotter(args, i, plane='z', rotated=True, slice=False):
    print(i)
    plotpath = args.plotpath
    plt.switch_backend('agg')
    # read the saved numpy zip array
    if slice:
        if rotated: quantities = np.load(plotpath+'quantities_plane{}_{}_rotated_sliced_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
        else: quantities = np.load(plotpath+'quantities_plane{}_{}_sliced_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
    else:
        if rotated: quantities = np.load(plotpath+'quantities_plane{}_{}_rotated_projected_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
        else: quantities = np.load(plotpath+'quantities_plane{}_{}_projected_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
    if quantities['arr_0'].shape[0]==12:
        X, Y, dens_slice, BX_slice, BY_slice, B_slice, V_slice, Z_slice, T_slice, VX_slice, VY_slice, Mcenter = quantities['arr_0']
        SFR_proj = np.zeros_like(dens_slice)
    else:
        X, Y, SFR_proj, dens_slice, BX_slice, BY_slice, B_slice, V_slice, Z_slice, T_slice, VX_slice, VY_slice, Mcenter = quantities['arr_0']

    # make 4 subplots in grid of 3 by 2
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    cmapdens = cmr.rainforest_r
    cmapB = cmr.cosmic
    # dens_slice[dens_slice==0] = 1e-10
    p0 = axs[0].pcolormesh(X, Y, dens_slice, norm=colors.LogNorm(vmin=1e-4,vmax=1e1), cmap=cmapdens)
    axs[0].set_aspect('equal')
    axs[0].set_title(r'$\rm Density~(m_p~cm^{-3})$')
    p1 = axs[1].pcolormesh(X, Y, B_slice*1e6, norm=colors.LogNorm(vmin=1e-1,vmax=10), cmap=cmapB)
    axs[1].streamplot(X, Y, BX_slice, BY_slice, color='y', linewidth=0.5, density=2, arrowsize=1.0, arrowstyle='->')
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_xlim([X.min(), X.max()])
    axs[1].set_ylim([Y.min(), Y.max()])
    axs[1].set_title(r'$\rm B~(\mathit{\mu}G)$')
    # fig.colorbar(p0,label="Density"); fig.colorbar(p1,label="B (uG)")
    fig.suptitle("Center={} kpc, Time={} Myr".format(np.round(Mcenter, 2), i*args.tstep/1e6))
    for ax in axs.flat:
        ax.set(xlabel='X (kpc)', ylabel='Y (kpc)')
        # ax.label_outer()
    if slice:
        if rotated: plt.savefig(plotpath+'movie_{}_plane{}_rotated_sliced_c{}.png'.format(i, plane, args.center_method), dpi=300)
        else: plt.savefig(plotpath+'movie_{}_plane{}_sliced_c{}.png'.format(i, plane, args.center_method), dpi=300)
    else:
        if rotated: plt.savefig(plotpath+'movie_{}_plane{}_rotated_projected_c{}.png'.format(i, plane, args.center_method), dpi=300)
        else: plt.savefig(plotpath+'movie_{}_plane{}_projected_c{}.png'.format(i, plane, args.center_method), dpi=300)
    plt.close()

def azimuthal_bin(RM, points, num_bins=20, key='average'): # key could be average or mag_average
    x_coords = points[:, 0]  # X-coordinates
    y_coords = points[:, 1]  # Y-coordinates
    # Compute angles in degrees (0° to 360°)
    angles = np.arctan2(y_coords, x_coords) * 180 / np.pi  # [-180°, 180°]
    angles = (angles + 360) % 360  # Convert to [0°, 360°)
    # Bin the angles
    bin_edges = np.linspace(0, 360, num_bins + 1)  # Create bin edges
    # find the bin index for each angle
    bin_indices = np.digitize(angles, bin_edges) - 1  # -1 to make it zero-indexed
    # Compute the average RM in each bin
    binned_RM = np.zeros(num_bins)
    for i in range(num_bins):
        that_bin_indices = np.where(bin_indices == i)[0]
        if len(that_bin_indices) > 0:
            if key == 'average':
                binned_RM[i] = np.mean(RM[that_bin_indices])
            elif key == 'mag_average':
                binned_RM[i] = np.mean(np.abs(RM[that_bin_indices]))
            else:
                raise ValueError("Invalid key. Use 'average' or 'mag_average'.")
        else:
            binned_RM[i] = np.nan
    # Compute the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, binned_RM

def paper_plot_RMaz(args, key='average'): # key could be average or mag_average
    # load the RM data
    args.Tmin = 1e1; args.Tmax = 1e7; args.snappath = '/scratch/jh2/hs9158/results/LMC_run_medhighres_wind_scenario_0_turb6_order2/'
    RMdata_isolated = la.read_RM_map(args, 100, 500, 'RM', method='new'); EMdata_isolated = la.read_RM_map(args, 100, 500, 'EM', method='new')
    points_isolated = RMdata_isolated[0]
    RM_isolated = RMdata_isolated[1]
    EM_isolated = EMdata_isolated[1]
    indRM_isolated = np.where(EM_isolated<100)[0]
    points_isolated = points_isolated[indRM_isolated]; RM_isolated = RM_isolated[indRM_isolated]
    args.snappath = '/scratch/jh2/hs9158/results/LMC_run_medhighres_wind_scenario_3_turb6_order2/'
    RMdata_wind = la.read_RM_map(args, 100, 500, 'RM', method='new'); EMdata_wind = la.read_RM_map(args, 100, 500, 'EM', method='new')
    points_wind = RMdata_wind[0]; RM_wind = RMdata_wind[1]; EM_wind = EMdata_wind[1]
    indRM_wind = np.where(EM_wind<100)[0]
    points_wind = points_wind[indRM_wind]; RM_wind = RM_wind[indRM_wind]
    angles_isolated, azRM_isolated = azimuthal_bin(RM_isolated, points_isolated, num_bins=60, key=key) # key could be average or mag_average
    angles_wind, azRM_wind = azimuthal_bin(RM_wind, points_wind, num_bins=60, key=key) # key could be average or mag_average
    # plot the azimuthal RM as a function of angle for both isolated and wind
    plt.figure(figsize=(6, 6))
    plt.plot(angles_isolated, azRM_isolated, label=r'\texttt(I-2-6-M)', color='k', linewidth=3)
    plt.plot(angles_wind, azRM_wind, label=r'\texttt(W-2-6-M)', color='k', linewidth=3, linestyle='--')
    plt.xlabel(r'$\phi$ (degrees)')
    plt.ylabel(r'$\langle \rm RM \rangle_{\phi}$ (rad/m$^2$)')
    # shade 4 regions of 90 degrees each on this plot
    for i in range(4):
        plt.axvspan(i*90, (i+1)*90, color='grey', alpha=0.1+0.1*i)
        # add vertical lines at 0, 90, 180, 270 degrees
        plt.axvline(i*90, color='grey', linestyle='--', linewidth=0.5)
        # Add text to each region where the coordinates define the center of the text prompt
        if i==0:plt.text(45, -75, 'Top-Right', fontsize=12, ha='center', va='center', rotation=0)
        if i==1:plt.text(135, -75, 'Top-Left', fontsize=12, ha='center', va='center', rotation=0)
        if i==2:plt.text(225, -75, 'Bottom-Left', fontsize=12, ha='center', va='center', rotation=0)
        if i==3:plt.text(315, -75, 'Bottom-Right', fontsize=12, ha='center', va='center', rotation=0)
    # add a horizontal line at 0
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.xlim([0, 360])
    if key=='average':
        plt.ylim([-100, 100])
    plt.legend()
    plt.savefig('/scratch/jh2/hs9158/results/paper_plots/azimuthal_RM_{}.pdf'.format(key), dpi=300)
    plt.close()
    stop()

def make_pretty_plots(args, rotation_matrix):
    i = args.snapnum
    if os.path.exists(args.snappath+"snapshot_{}_processed_quantities_T{}.npy".format(str(i).zfill(3), args.TRe_key)):
        print("Processed quantities already exist for this snapshot.")
        arr = np.load(args.snappath+"snapshot_{}_processed_quantities_T{}.npy".format(str(i).zfill(3), args.TRe_key), allow_pickle=True)
        arrdata = arr.item()
        # stop()
        ne_corrected_photoionized = arrdata['ne_corrected_photoionized']
        # ne = arrdata['ne']
        T_photo = arrdata['grackle_temperature_photo']
    ds = yt.load(args.snappath+args.filename)
    ad = ds.all_data()
    ad_rot = rm.RotateGalaxy(args, ad, rotation_matrix, center=args.center)
    BLOS = ad_rot['PartType0', 'MagneticField'][:, 2]*1e6
    vol = ad_rot['PartType0', 'SmoothingLength']**3
    coord = ad_rot['PartType0', 'Coordinates']
    coord = np.array(coord)-args.center
    # create a projection of the ne_corrected_photoionized and BLOS
    LOS = 10
    Msize = 10
    res = 400
    X = np.linspace(0-Msize/2, 0+Msize/2, res); Y = np.linspace(0-Msize/2, 0+Msize/2, res)
    X, Y = np.meshgrid(X, Y)
    ind_proj = np.where((np.abs(coord[:, 0]) <= X.max()) & (np.abs(coord[:, 1]) <= Y.max()) & (np.abs(coord[:, 2])<=LOS/2))[0]
    vol_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,0], coord[ind_proj][:,1], bins=res, weights=np.array(vol[ind_proj]), range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    num_counts, xedges, yedges = np.histogram2d(coord[ind_proj][:,0], coord[ind_proj][:,1], bins=res, range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    num_counts[num_counts==0] = 1
    
    BLOSvol_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,0], coord[ind_proj][:,1], bins=res, weights=np.array(vol[ind_proj])*np.array(BLOS[ind_proj]), range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    BLOS_proj = BLOSvol_proj/num_counts/vol_proj
    nevol_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,0], coord[ind_proj][:,1], bins=res, weights=np.array(vol[ind_proj])*np.array(ne_corrected_photoionized[ind_proj]), range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    ne_proj = nevol_proj/num_counts/vol_proj

    # load the RM data
    args.Tmin = 1e1; args.Tmax = 1e7
    RMdata = la.read_RM_map(args, args.snapnum, 500, 'RM', method='new')
    points = RMdata[0]
    RM = RMdata[1]
    indRM = np.where(np.abs(RM)<250)[0]
    points = points[indRM]
    RM = RM[indRM]
    plt.figure(figsize=(5,5))
    y, x = rm.histogram(RM, 40)
    filepath = "/scratch/jh2/hs9158/results/Gaensler2005Data/"
    filename = "rm_bkg_list.txt"
    table = ga.txt2npy(filepath+filename)
    ybkg, xbkg = rm.histogram(table['RM'], 20)
    # stop()
    plt.figure(figsize=(5,5))
    plt.plot(x, y, label='Simulated', linewidth=3)
    plt.plot(xbkg, ybkg, label='Gaensler 2005', linewidth=3)
    plt.xlabel(r'$\rm RM~(rad/m^2)$')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig(args.plotpath+'RM_hist.png', dpi=300)
    # stop()
    color = np.repeat('b', len(points))
    color[RM<0] = 'r'
    # stop()
    fig = plt.figure(figsize=(10, 5), constrained_layout=False)

    # First subplot
    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(BLOS_proj.T, origin='lower',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    interpolation='nearest', cmap='PuOr', 
                    norm=colors.SymLogNorm(linthresh=1e-3, linscale=1.0,
                                        vmin=-0.2, vmax=0.2), alpha=0.5)
    axins = inset_axes(ax1, width="90%", height="5%", loc='lower center', bbox_to_anchor=(0, -0.2, 1, 1), bbox_transform=ax1.transAxes, borderpad=0,)
    cbar1 = fig.colorbar(im1, cax=axins, orientation='horizontal')
    ax1.scatter(points[:, 0]/1e3, points[:,1]/1e3, c=color, 
                s=np.abs(RM)/2, alpha=1, edgecolors='k')
    ax1.set_ylabel('Y (kpc)')
    ax1.set_xlabel('X (kpc)')

    # Second subplot
    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(ne_proj.T, origin='lower',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    interpolation='nearest', cmap='viridis', 
                    norm=colors.LogNorm(vmin=1e-4,vmax=1e-1))
    ax2.scatter(points[:, 0]/1e3, points[:,1]/1e3, c=color, 
                s=np.abs(RM)/2, alpha=1, edgecolors='k')
    ax2.set_xlabel('X (kpc)')
    ax2.yaxis.set_ticklabels([])

    # Adjust subplot spacing
    plt.subplots_adjust(wspace=0, hspace=0)

    # Add colorbars at the bottom
    cbar1.set_label(r'$B_{\rm LOS}~{\rm (\mu G)}$')

    axins = inset_axes(ax2, width="90%", height="5%", loc='lower center', bbox_to_anchor=(0, -0.2, 1, 1), bbox_transform=ax2.transAxes, borderpad=0,)
    cbar2 = fig.colorbar(im2, cax=axins, orientation='horizontal')
    cbar2.set_label(r'$n_{\rm e}~{\rm (cm^{-3})}$')

    plt.savefig(args.plotpath+'pretty_plot_{}.png'.format(i), dpi=300, bbox_inches='tight')
    plt.savefig(args.plotpath+'pretty_plot_{}.pdf'.format(i), dpi=300, bbox_inches='tight')
    plt.show()

    stop()

def make_RM_plot_paper(args, rotation_matrix):
    i = args.snapnum
    if os.path.exists(args.snappath+"snapshot_{}_processed_quantities_T{}.npy".format(str(i).zfill(3), args.TRe_key)):
        print("Processed quantities already exist for this snapshot.")
        arr = np.load(args.snappath+"snapshot_{}_processed_quantities_T{}.npy".format(str(i).zfill(3), args.TRe_key), allow_pickle=True)
        arrdata = arr.item()
        # stop()
        ne_corrected_photoionized = arrdata['ne_corrected_photoionized']
        # ne = arrdata['ne']
        T_photo = arrdata['grackle_temperature_photo']
    ds = yt.load(args.snappath+args.filename)
    ad = ds.all_data()
    ad_rot = rm.RotateGalaxy(args, ad, rotation_matrix, center=args.center)
    BLOS = ad_rot['PartType0', 'MagneticField'][:, 2]*1e6
    vol = ad_rot['PartType0', 'SmoothingLength']**3
    coord = ad_rot['PartType0', 'Coordinates']
    coord = np.array(coord)-args.center
    # create a projection of the ne_corrected_photoionized and BLOS
    LOS = 10
    Msize = 10
    res = 400
    X = np.linspace(0-Msize/2, 0+Msize/2, res); Y = np.linspace(0-Msize/2, 0+Msize/2, res)
    X, Y = np.meshgrid(X, Y)
    ind_proj = np.where((np.abs(coord[:, 0]) <= X.max()) & (np.abs(coord[:, 1]) <= Y.max()) & (np.abs(coord[:, 2])<=LOS/2))[0]
    vol_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,0], coord[ind_proj][:,1], bins=res, weights=np.array(vol[ind_proj]), range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    num_counts, xedges, yedges = np.histogram2d(coord[ind_proj][:,0], coord[ind_proj][:,1], bins=res, range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    num_counts[num_counts==0] = 1
    
    BLOSvol_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,0], coord[ind_proj][:,1], bins=res, weights=np.array(vol[ind_proj])*np.array(BLOS[ind_proj]), range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    BLOS_proj = BLOSvol_proj/num_counts/vol_proj
    nevol_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,0], coord[ind_proj][:,1], bins=res, weights=np.array(vol[ind_proj])*np.array(ne_corrected_photoionized[ind_proj]), range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    ne_proj = nevol_proj/num_counts/vol_proj

    # load the RM data
    args.Tmin = 1e1; args.Tmax = 1e7
    RMdata = la.read_RM_map(args, args.snapnum, 200, 'RM', method='new')
    # read EM data
    EMdata = la.read_RM_map(args, args.snapnum, 200, 'EM', method='new')
    # stop()
    points = RMdata[0]
    RM = RMdata[1]
    EM = EMdata[1]
    indRM = np.where(np.abs(EM)<100)[0]
    points = points[indRM]
    RM = RM[indRM]
    y, x = rm.histogram(RM, 20)
    filepath = "/scratch/jh2/hs9158/results/Gaensler2005Data/"
    filename = "rm_bkg_list.txt"
    table = ga.txt2npy(filepath+filename)
    ybkg, xbkg = rm.histogram(table['RM'], 20)
    # stop()

    # stop()

    # stop()
    scale = 1.3
    fig = plt.figure(figsize=(13*scale, 5*scale))
    # Second subplot
    ax2 = fig.add_subplot(1, 3, 1)
    alpha = np.radians(+34.7-180); beta = np.radians(0); gamma = np.radians(90-(180-139.9))
    a = np.cos(alpha); b = np.cos(beta); c = np.cos(gamma)
    d = np.sin(alpha); e = np.sin(beta); f = np.sin(gamma)    
    R_obs_new = np.array([[b*c, d*e*c-a*f, a*e*c+d*f], [b*f, d*e*f+a*c, a*e*f-d*c], [-e, d*b, a*b]])
    ra_center, dec_center, node_rot, node_line = ga.LMC_center(R_obs_new)
    sign = table["sign"]
    # make a scatter plot of the RMs
    color = np.repeat('b', len(table["RM"]))
    color[table["RM"]<0] = 'r'

    RM_scale = 30
    ax2.scatter(table["x_kpc"], table["y_kpc"], c=color, s=np.sqrt(RM_scale*np.abs(table["RM"])))
    ax2.scatter(0, 0, c='k', marker='x', s=15)
    # make a line passing through the center of the LMC pointing towards node_rot vector
    # stop()
    ax2.set_xlim([-5, 5])
    ax2.set_ylim([-5, 5])
    x_arr = np.linspace(-5, 5, 100)
    y_arr = node_line(x_arr, [0, 0])
    ax2.plot(x_arr, y_arr, c='k', linestyle='--')
    ax2.set_xlabel("X (kpc)")
    ax2.set_ylabel("Y (kpc)")
    ax2.invert_xaxis()
    ax2.set_title("LMC RMs")
    
    ax2.set_aspect('equal', adjustable='box')
    # First subplot
    ax1 = fig.add_subplot(1, 3, 2)
    color = np.repeat('b', len(points))
    color[RM<0] = 'r'
    im1 = ax1.imshow(BLOS_proj.T, origin='lower',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    interpolation='nearest', cmap='PuOr', 
                    norm=colors.SymLogNorm(linthresh=1e-3, linscale=1.0,
                                        vmin=-0.2, vmax=0.2), alpha=0.5)
    # axins = inset_axes(ax1, width="90%", height="5%", loc='lower center', bbox_to_anchor=(0, -0.2, 1, 1), bbox_transform=ax1.transAxes, borderpad=0,)

    # stop()
    ax1.scatter(points[:, 0]/1e3, points[:,1]/1e3, c=color, 
                s=np.sqrt(RM_scale*np.abs(RM)), alpha=1)
    # ax1.set_ylabel('Y (kpc)')
    ax1.set_xlabel('X (kpc)')
    ax1.plot(-1*x_arr, y_arr, c='k', linestyle='--')
    ax1.scatter(0, 0, c='k', marker='x', s=15)
    ax1.set_title("Simulated True RMs")
    ax1.set_aspect('equal', adjustable='box')
    # Adjust subplot spacing
    plt.subplots_adjust(wspace=0, hspace=0)
    ax1.set_yticklabels([])

    ax3 = fig.add_subplot(1, 3, 3)
    np.random.seed(0)
    e_RM = rma.error_KDE().resample(size=int(len(RM)*1.1), seed=i)[0]
    # consider only points with positive RM error equal to the size of RM
    ind = np.where(e_RM>0)[0]; e_RM = e_RM[ind][0:len(RM)]
    # stop()
    # Generate samples
    samples = np.random.normal(loc=RM, scale=e_RM, size=len(RM))
    color = np.repeat('b', len(points))
    color[samples<0] = 'r'
    im3 = ax3.imshow(BLOS_proj.T, origin='lower',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    interpolation='nearest', cmap='PuOr', 
                    norm=colors.SymLogNorm(linthresh=1e-3, linscale=1.0,
                                        vmin=-0.2, vmax=0.2), alpha=0.5)
    # axins = inset_axes(ax1, width="90%", height="5%", loc='lower center', bbox_to_anchor=(0, -0.2, 1, 1), bbox_transform=ax1.transAxes, borderpad=0,)
    # cbar1 = fig.colorbar(im1, orientation='horizontal', shrink=0.7)
    # cbar1.set_label(r'$B_{\rm LOS}~{\rm (\mu G)}$')
    # stop()
    ax3.scatter(points[:, 0]/1e3, points[:,1]/1e3, c=color, 
                s=np.sqrt(RM_scale*np.abs(samples)), alpha=1)
    # ax1.set_ylabel('Y (kpc)')
    ax3.set_xlabel('X (kpc)')
    ax3.plot(-1*x_arr, y_arr, c='k', linestyle='--')
    ax3.set_title("Mock Observed RMs")
    ax3.scatter(0, 0, c='k', marker='x', s=15)
    ax3.set_aspect('equal', adjustable='box')
    ax3.set_yticklabels([])

    cbar1 = fig.colorbar(im3, orientation='vertical', shrink=0.75)
    cbar1.set_label(r'$B_{\rm LOS}~{\rm (\mu G)}$')


    plt.savefig('/scratch/jh2/hs9158/results/paper_plots/RM_plot.pdf', dpi=100, bbox_inches='tight')
    plt.savefig('/scratch/jh2/hs9158/results/paper_plots/RM_plot.png', dpi=100, bbox_inches='tight')

    plt.figure(figsize=(6,6))
    plt.plot(x, y, label='Simulated', linewidth=3)
    plt.plot(xbkg, ybkg, label='Gaensler et al. 2005', linewidth=3)
    plt.ylim([0, 1.2*np.max([np.max(y), np.max(ybkg)])])
    plt.xlabel(r'$\rm RM~(rad/m^2)$')
    plt.ylabel('Probability Density')
    plt.legend(loc='best')

    # plt.set_aspect('auto')  # This allows different scales while maintaining panel size
    plt.savefig('/scratch/jh2/hs9158/results/paper_plots/RM_compare.pdf', dpi=100, bbox_inches='tight')

    plt.show()




def make_pressure_balance_plots(args, plane='y'):
    snappath = args.snappath; plotpath = args.plotpath; filename = 'snapshot_{}.hdf5'.format(str(args.snapnum).zfill(3))
    particletype = "PartType0"
    if plane=='x': ax1=1; ax2=2; axLOS=0
    if plane=='y': ax1=0; ax2=2; axLOS=1
    if plane=='z': ax1=0; ax2=1; axLOS=2
    res = 200
    ds = yt.load(args.snappath+args.filename)
    rm.CorrectIons(args,ds)
    ad = ds.all_data()
    T = ad['PartType0', 'grackle_temperature']
    rho = ad['PartType0', 'density'].in_units('g/cm**3')
    molecular_weight = (ad['PartType0', 'molecular_weight']*u.mh).in_units('g')
    # close the yt file
    ds.close()
    with h5py.File(snappath+filename, "r") as F:
        coord = F[particletype]["Coordinates"][:]
        mass = F[particletype]["Masses"][:]
        B = F[particletype]["MagneticField"][:]
        V = F[particletype]["Velocities"][:]
        ParticleIDs = F[particletype]['ParticleIDs'][:]
        rho_gizmo = F[particletype]['Density'][:]

    # convert rho_gizmo from units of 1e10*msol/kpc**3 to g/cm**3
    rho_gizmo = rho_gizmo*1e10*constants.m_sol/(1e3*constants.pc)**3
    thermal_pressure = rho*T*u.kboltz_cgs/molecular_weight
    magnetic_pressure = np.linalg.norm(B, axis=1)**2/(8*np.pi)
    total_pressure = np.array(thermal_pressure) + magnetic_pressure
    # create a histogram
    pres_hist, xedges, yedges = np.histogram2d(coord[:,ax1], coord[:,ax2], bins=res, weights=total_pressure,
                                                range=[[coord[:,ax1].min(), coord[:,ax1].max()], [coord[:,ax2].min(), coord[:,ax2].max()]])
    beta_pres_hist, xedges, yedges = np.histogram2d(coord[:,ax1], coord[:,ax2], bins=res, weights=thermal_pressure/magnetic_pressure,
                                                        range=[[coord[:,ax1].min(), coord[:,ax1].max()], [coord[:,ax2].min(), coord[:,ax2].max()]])
    T_hist, xedges, yedges = np.histogram2d(coord[:,ax1], coord[:,ax2], bins=res, weights=T,
                                                range=[[coord[:,ax1].min(), coord[:,ax1].max()], [coord[:,ax2].min(), coord[:,ax2].max()]])
    rho_hist, xedges, yedges = np.histogram2d(coord[:,ax1], coord[:,ax2], bins=res, weights=rho,
                                                range=[[coord[:,ax1].min(), coord[:,ax1].max()], [coord[:,ax2].min(), coord[:,ax2].max()]])
    rho_hist_gizmo, xedges, yedges = np.histogram2d(coord[:,ax1], coord[:,ax2], bins=res, weights=rho_gizmo,
                                                range=[[coord[:,ax1].min(), coord[:,ax1].max()], [coord[:,ax2].min(), coord[:,ax2].max()]])
    num_counts, xedges, yedges = np.histogram2d(coord[:,ax1], coord[:,ax2], bins=res,
                                                range=[[coord[:,ax1].min(), coord[:,ax1].max()], [coord[:,ax2].min(), coord[:,ax2].max()]])
    num_counts[num_counts==0] = 1
    pres_hist = pres_hist/num_counts
    beta_pres_hist = beta_pres_hist/num_counts
    T_hist = T_hist/num_counts
    rho_hist = rho_hist/num_counts
    rho_hist_gizmo = rho_hist_gizmo/num_counts
    X, Y = np.meshgrid(xedges, yedges)
    # plot two subplots as heatmaps of all particles, one for total pressure and one for ratio of thermal to magnetic pressure
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    p0 = axs[0].pcolormesh(X, Y, pres_hist.T, norm=colors.LogNorm(vmin=1e-14,vmax=1e-13), cmap='viridis')
    axs[0].set_title('Total Pressure (dyne/cm**2)')
    fig.colorbar(p0, ax=axs[0], label='Pressure (dyne/cm**2)')
    p1 = axs[1].pcolormesh(X, Y, beta_pres_hist.T, norm=colors.LogNorm(vmin=1e-1, vmax=1e1))
    axs[1].set_title('Plasma Beta')
    fig.colorbar(p1, ax=axs[1], label='Beta')
    plt.show()

    # plot the same subplots as above for density and temperatures
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    p0 = axs[0].pcolormesh(X, Y, rho_hist.T, norm=colors.LogNorm(vmin=1e-30,vmax=1e-28), cmap='viridis')
    axs[0].set_title('Density (g/cm**3)')
    fig.colorbar(p0, ax=axs[0], label='Density (g/cm**3)')
    p1 = axs[1].pcolormesh(X, Y, T_hist.T, norm=colors.LogNorm(vmin=1e5, vmax=1e7), cmap='inferno')
    axs[1].set_title('Temperature (K)')
    fig.colorbar(p1, ax=axs[1], label='Temperature (K)')
    plt.show()

    # plot the same subplots as above for gizmo density and number counts
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    p0 = axs[0].pcolormesh(X, Y, rho_hist_gizmo.T, norm=colors.LogNorm(vmin=1e-30,vmax=1e-28), cmap='viridis')
    axs[0].set_title('Gizmo Density (g/cm**3)')
    fig.colorbar(p0, ax=axs[0], label='Density (g/cm**3)')
    p1 = axs[1].pcolormesh(X, Y, num_counts.T, norm=colors.LogNorm(vmin=1e0, vmax=1e3), cmap='inferno')
    axs[1].set_title('Number Counts')
    fig.colorbar(p1, ax=axs[1], label='Number Counts')
    plt.show()


    stop()

def particle_plotter(args, i, plane='z', rotated=False):
    F = h5py.File(args.snappath+'snapshot_{}.hdf5'.format(str(i).zfill(3)), 'r')
    # stop()
    coord0 = F['PartType0']['Coordinates'][:]
    coord1 = F['PartType1']['Coordinates'][:]
    coord2 = F['PartType2']['Coordinates'][:]
    # correct the coordinates for the center
    coord0 = coord0 - args.center
    coord1 = coord1 - args.center
    coord2 = coord2 - args.center
    if plane=='x': ax1=1; ax2=2; axLOS=0
    if plane=='y': ax1=0; ax2=2; axLOS=1
    if plane=='z': ax1=0; ax2=1; axLOS=2
    if rotated:
        coord0 = np.dot(args.rotation_matrix, coord0.T).T
        coord1 = np.dot(args.rotation_matrix, coord1.T).T
        coord2 = np.dot(args.rotation_matrix, coord2.T).T
    ind = np.random.choice(len(coord0), 500000)
    # stop()
    plt.figure()
    plt.scatter(coord1[:,ax1][ind], coord1[:,ax2][ind], s=0.1, alpha=0.1, c='k', label='Dark Matter')
    plt.scatter(coord0[:,ax1][ind], coord0[:,ax2][ind], s=0.1, alpha=0.1, c='b', label='Gas')
    plt.scatter(coord2[:,ax1][ind], coord2[:,ax2][ind], s=0.1, alpha = 0.1, c='r', label='Old Stars')

    plt.xlabel('X (kpc)'); plt.ylabel('Y (kpc)')
    plt.xlim([-200, 200]); plt.ylim([-200, 200])
    # make the aspect ratio equal
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.savefig(args.plotpath+'buildgalaxy_plane{}_c{}.png'.format(i, plane, args.center_method), dpi=300)
    # plt.show()
    stop()

def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return math.sqrt(variance)

def plot_stars(args, plane='x'):
    if plane=='x': ax1=1; ax2=2; axLOS=0
    if plane=='y': ax1=0; ax2=2; axLOS=1
    if plane=='z': ax1=0; ax2=1; axLOS=2
    F = h5py.File(args.snappath+'snapshot_{}.hdf5'.format(str(args.snapnum).zfill(3)), 'r')
    coord_gas = F['PartType0']['Coordinates'][:]
    coord = F['PartType4']['Coordinates'][:]
    coord_old = F['PartType2']['Coordinates'][:]
    mass = F['PartType4']['Masses'][:]
    plt.figure(figsize=(10, 10))
    plt.scatter(coord_gas[:,ax1], coord_gas[:,ax2], s=0.02, alpha=0.01, c='b', label='Gas')
    plt.scatter(coord[:,ax1], coord[:,ax2], s=0.1, alpha=1, c='r', label='Young Stars')
    plt.scatter(coord_old[:,ax1], coord_old[:,ax2], s=0.1, alpha=1, c='k', label='Old Stars')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X (kpc)'); plt.ylabel('Y (kpc)')
    plt.xlim([0, 200]); plt.ylim([0, 400])
    plt.legend()
    plt.show()
    F.close()
    stop()

def make_RM_arrow_movie(args, indz, indRM):
    # Create figure and axis
    fig, ax = plt.subplots()

    # Set equal aspect ratio and limits
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

    # Add grid
    ax.grid(True)

    # Create initial arrow
    arrow = ax.quiver(0, 0, 0, 1, angles='xy', scale=1, scale_units='xy')

    def update(theta):
        
        # Calculate x and y components
        x = np.sin(theta)  # x component
        y = np.cos(theta)  # y component
        
        # Update arrow
        arrow.set_UVC(x, y)
        return arrow,

    # Create animation
    # Replace angles_list with your list of angles
    args.Tmin = 1e1; args.Tmax = 1e7
    dataz = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}z.npz".format('RM', 200, args.snapnum, args.Tmin, args.Tmax, args.TRe_key))
    angles_list = np.cumsum(dataz['data'][:, indRM][indz])*5 # list of angles in radians
    anim = FuncAnimation(fig, update, frames=angles_list, 
                        interval=50, blit=True)

    # Save animation
    anim.save(args.plotpath+'RM_movie/arrow_rotation.mp4', fps=10, writer='ffmpeg')

def make_RM_movie(args, rotation_matrix):
    args.Tmin = 1e1; args.Tmax = 1e7
    dataz = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}z.npz".format('RM', 200, args.snapnum, args.Tmin, args.Tmax, args.TRe_key))
    z = dataz['data'][:,-1]
    data = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}.npz".format('RM', 200, args.snapnum, args.Tmin, args.Tmax, args.TRe_key))
    points = data['data'][:, 0:2]; RM = data['data'][:, 2]
    indz = np.where((z>=-2500) & (z<=2500))[0]
    indz_arg = []

    # make a 10 frames per second movie out of the images
    def natural_sort_key(s):
        # Extract the number from the filename
        number = re.findall(r'movie_plot_(\d+)\.png', s)[0]
        return int(number)

    # Get list of PNG images and sort them naturally
    image_files = sorted(glob.glob(args.plotpath+'RM_movie/movie_plot_*.png'), key=natural_sort_key)
    clip = ImageSequenceClip(image_files, fps=10)
    clip.write_videofile("final_movie.mp4", audio=True)
    stop()
    # find the index closest to RM of 100 in RM
    indRM = np.argmin(np.abs(RM-100))
    for i in range(int(len(indz)/50)):
        indz_arg.append(indz[i*50])
    make_RM_arrow_movie(args, np.array(indz_arg), indRM)

    if os.path.exists(args.snappath+"snapshot_{}_processed_quantities_T{}.npy".format(str(args.snapnum).zfill(3), args.TRe_key)):
        print("Processed quantities already exist for this snapshot.")
        arr = np.load(args.snappath+"snapshot_{}_processed_quantities_T{}.npy".format(str(args.snapnum).zfill(3), args.TRe_key), allow_pickle=True)
        arrdata = arr.item()
        # stop()
        ne_corrected_photoionized = arrdata['ne_corrected_photoionized']
        # ne = arrdata['ne']
        T_photo = arrdata['grackle_temperature_photo']
    ds = yt.load(args.snappath+args.filename)
    ad = ds.all_data()
    ad_rot = rm.RotateGalaxy(args, ad, rotation_matrix, center=args.center)
    BLOS = ad_rot['PartType0', 'MagneticField'][:, 2]*1e6
    vol = ad_rot['PartType0', 'SmoothingLength']**3
    coord = ad_rot['PartType0', 'Coordinates']
    coord = np.array(coord)-args.center
    # create a projection of the ne_corrected_photoionized and BLOS
    LOS = 1
    Msize = 10
    res = 400
    # get indices of z values between -5000 and 5000
    # stop()


    # for i in range(int(len(indz)/50)):
    #     print(i)
    #     depth = z[indz[i*50]]/1000
    #     X = np.linspace(0-Msize/2, 0+Msize/2, res); Y = np.linspace(0-Msize/2, 0+Msize/2, res)
    #     X, Y = np.meshgrid(X, Y)
    #     ind_proj = np.where((np.abs(coord[:, 0]) <= X.max()) & (np.abs(coord[:, 1]) <= Y.max()) & (np.abs(coord[:, 2]-depth)<=LOS/2))[0]
    #     vol_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,0], coord[ind_proj][:,1], bins=res, weights=np.array(vol[ind_proj]), range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    #     num_counts, xedges, yedges = np.histogram2d(coord[ind_proj][:,0], coord[ind_proj][:,1], bins=res, range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    #     num_counts[num_counts==0] = 1
        
    #     BLOSvol_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,0], coord[ind_proj][:,1], bins=res, weights=np.array(vol[ind_proj])*np.array(BLOS[ind_proj]), range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    #     BLOS_proj = BLOSvol_proj/num_counts/vol_proj
    #     nevol_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,0], coord[ind_proj][:,1], bins=res, weights=np.array(vol[ind_proj])*np.array(ne_corrected_photoionized[ind_proj]), range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    #     ne_proj = nevol_proj/num_counts/vol_proj
    #     # stop()
    #     fig = plt.figure(figsize=(10, 5), constrained_layout=False)

    #     # First subplot
    #     ax1 = fig.add_subplot(1, 2, 1)
    #     im1 = ax1.imshow(BLOS_proj.T, origin='lower',
    #                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    #                     interpolation='nearest', cmap='PuOr', 
    #                     norm=colors.SymLogNorm(linthresh=1e-3, linscale=1.0,
    #                                         vmin=-0.2, vmax=0.2))
    #     axins = inset_axes(ax1, width="90%", height="5%", loc='lower center', bbox_to_anchor=(0, -0.2, 1, 1), bbox_transform=ax1.transAxes, borderpad=0,)
    #     cbar1 = fig.colorbar(im1, cax=axins, orientation='horizontal')
    #     ax1.set_ylabel('Y (kpc)')
    #     ax1.set_xlabel('X (kpc)')
    #     # plot a cross at points[indRM]
    #     ax1.scatter(points[indRM, 0]/1e3, points[indRM,1]/1e3, c='r', s=10, marker='x')

    #     # Second subplot
    #     ax2 = fig.add_subplot(1, 2, 2)
    #     im2 = ax2.imshow(ne_proj.T, origin='lower',
    #                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    #                     interpolation='nearest', cmap='viridis', 
    #                     norm=colors.LogNorm(vmin=1e-4,vmax=1e-1))
    #     ax2.set_xlabel('X (kpc)')
    #     ax2.yaxis.set_ticklabels([])
    #     ax2.scatter(points[indRM, 0]/1e3, points[indRM,1]/1e3, c='r', s=10, marker='x')

    #     # Adjust subplot spacing
    #     plt.subplots_adjust(wspace=0, hspace=0)

    #     # Add colorbars at the bottom
    #     cbar1.set_label(r'$B_{\rm LOS}~{\rm (\mu G)}$')

    #     axins = inset_axes(ax2, width="90%", height="5%", loc='lower center', bbox_to_anchor=(0, -0.2, 1, 1), bbox_transform=ax2.transAxes, borderpad=0,)
    #     cbar2 = fig.colorbar(im2, cax=axins, orientation='horizontal')
    #     cbar2.set_label(r'$n_{\rm e}~{\rm (cm^{-3})}$')
    #     plt.suptitle("Depth = {} kpc".format(depth))
    #     # stop()
    #     plt.savefig(args.plotpath+'RM_movie/movie_plot_{}.png'.format(i), dpi=300, bbox_inches='tight')
    


    stop()

def find_center(args, i, scenario, resolution, key, center_method='com'):
    snappath = '/scratch/jh2/hs9158/results/LMC_run_{}_wind_scenario_{}{}/'.format(resolution, scenario, key)
    filename = 'snapshot_{}.hdf5'.format(str(i).zfill(3))
    if resolution == 'medres': ngasdisk = int(22e5)
    if resolution == 'lowres': ngasdisk = int(1e5) # number of gas particles initialized in the disk
    if resolution == 'medhighres': ngasdisk = int(88e5)
    if resolution == 'highres': ngasdisk = int(22e6)
    V_LMC = np.array([-314.8, 29.03, -53.951])  # in km/s
    tstep = 5e6 # in yr
    if scenario<2: center = np.array([100, 100, 100])
    else:
        if center_method == 'density':
            with h5py.File(snappath+filename, 'r') as F:
                coords = F['PartType0']['Coordinates'][:]
                rho = F['PartType0']['Density'][:]
            center = coords[np.argmax(rho)]
        if args.center_method == 'com':
            with h5py.File(snappath+filename, 'r') as F:
                coords = F['PartType2']['Coordinates'][0:ngasdisk][:]
                fake_center = np.array([100, 100, 100+np.linalg.norm(V_LMC)*1e5*constants.year/constants.pc/1e3*i*tstep])
                # find indices within 20 kpc of the fake center
                ind = np.where(np.linalg.norm(coords-fake_center, axis=1)<5)[0]
                # find indices common between ind and np.arange(args.ngasdisk)
                ind = np.intersect1d(ind, np.arange(ngasdisk))
                # args.center = np.average(coords[ind], axis=0)
                center = np.average(coords, axis=0)
    args.center = center
    return center

def plot_rcyl_profile(args, data, zscale = 10, rscale=10, weight='Density', property='MagneticField', component='azimuthal', rotation_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), plot=False): # weight could be any property of the particles or 'Volume'
    # This function plots the radial profile of the cylindrical radius of any gas property the particles
    # in the simulation
    ad = data.all_data()
    # read the coordinates of the particles
    coord = ad['PartType0', 'Coordinates'].in_units('kpc')
    print("Center is ", args.center)
    coord = coord-(args.center)*unyt.kpc
    # rotate the coordinates by the rotation matrix
    coord = np.dot(rotation_matrix, coord.T).T
    # read the property of the particles
    prop = ad['PartType0', property]
    # if prop is 3D, rotate it by the rotation matrix
    if len(prop.shape)==2:
        prop = np.dot(rotation_matrix, prop.T).T
        print("rotated the {} by the rotation matrix".format(property))

    # read the weights for the property
    if weight=='Volume': weight = 4/3*np.pi*(ad['PartType0', 'SmoothingLength'].in_units('cm'))**3
    else: weight = ad['PartType0', weight]
    if property=='MagneticField':
        if component=='azimuthal':
            B_pol = la.convert_to_cylindrical(coord, prop)
            prop = B_pol[:, 1]*1e6
        elif component=='radial':
            B_pol = la.convert_to_cylindrical(coord, prop)
            prop = B_pol[:, 0]*1e6
        elif component=='vertical':
            B_pol = la.convert_to_cylindrical(coord, prop)
            prop = B_pol[:, 2]*1e6
        elif component=='turb':
            rcyl = np.linalg.norm(coord, axis=1)
            z = coord[:, 2]
            ind_cyl = np.where((np.abs(z)<zscale) & (rcyl<rscale))[0]
            B_pol = la.convert_to_cylindrical(coord, prop)*1e6
            rcyl_cyl = rcyl[ind_cyl]; B_pol_cyl = B_pol[ind_cyl]; weight_cyl = weight[ind_cyl]
            B_az_cyl = B_pol_cyl[:, 1]
            bins = int(rscale)*3; r_bins = np.linspace(0, rscale, bins)
            r_bins = (r_bins[1:]+r_bins[:-1])/2
            B_turb = np.zeros(len(r_bins))
            for i in range(len(r_bins)-1):
                ind = np.where((rcyl_cyl>=r_bins[i]-rscale/bins/2) & (rcyl_cyl<r_bins[i]+rscale/bins/2))[0]
                # subtract the average azimuthal field from the second column of B_pol_cyl[ind]
                B_pol_cyl[ind, 1] -= np.average(B_az_cyl[ind], weights=weight_cyl[ind], axis=0)
                # compute standard deviation of the B_pol_cyl in rach column
                # take weight weighted standard deviation of B_turb
                # B_turb[i] = np.linalg.norm([weighted_std(B_pol_cyl[ind,0], weight_cyl[ind]),
                #                             weighted_std(B_pol_cyl[ind,1], weight_cyl[ind]),
                #                             weighted_std(B_pol_cyl[ind,2], weight_cyl[ind])])
                B_turb[i] = np.average(np.linalg.norm(B_pol_cyl[ind], axis=1), weights=weight_cyl[ind], axis=0)

            return r_bins, B_turb

        else: prop = np.linalg.norm(prop, axis=1)*1e6
    if property=='Density': prop = prop.in_units('g/cm**3')

    # convert the coordinates to cylindrical coordinates
    rcyl = np.linalg.norm(coord, axis=1)
    z = coord[:, 2]
    ind_cyl = np.where((np.abs(z)<zscale) & (rcyl<rscale))[0]
    rcyl_cyl = rcyl[ind_cyl]; prop_cyl = prop[ind_cyl]; weight_cyl = weight[ind_cyl]
    # create a radial profile of the property
    print("plotting the magnitude of the property")
    # create bins for radius and calculate the weighted mean of the property in each bin
    bins = int(rscale)*4; r_bins = np.linspace(0, rscale, bins)
    r_bins = (r_bins[1:]+r_bins[:-1])/2
    prop_profile = np.zeros(len(r_bins))
    for i in range(len(r_bins)-1):
        ind = np.where((rcyl_cyl>=r_bins[i]-rscale/bins/2) & (rcyl_cyl<r_bins[i]+rscale/bins/2))[0]
        prop_profile[i] = np.average(prop_cyl[ind], weights=weight_cyl[ind], axis=0)
    if plot:
        plt.figure()
        plt.plot(r_bins, prop_profile)
        plt.xlabel('r (kpc)'); plt.ylabel('B (uG)')
        plt.yscale('log')
        plt.title("Time = {} Myr".format(args.snapnum*args.tstep/1e6))
        plt.ylim([1e-2, 1e1])
        plt.show()
        plt.close()
        return r_bins, prop_profile
    else:
        return r_bins, prop_profile
        
def plot_time_rcyl_profile(args, snaprange, timesteps = 10, zscale = 10, rscale=10, weight='Density', property='MagneticField', component = 'azimuthal', rotation_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), save=True):
    r_list = []; prop_list = []; time_list = []
    
    for t in range(timesteps+1):
        filename = 'snapshot_{}.hdf5'.format(str(int(snaprange/timesteps)*t).zfill(3))
        args.snapnum = int(snaprange/timesteps)*t
        time = int(snaprange/timesteps)*t*args.tstep/1e6; time_list.append(time)
        ds = yt.load(args.snappath+filename)
        # F = h5py.File(args.snappath+'snapdir_{}/snapshot_{}.0.hdf5'.format(str(int(snaprange/timesteps)*t).zfill(3), str(int(snaprange/timesteps)*t).zfill(3)), 'r')
        # ds = yt.load(args.snappath+'snapdir_{}/snapshot_{}.0.hdf5'.format(str(int(snaprange/timesteps)*t).zfill(3), str(int(snaprange/timesteps)*t).zfill(3)))
        field_list = []
        for field in ds.field_list:
            field_list.append(field[0])

        # UNCOMMENT OLY WHEN MAKING RADIAL PROFILES OF IONIZATION RELATED QUANTITIES
        # if "PartType4" in ds.field_list:
        #     ds = rm.CorrectneCloudySlug(args, ds, Tmin=1e1, Tmax=10**(4.3), cond='T', col=1e19, rho=1e-1, ne_corrected=True)  # here Tmin and Tmax are ranges to correct the electron number density
        # else: ds = rm.CorrectIons(ds)
        
        r, prop = plot_rcyl_profile(args, ds, zscale=zscale, rscale=rscale, weight=weight, property=property, component=component, rotation_matrix=rotation_matrix, plot=False)
        r_list.append(r); prop_list.append(prop)

    # save the r_list, prop_list and time_list in a npz file
    if save: np.savez(args.plotpath+'time_rcyl_profile_{}_{}.npz'.format(property, component), r_list, prop_list, time_list)
    
    # plot all arrays of r_list and prop_list with different colors, from light grey to black with increasing gradient
    # also plot the time in the legend
    fig, ax = plt.subplots()
    for i in range(timesteps+1):
        ax.plot(r_list[i], prop_list[i], label='{} Myr'.format(time_list[i]), color='k', alpha=(i+1)/(timesteps+1))
    ax.set_xlabel('r (kpc)'); ax.set_ylabel(r'${\rm B}~(\mu{\rm G})$')
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.set_title("Component = {}".format(component))
    plt.ylim([1e-1, 1e2])
    plt.show()
    stop()

def plot_last_rcyl_profile(args, lastsnap, zscale = 10, rscale=10, weight='Density', property='MagneticField', component = 'azimuthal', rotation_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
    r_list = []; prop_list = []
    scenarios = [0, 0, 3]
    keys = ['_turb12', '_turb6_order2', '_turb6_order2']
    legend = ['turb12_s0', 'turb6_order2_s0', 'turb6_order2_s3']
    for i in range(len(keys)):
        args.snappath = '/scratch/jh2/hs9158/results/LMC_run_{}_wind_scenario_{}{}/'.format(args.resolution, scenarios[i], keys[i])
        args.plotpath = "/scratch/jh2/hs9158/results/plots/LMC_run_{}_wind_scenario_{}{}/".format(args.resolution, scenarios[i], keys[i])
        filename = 'snapshot_{}.hdf5'.format(str(lastsnap).zfill(3))
        ds = yt.load(args.snappath+filename)
        r, prop = plot_rcyl_profile(args, ds, zscale=zscale, rscale=rscale, weight=weight, property=property, component = component, rotation_matrix=rotation_matrix, plot=False)
        r_list.append(r); prop_list.append(prop)
    # save the r_list, prop_list and time_list in a npz file
    np.savez(args.plotpath+'last_rcyl_profile_{}.npz'.format(property), r_list, prop_list)
    # plot all arrays of r_list and prop_list with different colors
    fig, ax = plt.subplots()
    for i in range(3):
        ax.plot(r_list[i], prop_list[i], label='{}'.format(legend[i]))
    ax.set_xlabel('r (kpc)'); ax.set_ylabel(r'${\rm B}~(\mu{\rm G})$')
    ax.set_yscale('log')
    plt.title('Time = {} Myr'.format(lastsnap*args.tstep/1e6))
    ax.legend(loc='best')
    plt.ylim([1e-1, 1e2])
    plt.show()
    stop()
    
def my_axis_field0(field, data):
    return data["PartType0", "Coordinates"][:, 0]

def my_axis_field1(field, data):
    return data["PartType0", "Coordinates"][:, 1]

def my_axis_field2(field, data):
    return data["PartType0", "Coordinates"][:, 2]

def make_particle_plot(args, i, rotation_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
    data = yt.load(args.snappath+'snapshot_{}.hdf5'.format(str(i).zfill(3)))
    data.add_field(("PartType0", "coord_x"), function=my_axis_field0, units='kpc', sampling_type='local')
    data.add_field(("PartType0", "coord_y"), function=my_axis_field1, units='kpc', sampling_type='local')
    data.add_field(("PartType0", "coord_z"), function=my_axis_field2, units='kpc', sampling_type='local')
    plot = yt.ParticlePlot(data, ("PartType0", "coord_x"),("PartType0", "coord_z"), ("PartType0", "Masses"), figure_size=5)
    plot.set_log(("PartType0", "Masses"), True)
    plot.set_log("coord_x", False); plot.set_log("coord_z", False)
    plot.set_xlim(0, 200); plot.set_ylim(0, 400)
    plot.save(args.plotpath+'particle_fullbox_{}_part0.png'.format(i))

def plot_mass_histogram(x, y, x_label='x', y_label='y', bins=100, log=True):
    if log==True:
        ind_n0 = np.where(y!=0)[0]
        x = x[ind_n0]; x = np.log10(x)
        y = y[ind_n0]; y = np.log10(y)
    plt.figure(figsize=(5,5))
    hist, xedges, yedges = np.histogram2d(x, y, bins=[bins, bins])
    # take log of all hist values that are non-zero
    hist[hist!=0] = np.log10(hist[hist!=0])
    im = plt.imshow(hist.T, origin='lower', aspect='auto',
            extent=[min(x), max(x), min(y), max(y)],
            cmap='viridis',  vmin=2, vmax=6); cbar = plt.colorbar(label=r'$\rm Log_{10}$'+' Counts');plt.xlabel(r'$\rm Log_{10}$'+x_label);plt.ylabel(r'$\rm Log_{10}$'+y_label); plt.xlim([1.05, 6]); plt.ylim([-3, 0.06])
    cbar.solids.set_edgecolor("face")
    plt.savefig("/scratch/jh2/hs9158/results/paper_plots/phase_plot.pdf", dpi=100)
    stop()

def make_paper_plot_rho(args, i, rotated=True, slice=False):
    def return_quantities(plotpath, plane):
        if slice:
            if rotated: quantities = np.load(plotpath+'quantities_plane{}_{}_rotated_sliced_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
            else: quantities = np.load(plotpath+'quantities_plane{}_{}_sliced_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
        else:
            if rotated: quantities = np.load(plotpath+'quantities_plane{}_{}_rotated_projected_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
            else: quantities = np.load(plotpath+'quantities_plane{}_{}_projected_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
        if quantities['arr_0'].shape[0]==12:
            X, Y, dens_slicez, BX_slicez, BY_slicez, B_slicez, V_slice, Z_slice, T_slice, VX_slice, VY_slice, Mcenter = quantities['arr_0']
            SFR_proj = np.zeros_like(dens_slicez)
        else:
            X, Y, SFR_proj, dens_slicez, BX_slicez, BY_slicez, B_slicez, V_slice, Z_slice, T_slice, VX_slice, VY_slice, Mcenter = quantities['arr_0']
        return X, Y, dens_slicez


    print(i)
    plotpath = "/scratch/jh2/hs9158/results/plots/LMC_run_medhighres_wind_scenario_0_turb6_order2/"
    X, Y, dens_slice_Iz = return_quantities(plotpath, 'z')
    X, Y, dens_slice_Iy = return_quantities(plotpath, 'y')
    plotpath = "/scratch/jh2/hs9158/results/plots/LMC_run_medhighres_wind_scenario_3_turb6_order2/"
    X, Y, dens_slice_Wz = return_quantities(plotpath, 'z')
    X, Y, dens_slice_Wy = return_quantities(plotpath, 'y')

    # transpose all the arrays
    # dens_slice_Iz = dens_slice_Iz.T; dens_slice_Iy = dens_slice_Iy.T; dens_slice_Wz = dens_slice_Wz.T; dens_slice_Wy = dens_slice_Wy.T
    # plt.switch_backend('agg')

    # make 4 subplots in grid of 3 by 2
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 2)
    gs.update(wspace=0, hspace=0)

    # Create subplots using the gridspec
    axs = []
    for i in range(2):
        for j in range(2):
            axs.append(fig.add_subplot(gs[i, j]))

    observer_vector = np.array([0.4, 0.4, -0.82])
    wind_vector = np.array([-0.8, 0, -0.6])
    len_scale = 5
    cmapdens = cmr.rainforest_r
    cmapB = cmr.cosmic
    # dens_slice[dens_slice==0] = 1e-10
    p0 = axs[0].imshow(dens_slice_Iz, 
                    norm=colors.LogNorm(vmin=1e-4, vmax=1e1),
                    cmap=cmapdens,
                    extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower')
    axs[0].set_aspect('equal')

    # axs[0, 0].set_title(r'$\rm Density~(m_p~cm^{-3})$')

    p1 = axs[1].imshow(dens_slice_Iy, 
                    norm=colors.LogNorm(vmin=1e-4, vmax=1e1),
                    cmap=cmapdens,
                    extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower')
    axs[1].set_aspect('equal')
    # axs[0, 1].set_title(r'$\rm Density~(m_p~cm^{-3})$')
    # fig.colorbar(p0,label="Density"); fig.colorbar(p1,label="B (uG)")
    p2 = axs[2].imshow(dens_slice_Wz, 
                    norm=colors.LogNorm(vmin=1e-4, vmax=1e1),
                    cmap=cmapdens,
                    extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower')
    axs[2].set_aspect('equal')

    p3 = axs[3].imshow(dens_slice_Wy, 
                    norm=colors.LogNorm(vmin=1e-4, vmax=1e1),
                    cmap=cmapdens,
                    extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower')
    axs[3].set_aspect('equal')

    # make the observer and the wind arrows
    axs[0].arrow(0, 0, observer_vector[0]*len_scale, observer_vector[1]*len_scale, head_width=0.3, head_length=0.3, fc='lavenderblush', ec='lavenderblush')
    axs[1].arrow(0, 0, observer_vector[0]*len_scale, observer_vector[2]*len_scale, head_width=0.3, head_length=0.3, fc='lavenderblush', ec='lavenderblush')
    axs[2].arrow(0, 0, observer_vector[0]*len_scale, observer_vector[1]*len_scale, head_width=0.3, head_length=0.3, fc='lavenderblush', ec='lavenderblush')
    axs[3].arrow(0, 0, observer_vector[0]*len_scale, observer_vector[2]*len_scale, head_width=0.3, head_length=0.3, fc='lavenderblush', ec='lavenderblush')

    axs[2].arrow(0, 0, wind_vector[0]*len_scale, wind_vector[1]*len_scale, head_width=0.3, head_length=0.3,fc='lavenderblush', ec='lavenderblush', linestyle='--')
    axs[3].arrow(0, 0, wind_vector[0]*len_scale, wind_vector[2]*len_scale, head_width=0.3, head_length=0.3,fc='lavenderblush', ec='lavenderblush', linestyle='--')

    # add a colorbar to the bottom of the plot with a controllable size
    # After creating your plots
    # Create a new axes for the colorbar
    # cax = fig.add_axes([0.16, 0.02, 0.7, 0.025])  # [left, bottom, width, height]

    # Create the colorbar
    cbar = fig.colorbar(p2, ax=[axs[2], axs[3]], orientation='horizontal', shrink=0.92)
    # cbar = fig.colorbar(p3, orientation='horizontal')
    cbar.set_label(r'$\rm Density~(cm^{-3})$')
    # for ax in axs.flat:
    #     ax.set(xlabel='X (kpc)', ylabel='Y (kpc)')
        # ax.label_outer()
    axs[0].set_ylabel('Y (kpc)'); axs[2].set_ylabel('Y (kpc)')
    axs[2].set_xlabel('X (kpc)'); axs[3].set_xlabel('X (kpc)')
    axs[0].set_xticklabels([])
    axs[1].set_xticklabels([])
    axs[1].set_yticklabels([])
    axs[3].set_yticklabels([])

    ax_right_top = axs[1].twinx()
    ax_right_bottom = axs[3].twinx()

    # # Remove ticks from twin axes
    # ax_right_top.set_yticks([])
    # ax_right_bottom.set_yticks([])

    # Add labels
    ax_right_top.set_ylabel('Z (kpc)', rotation=270, labelpad=15)
    ax_right_bottom.set_ylabel('Z (kpc)', rotation=270, labelpad=15)

    # Set aspect ratio using datalim
    # for ax in axs:
    #     ax.set_aspect('equal', adjustable='datalim')
    ax_right_top.set_aspect('equal', adjustable='datalim')
    ax_right_bottom.set_aspect('equal', adjustable='datalim')

    # Adjust label positions (modify these values as needed)
    ax_right_top.yaxis.set_label_coords(1.15, 0.5)
    ax_right_bottom.yaxis.set_label_coords(1.15, 0.5)

    # Adjust label positions if needed
    ax_right_top.yaxis.set_label_coords(1.15, 0.5)
    ax_right_bottom.yaxis.set_label_coords(1.15, 0.5)
    axs[0].set_xlim(-9.99, 9.99); axs[1].set_xlim(-9.99, 9.99); axs[2].set_xlim(-9.99, 9.99); axs[3].set_xlim(-9.99, 9.99)
    axs[0].set_ylim(-9.99, 9.99); axs[1].set_ylim(-9.99, 9.99); axs[2].set_ylim(-9.99, 9.99); axs[3].set_ylim(-9.99, 9.99)
    ax_right_top.set_ylim(-9.99, 9.99)
    ax_right_bottom.set_ylim(-9.99, 9.99)
    
    plt.savefig("/scratch/jh2/hs9158/results/paper_plots/density_plot.pdf", dpi=100)
    # plt.savefig("/scratch/jh2/hs9158/results/paper_plots/density_plot.png", dpi=50, bbox_inches='tight')
    plt.close()

def make_paper_plot_Brad(args, i):
    snappath1 = '/scratch/jh2/hs9158/results/LMC_run_medhighres_wind_scenario_0_turb6_order2/'
    snappath2 = '/scratch/jh2/hs9158/results/LMC_run_medhighres_wind_scenario_3_turb6_order2/'
    ds = yt.load(snappath1+'snapshot_{}.hdf5'.format(str(i).zfill(3)))
    alpha = np.radians(0); beta = args.theta_LMC; gamma = np.radians(225)
    a = np.cos(alpha); b = np.cos(beta); c = np.cos(gamma)
    d = np.sin(alpha); e = np.sin(beta); f = np.sin(gamma)    
    R_new = np.array([[b*c, d*e*c-a*f, a*e*c+d*f], [b*f, d*e*f+a*c, a*e*f-d*c], [-e, d*b, a*b]])
    center = find_center(args, 100, 0, 'medhighres', '_turb6_order2', center_method='com')
    r, brI = pl.plot_rcyl_profile(args, ds, zscale = 0.5, rscale=10, weight='Volume', property='MagneticField', component = 'radial', rotation_matrix=R_new, plot=False)
    r, bazI = pl.plot_rcyl_profile(args, ds, zscale = 0.5, rscale=10, weight='Volume', property='MagneticField', component = 'azimuthal', rotation_matrix=R_new, plot=False)
    r, BzI = pl.plot_rcyl_profile(args, ds, zscale = 0.5, rscale=10, weight='Volume', property='MagneticField', component = 'vertical', rotation_matrix=R_new, plot=False)
    ds = yt.load(snappath2+'snapshot_{}.hdf5'.format(str(i).zfill(3)))
    center = find_center(args, 100, 3, 'medhighres', '_turb6_order2', center_method='com')
    r, brW = pl.plot_rcyl_profile(args, ds, zscale = 0.5, rscale=10, weight='Volume', property='MagneticField', component = 'radial', rotation_matrix=R_new, plot=False)
    r, bazW = pl.plot_rcyl_profile(args, ds, zscale = 0.5, rscale=10, weight='Volume', property='MagneticField', component = 'azimuthal', rotation_matrix=R_new, plot=False)
    r, BzW = pl.plot_rcyl_profile(args, ds, zscale = 0.5, rscale=10, weight='Volume', property='MagneticField', component = 'vertical', rotation_matrix=R_new, plot=False)
    fig, ax = plt.subplots()
    ax.plot(r, brI, label=r'$B_{\rm r}~{\rm (I)}$', color='k', linewidth=3)
    ax.plot(r, bazI, label=r'$B_{\rm \phi}~{\rm (I)}$', color='crimson', linewidth=3)
    ax.plot(r, BzI, label=r'$B_{\rm z}~{\rm (I)}$', color='lightblue', linewidth=3)
    ax.plot(r, brW, label=r'$B_{\rm r}~{\rm (W)}$', color='k', linestyle='--', linewidth=3)
    ax.plot(r, bazW, label=r'$B_{\rm \phi}~{\rm (W)}$', color='crimson', linestyle='--', linewidth=3)
    ax.plot(r, BzW, label=r'$B_{\rm z}~{\rm (W)}$', color='lightblue', linestyle='--', linewidth=3)
    ax.set_xlabel('r (kpc)'); ax.set_ylabel(r'$B~(\mu{\rm G})$')
    ax.set_xlim([0, 9])
    ax.legend(loc='best')
    plt.savefig("/scratch/jh2/hs9158/results/paper_plots/Brad_profile.pdf", dpi=100)

def analyse_RM_stats(resolution, scenario, key):
    RMstats = np.load("/scratch/jh2/hs9158/results/data/s_RM_master_{}_{}_{}.npy".format(resolution, scenario, key))
    s_LOS_i = RMstats[0]
    s_plane_i = RMstats[-1]
    s_RM = np.sum(RMstats[1])/np.sum(RMstats[2])
    sigma_LOS = np.sqrt(np.sum((s_LOS_i-s_plane_i)**2)/len(s_LOS_i))
    return s_LOS_i, s_RM, sigma_LOS, s_plane_i

def make_paper_plot_Bedge(args, i, rotated=True, slice=False, key='magnitude'): # key could be 'magnitude' or 'vertical' or 'histogram'
    def return_quantities(plotpath, plane, i):
        if slice:
            if rotated: quantities = np.load(plotpath+'quantities_plane{}_{}_rotated_sliced_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
            else: quantities = np.load(plotpath+'quantities_plane{}_{}_sliced_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
        else:
            if rotated: quantities = np.load(plotpath+'quantities_plane{}_{}_rotated_projected_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
            else: quantities = np.load(plotpath+'quantities_plane{}_{}_projected_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
        if quantities['arr_0'].shape[0]==12:
            X, Y, dens_slicez, BX_slicez, BY_slicez, B_slicez, V_slice, Z_slice, T_slice, VX_slice, VY_slice, Mcenter = quantities['arr_0']
            SFR_proj = np.zeros_like(dens_slicez)
        else:
            X, Y, SFR_proj, dens_slicez, BX_slicez, BY_slicez, B_slicez, V_slice, Z_slice, T_slice, VX_slice, VY_slice, Mcenter = quantities['arr_0']
        return X, Y, BX_slicez, BY_slicez, B_slicez
    
    # load the RMstats data
    # RMstats_I = np.load("/scratch/jh2/hs9158/results/data/s_RM_master_{}_{}_{}.npy".format('medhighres', 0, '_turb6_order2'))
    s_LOS_i_I, s_RM_I, sigma_LOS_I, s_plane_i_I = analyse_RM_stats('medhighres', 0, '_turb6_order2')
    # RMstats_W = np.load("/scratch/jh2/hs9158/results/data/s_RM_master_{}_{}_{}.npy".format('medhighres', 3, '_turb6_order2'))
    s_LOS_i_W, s_RM_W, sigma_LOS_W, s_plane_i_W = analyse_RM_stats('medhighres', 3, '_turb6_order2')
    # load the RM and EM data
    args.Tmin = 1e1; args.Tmax = 1e7
    snappath = "/scratch/jh2/hs9158/results/LMC_run_medhighres_wind_scenario_0_turb6_order2/"
    dataz_I = np.load(snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}z.npz".format('RM', 200, i, args.Tmin, args.Tmax, args.TRe_key))
    data_I = np.load(snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}.npz".format('RM', 200, i, args.Tmin, args.Tmax, args.TRe_key))
    dataEM_I = np.load(snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}.npz".format('EM', 200, i, args.Tmin, args.Tmax, args.TRe_key))
    points_I = data_I['data'][:, 0:2]
    pointsz_I = dataz_I['data'][:, -1]
    RMdataz_I = dataz_I['data'][:, 0:-1]
    RM_I = data_I['data'][:, 2]
    EM_I = dataEM_I['data'][:, 2]
    points_I = points_I[np.where(EM_I<100)[0]]
    RM_I = RM_I[np.where(EM_I<100)[0]]
    # concatenate the points_I and s_LOS_i_I
    points_I = np.concatenate((points_I, s_LOS_i_I.reshape(-1, 1)), axis=1)
    snappath = "/scratch/jh2/hs9158/results/LMC_run_medhighres_wind_scenario_3_turb6_order2/"
    dataz_W = np.load(snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}z.npz".format('RM', 200, i, args.Tmin, args.Tmax, args.TRe_key))
    data_W = np.load(snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}.npz".format('RM', 200, i, args.Tmin, args.Tmax, args.TRe_key))
    dataEM_W = np.load(snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}.npz".format('EM', 200, i, args.Tmin, args.Tmax, args.TRe_key))
    points_W = data_W['data'][:, 0:2]
    pointsz_W = dataz_W['data'][:, -1]
    RMdataz_W = dataz_W['data'][:, 0:-1]
    RM_W = data_W['data'][:, 2]
    EM_W = dataEM_W['data'][:, 2]
    points_W = points_W[np.where(EM_W<100)[0]]
    RM_W = RM_W[np.where(EM_W<100)[0]]
    # concatenate the points_W and s_LOS_i_W
    points_W = np.concatenate((points_W, s_LOS_i_W.reshape(-1, 1)), axis=1)

    
    plotpath = "/scratch/jh2/hs9158/results/plots/LMC_run_medhighres_wind_scenario_0_turb6_order2/"
    X, Y, BX_slice_Iy, BY_slice_Iy, B_slice_Iy = return_quantities(plotpath, 'y', 100)
    plotpath = "/scratch/jh2/hs9158/results/plots/LMC_run_medhighres_wind_scenario_3_turb6_order2/"
    X, Y, BX_slice_Wy, BY_slice_Wy, B_slice_Wy = return_quantities(plotpath, 'y', 100)
    BX_slice_Iy = BX_slice_Iy*1e6; BY_slice_Iy = BY_slice_Iy*1e6; B_slice_Iy = B_slice_Iy*1e6
    BX_slice_Wy = BX_slice_Wy*1e6; BY_slice_Wy = BY_slice_Wy*1e6; B_slice_Wy = B_slice_Wy*1e6

    observer_vector = np.array([0, 0, 1])
    # rotate the points_I and points_W to the original simulation frame by taking the inverse of the rotation_matrix
    # take inverse of the rotation matrix
    points_I_og = np.dot(np.linalg.inv(args.rotation_matrix), points_I.T).T
    points_W_og = np.dot(np.linalg.inv(args.rotation_matrix), points_W.T).T
    observer_vector_og = np.dot(np.linalg.inv(args.rotation_matrix), observer_vector.T).T
    R = np.array([[np.cos(args.theta_LMC), 0, np.sin(args.theta_LMC)], [0, 1, 0], [-np.sin(args.theta_LMC), 0, np.cos(args.theta_LMC)]])  # negative of the making of LMC rotation_matrix
    # rotate points_I_og and points_W_og by R
    points_I_plot = np.dot(R, points_I_og.T).T
    points_W_plot = np.dot(R, points_W_og.T).T
    # rotate the observer vector by R
    observer_vector_plot = np.dot(R, observer_vector_og.T).T
    s_LOS_plane_I = s_LOS_i_I-s_plane_i_I; s_LOS_plane_W = s_LOS_i_W-s_plane_i_W
    # stop()
    # transpose all the arrays
    # dens_slice_Iz = dens_slice_Iz.T; dens_slice_Iy = dens_slice_Iy.T; dens_slice_Wz = dens_slice_Wz.T; dens_slice_Wy = dens_slice_Wy.T
    # plt.switch_backend('agg')
    # stop()
    if key=='histogram':
        # create two subplots where the first subplot is number of positive and negative RM values as a function of
        # s_LOS_plane_I and s_LOS_plane_W and the second subplot is the average positive and negative RM values
        #  as a function of s_LOS_plane_I and s_LOS_plane_W
        fig = plt.figure(figsize=(6, 6), constrained_layout=False)
        gs = fig.add_gridspec(1, 1)

        # Create subplots using the gridspec
        axs = []
        for i in range(1):
            for j in range(1):
                axs.append(fig.add_subplot(gs[i, j]))

        # create a histogram of the number of positive and negative RM values as a function of s_LOS_plane_I and s_LOS_plane_W
        # create the average positive and negative RM values as a function of s_LOS_plane_I and s_LOS_plane_W
        # first subplot
        bins_I = 20
        bins_W = 20
        # bins_I_pos = int((np.max(s_LOS_plane_I[np.where(RM_I>0)[0]])-np.min(s_LOS_plane_I[np.where(RM_I>0)[0]]))/100)
        # bins_I_neg = int((np.max(s_LOS_plane_I[np.where(RM_I<0)[0]])-np.min(s_LOS_plane_I[np.where(RM_I<0)[0]]))/100)
        # bins_W_pos = int((np.max(s_LOS_plane_W[np.where(RM_W>0)[0]])-np.min(s_LOS_plane_W[np.where(RM_W>0)[0]]))/200)
        # bins_W_neg = int((np.max(s_LOS_plane_W[np.where(RM_W<0)[0]])-np.min(s_LOS_plane_W[np.where(RM_W<0)[0]]))/200)
        axs[0].hist(s_LOS_plane_I, bins=bins_I, color='rebeccapurple', alpha=0.8, linewidth = 3, label=r'\texttt{I-2-6-M}', histtype='step')
        # axs[0].hist(s_LOS_plane_I[np.where(RM_I<0)[0]], bins=bins_I_neg, color='r', alpha=0.9, linewidth = 3, label=r'$\rm RM \textless 0~(I)$', histtype='step')
        axs[0].hist(s_LOS_plane_W, bins=bins_W, color='rebeccapurple', alpha=0.8, linestyle='--', linewidth = 3, label=r'\texttt{W-2-6-M}', histtype='step')
        # axs[0].hist(s_LOS_plane_W[np.where(RM_W<0)[0]], bins=bins_W_neg, color='r', alpha=0.7, linestyle='--', linewidth = 3, label=r'$\rm RM \textless 0~(W)$', histtype='step')
        axs[0].set_xlabel(r'$\langle s \rangle_{\rm RM, i}~\mathrm{(pc)}$')
        axs[0].set_ylabel('Number of RMs')
        axs[0].legend(loc='best')
        axs[0].set_xlim([-2000, 4000])

        # z_bins_I = np.linspace(-1000, 1000, 21)
        # z_bins_W = np.linspace(-2000, 4500, 21)
        # RM_I_pos = []; RM_I_neg = []; RM_W_pos = []; RM_W_neg = []
        # for i in range(len(z_bins_I)-1):
        #     ind = np.where((s_LOS_plane_I>z_bins_I[i]) & (s_LOS_plane_I<z_bins_I[i+1]))[0]
        #     if len(RM_I[ind])>0: RM_I_pos.append(np.mean(RM_I[ind][RM_I[ind]>0])); RM_I_neg.append(np.abs(np.mean(RM_I[ind][RM_I[ind]<0])))
        #     else: RM_I_pos.append(0); RM_I_neg.append(0)
        #     ind = np.where((s_LOS_plane_W>z_bins_W[i]) & (s_LOS_plane_W<z_bins_W[i+1]))[0]
        #     if len(RM_W[ind])>0: RM_W_pos.append(np.mean(RM_W[ind][RM_W[ind]>0])); RM_W_neg.append(np.abs(np.mean(RM_W[ind][RM_W[ind]<0])))
        #     else: RM_W_pos.append(0); RM_W_neg.append(0)

        # # second subplot
        # axs[1].plot((z_bins_I[1:]+z_bins_I[:-1])/2, RM_I_pos, color='b', linewidth=3, label='I')
        # axs[1].plot((z_bins_I[1:]+z_bins_I[:-1])/2, RM_I_neg, color='r', linewidth=3, label='I')
        # axs[1].plot((z_bins_W[1:]+z_bins_W[:-1])/2, RM_W_pos, color='b', linestyle='--', linewidth=3, label='W')
        # axs[1].plot((z_bins_W[1:]+z_bins_W[:-1])/2, RM_W_neg, color='r', linestyle='--', linewidth=3, label='W')
        # axs[1].set_xlabel(r'$s_{\rm LOS}-s_{\rm plane}$')
        # axs[1].set_ylabel(r'$\langle |RM| \rangle$')
        # axs[1].legend(loc='best')

        # stop()
        plt.savefig("/scratch/jh2/hs9158/results/paper_plots/RM_histogram.pdf", dpi=100)
        stop()
        # second subplot
        



    if key=='magnitude' or key=='vertical':
    # make 4 subplots in grid of 3 by 2
        fig = plt.figure(figsize=(12, 9.7), constrained_layout=False)
        gs = fig.add_gridspec(2, 2)
        gs.update(wspace=0, hspace=0)

        # Create subplots using the gridspec
        axs = []
        for i in range(2):
            for j in range(2):
                axs.append(fig.add_subplot(gs[i, j]))

        cmapdens = cmr.rainforest_r
        cmapB = cmr.cosmic
        cmapBz = cmr.holly
        # dens_slice[dens_slice==0] = 1e-10
        p0 = axs[0].imshow(B_slice_Iy, 
                        norm=colors.LogNorm(vmin=1e-1, vmax=1e1),
                        cmap=cmapB,
                        extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower')
        p2 = axs[2].imshow(BY_slice_Iy/B_slice_Iy, 
                        norm=colors.SymLogNorm(linthresh=0.1, linscale=1.0,
                                        vmin=-1, vmax=1), cmap=cmapBz,
                        extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower')
        axs[0].set_aspect('equal'); axs[2].set_aspect('equal')
        den=0.7
        axs[0].streamplot(X, Y, BX_slice_Iy, BY_slice_Iy, color='y', linewidth=0.5, density=den, arrowsize=1.0, arrowstyle='->', broken_streamlines=False)
        axs[2].streamplot(X, Y, BX_slice_Iy, BY_slice_Iy, color='y', linewidth=0.5, density=den, arrowsize=1.0, arrowstyle='->', broken_streamlines=False)
        # draw a scatter plot of the points_I_plot and points_W_plot
        color = np.repeat('b', len(RM_I))
        color[RM_I<0] = 'r'
        axs[0].scatter(points_I_plot[:, 0]/1e3, points_I_plot[:, 2]/1e3, c=color, s=np.sqrt(np.abs(RM_I)*20), edgecolors='w', linewidths=0.5)
        axs[2].scatter(points_I_plot[:, 0]/1e3, points_I_plot[:, 2]/1e3, c=color, s=np.sqrt(np.abs(RM_I)*20), edgecolors='w', linewidths=0.5)
        # draw an arrow towards the observer_vector_plot
        len_scale = 5
        axs[0].arrow(0, 0, observer_vector_plot[0]*len_scale, observer_vector_plot[2]*len_scale, head_width=0.3, head_length=0.3, fc='lavenderblush', ec='lavenderblush')
        axs[2].arrow(0, 0, observer_vector_plot[0]*len_scale, observer_vector_plot[2]*len_scale, head_width=0.3, head_length=0.3, fc='lavenderblush', ec='lavenderblush')
        # axs[0].arrow(0, 0, -0.8*len_scale, -0.6*len_scale, head_width=0.3, head_length=0.3,fc='lavenderblush', ec='lavenderblush', linestyle='--')
        # axs[2].arrow(0, 0, -0.8*len_scale, -0.6*len_scale, head_width=0.3, head_length=0.3,fc='lavenderblush', ec='lavenderblush', linestyle='--')

        p1 = axs[1].imshow(B_slice_Wy, 
                        norm=colors.LogNorm(vmin=1e-1, vmax=1e1),
                        cmap=cmapB,
                        extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower')
        p3 = axs[3].imshow(BY_slice_Wy/B_slice_Wy, 
                        norm=colors.SymLogNorm(linthresh=0.1, linscale=1.0,
                                        vmin=-1, vmax=1), cmap=cmapBz,
                        extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower')
        axs[1].set_aspect('equal'); axs[3].set_aspect('equal')
        axs[1].streamplot(X, Y, BX_slice_Wy, BY_slice_Wy, color='y', linewidth=0.5, density=den, arrowsize=1.0, arrowstyle='->', broken_streamlines=False)
        axs[3].streamplot(X, Y, BX_slice_Wy, BY_slice_Wy, color='y', linewidth=0.5, density=den, arrowsize=1.0, arrowstyle='->', broken_streamlines=False)
        # draw a scatter plot of the points_I_plot and points_W_plot
        color = np.repeat('b', len(RM_W))
        color[RM_W<0] = 'r'
        axs[1].scatter(points_W_plot[:, 0]/1e3, points_W_plot[:, 2]/1e3, c=color, s=np.sqrt(np.abs(RM_W)*20), edgecolors='w', linewidths=0.5)
        axs[3].scatter(points_W_plot[:, 0]/1e3, points_W_plot[:, 2]/1e3, c=color, s=np.sqrt(np.abs(RM_W)*20), edgecolors='w', linewidths=0.5)
        # draw an arrow towards the observer_vector_plot
        axs[1].arrow(0, 0, observer_vector_plot[0]*len_scale, observer_vector_plot[2]*len_scale, head_width=0.3, head_length=0.3,fc='lavenderblush', ec='lavenderblush')
        axs[3].arrow(0, 0, observer_vector_plot[0]*len_scale, observer_vector_plot[2]*len_scale, head_width=0.3, head_length=0.3,fc='lavenderblush', ec='lavenderblush')
        axs[1].arrow(0, 0, -0.8*len_scale, -0.6*len_scale, head_width=0.3, head_length=0.3,fc='lavenderblush', ec='lavenderblush', linestyle='--')
        axs[3].arrow(0, 0, -0.8*len_scale, -0.6*len_scale, head_width=0.3, head_length=0.3,fc='lavenderblush', ec='lavenderblush', linestyle='--')

        # fig.suptitle("Time = 500 Myr")
        # make whitespace between subplots 0
        plt.subplots_adjust(wspace=0, hspace=0)
        # add a colorbar to the bottom of the plot with a controllable size
        # After creating your plots
        # Create a new axes for the colorbar
        # cax = fig.add_axes([0.16, 0.2, 0.7, 0.025])  # [left, bottom, width, height]
        # cax1 = fig.add_axes([0.16, 0.7, 0.7, 0.025])  # [left, bottom, width, height]

        # Create the colorbar
        # For the first row colorbar
        cbar = fig.colorbar(p0, ax=[axs[0], axs[1]], orientation='vertical', shrink=0.92)

        # For the second row colorbar
        cbar1 = fig.colorbar(p3, ax=[axs[2], axs[3]], orientation='vertical', shrink=0.92)
        axs[0].set_xlim([-9.99, 9.99]); axs[0].set_ylim([-9.99, 9.99])
        axs[1].set_xlim([-9.99, 9.99]); axs[1].set_ylim([-9.99, 9.99])
        axs[2].set_xlim([-9.99, 9.99]); axs[2].set_ylim([-9.99, 9.99])
        axs[3].set_xlim([-9.99, 9.99]); axs[3].set_ylim([-9.99, 9.99])

        axs[0].set_ylabel('Z (kpc)'); axs[3].set_xlabel('X (kpc)')
        axs[2].set_xlabel('X (kpc)'); axs[2].set_ylabel('Z (kpc)')
        axs[1].set_yticklabels([]); axs[3].set_yticklabels([]); axs[1].set_xticklabels([]); axs[1].set_yticklabels([])

        cbar.set_label(r'$B~{\rm (\mu G)}$')
        cbar1.set_label(r'$B_{\rm z}/B$')
        
        plt.savefig("/scratch/jh2/hs9158/results/paper_plots/B_edgeplot_{}.pdf".format(key), dpi=50, bbox_inches='tight', pad_inches=0.5)
        plt.savefig("/scratch/jh2/hs9158/results/paper_plots/B_edgeplot_{}.png".format(key), dpi=50, bbox_inches='tight')
        plt.close()

def make_paper_SFR_plot():
    SFR_I = np.loadtxt("/scratch/jh2/hs9158/results/LMC_run_medhighres_wind_scenario_0_turb6_order2/SFR.txt")
    SFR_W = np.loadtxt("/scratch/jh2/hs9158/results/LMC_run_medhighres_wind_scenario_3_turb6_order2/SFR.txt")
    time = np.arange(0, 505, 5)
    plt.figure(figsize=(6, 6))
    plt.plot(time, SFR_I, label=r'\texttt{I-2-6-M}', color='rebeccapurple', linewidth=3)
    plt.plot(time, SFR_W, label=r'\texttt{W-2-6-M}', color='rebeccapurple', linewidth=3, linestyle='--')
    plt.xlabel('Simulation time (Myr)'); plt.ylabel('SFR (M$_\odot$ yr$^{-1}$)')
    plt.xlim([0, 500]); plt.ylim([0, 0.25])
    plt.legend(loc='best')
    plt.savefig("/scratch/jh2/hs9158/results/paper_plots/SFR_plot.pdf", dpi=100)
    plt.close()

    
def make_paper_plot_B(args, rotated=True, slice=False):
    def return_quantities(plotpath, plane, i):
        if slice:
            if rotated: quantities = np.load(plotpath+'quantities_plane{}_{}_rotated_sliced_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
            else: quantities = np.load(plotpath+'quantities_plane{}_{}_sliced_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
        else:
            if rotated: quantities = np.load(plotpath+'quantities_plane{}_{}_rotated_projected_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
            else: quantities = np.load(plotpath+'quantities_plane{}_{}_projected_c{}.npz'.format(plane, i, args.center_method), allow_pickle=True)
        if quantities['arr_0'].shape[0]==12:
            X, Y, dens_slicez, BX_slicez, BY_slicez, B_slicez, V_slice, Z_slice, T_slice, VX_slice, VY_slice, Mcenter = quantities['arr_0']
            SFR_proj = np.zeros_like(dens_slicez)
        else:
            X, Y, SFR_proj, dens_slicez, BX_slicez, BY_slicez, B_slicez, V_slice, Z_slice, T_slice, VX_slice, VY_slice, Mcenter = quantities['arr_0']
        return X, Y, BX_slicez, BY_slicez, B_slicez
    
    plotpath = "/scratch/jh2/hs9158/results/plots/LMC_run_medhighres_wind_scenario_0_turb6_order2/"
    X, Y, BX_slice_Iz0, BY_slice_Iz0, B_slice_Iz0 = return_quantities(plotpath, 'z', 0)
    X, Y, BX_slice_Iz1, BY_slice_Iz1, B_slice_Iz1 = return_quantities(plotpath, 'z', 25)
    X, Y, BX_slice_Iz2, BY_slice_Iz2, B_slice_Iz2 = return_quantities(plotpath, 'z', 50)
    X, Y, BX_slice_Iz3, BY_slice_Iz3, B_slice_Iz3 = return_quantities(plotpath, 'z', 75)
    X, Y, BX_slice_Iz4, BY_slice_Iz4, B_slice_Iz4 = return_quantities(plotpath, 'z', 100)
    plotpath = "/scratch/jh2/hs9158/results/plots/LMC_run_medhighres_wind_scenario_3_turb6_order2/"
    X, Y, BX_slice_Wz0, BY_slice_Wz0, B_slice_Wz0 = return_quantities(plotpath, 'z', 0)
    X, Y, BX_slice_Wz1, BY_slice_Wz1, B_slice_Wz1 = return_quantities(plotpath, 'z', 25)
    X, Y, BX_slice_Wz2, BY_slice_Wz2, B_slice_Wz2 = return_quantities(plotpath, 'z', 50)
    X, Y, BX_slice_Wz3, BY_slice_Wz3, B_slice_Wz3 = return_quantities(plotpath, 'z', 75)
    X, Y, BX_slice_Wz4, BY_slice_Wz4, B_slice_Wz4 = return_quantities(plotpath, 'z', 100)

    # make 4 subplots in grid of 3 by 2
    fig = plt.figure(figsize=(15, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 5)
    gs.update(wspace=0.05, hspace=0.05)
    # Create subplots using the gridspec
    axs = []
    for i in range(2):
        for j in range(5):
            axs.append(fig.add_subplot(gs[i, j]))

    cmapB = cmr.cosmic
    den=0.7
    p0 = axs[0].imshow(B_slice_Iz0*1e6, norm=colors.LogNorm(vmin=1e-1,vmax=10), cmap=cmapB, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    axs[0].streamplot(X, Y, BX_slice_Iz0, BY_slice_Iz0, color='y', linewidth=0.5, density=den, arrowsize=1.0, arrowstyle='->', broken_streamlines=False)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].set_xlim([-9.99, 9.99]); axs[0].set_ylim([-9.99, 9.99])
    axs[0].set_title('0 Myr')

    p1 = axs[1].imshow(B_slice_Iz1*1e6, norm=colors.LogNorm(vmin=1e-1,vmax=10), cmap=cmapB, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    axs[1].streamplot(X, Y, BX_slice_Iz1, BY_slice_Iz1, color='y', linewidth=0.5, density=den, arrowsize=1.0, arrowstyle='->', broken_streamlines=False)
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_xlim([-9.99, 9.99]); axs[1].set_ylim([-9.99, 9.99])
    axs[1].set_title('125 Myr')

    p2 = axs[2].imshow(B_slice_Iz2*1e6, norm=colors.LogNorm(vmin=1e-1,vmax=10), cmap=cmapB, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    axs[2].streamplot(X, Y, BX_slice_Iz2, BY_slice_Iz2, color='y', linewidth=0.5, density=den, arrowsize=1.0, arrowstyle='->', broken_streamlines=False)
    axs[2].set_aspect('equal', adjustable='box')
    axs[2].set_xlim([-9.99, 9.99]); axs[2].set_ylim([-9.99, 9.99])
    axs[2].set_title('250 Myr')

    p3 = axs[3].imshow(B_slice_Iz3*1e6, norm=colors.LogNorm(vmin=1e-1,vmax=10), cmap=cmapB, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    axs[3].streamplot(X, Y, BX_slice_Iz3, BY_slice_Iz3, color='y', linewidth=0.5, density=den, arrowsize=1.0, arrowstyle='->', broken_streamlines=False)
    axs[3].set_aspect('equal', adjustable='box')
    axs[3].set_xlim([-9.99, 9.99]); axs[3].set_ylim([-9.99, 9.99])
    axs[3].set_title('375 Myr')

    p4 = axs[4].imshow(B_slice_Iz4*1e6, norm=colors.LogNorm(vmin=1e-1,vmax=10), cmap=cmapB, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    axs[4].streamplot(X, Y, BX_slice_Iz4, BY_slice_Iz4, color='y', linewidth=0.5, density=den, arrowsize=1.0, arrowstyle='->', broken_streamlines=False)
    axs[4].set_aspect('equal', adjustable='box')
    axs[4].set_xlim([-9.99, 9.99]); axs[4].set_ylim([-9.99, 9.99])
    axs[4].set_title('500 Myr')

    # now the wind cases
    p5 = axs[5].imshow(B_slice_Wz0*1e6, norm=colors.LogNorm(vmin=1e-1,vmax=10), cmap=cmapB, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    axs[5].streamplot(X, Y, BX_slice_Wz0, BY_slice_Wz0, color='y', linewidth=0.5, density=den, arrowsize=1.0, arrowstyle='->', broken_streamlines=False)
    axs[5].set_aspect('equal', adjustable='box')
    axs[5].set_xlim([-9.99, 9.99]); axs[5].set_ylim([-9.99, 9.99])

    p6 = axs[6].imshow(B_slice_Wz1*1e6, norm=colors.LogNorm(vmin=1e-1,vmax=10), cmap=cmapB, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    axs[6].streamplot(X, Y, BX_slice_Wz1, BY_slice_Wz1, color='y', linewidth=0.5, density=den, arrowsize=1.0, arrowstyle='->', broken_streamlines=False)
    axs[6].set_aspect('equal', adjustable='box')
    axs[6].set_xlim([-9.99, 9.99]); axs[6].set_ylim([-9.99, 9.99])

    p7 = axs[7].imshow(B_slice_Wz2*1e6, norm=colors.LogNorm(vmin=1e-1,vmax=10), cmap=cmapB, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    axs[7].streamplot(X, Y, BX_slice_Wz2, BY_slice_Wz2, color='y', linewidth=0.5, density=den, arrowsize=1.0, arrowstyle='->', broken_streamlines=False)
    axs[7].set_aspect('equal', adjustable='box')
    axs[7].set_xlim([-9.99, 9.99]); axs[7].set_ylim([-9.99, 9.99])

    p8 = axs[8].imshow(B_slice_Wz3*1e6, norm=colors.LogNorm(vmin=1e-1,vmax=10), cmap=cmapB, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    axs[8].streamplot(X, Y, BX_slice_Wz3, BY_slice_Wz3, color='y', linewidth=0.5, density=den, arrowsize=1.0, arrowstyle='->', broken_streamlines=False)
    axs[8].set_aspect('equal', adjustable='box')
    axs[8].set_xlim([-9.99, 9.99]); axs[8].set_ylim([-9.99, 9.99])

    p9 = axs[9].imshow(B_slice_Wz4*1e6, norm=colors.LogNorm(vmin=1e-1,vmax=10), cmap=cmapB, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    axs[9].streamplot(X, Y, BX_slice_Wz4, BY_slice_Wz4, color='y', linewidth=0.5, density=den, arrowsize=1.0, arrowstyle='->', broken_streamlines=False)
    axs[9].set_aspect('equal', adjustable='box')
    axs[9].set_xlim([-9.99, 9.99]); axs[9].set_ylim([-9.99, 9.99])

    axs[0].set_ylabel('Y (kpc)'); 
    axs[5].set_xlabel('X (kpc)'); axs[5].set_ylabel('Y (kpc)')
    axs[6].set_xlabel('X (kpc)')
    axs[7].set_xlabel('X (kpc)')
    axs[8].set_xlabel('X (kpc)')
    axs[9].set_xlabel('X (kpc)')
    axs[0].set_xticklabels([])
    axs[1].set_xticklabels([]); axs[1].set_yticklabels([])
    axs[2].set_xticklabels([]); axs[2].set_yticklabels([])
    axs[3].set_xticklabels([]); axs[3].set_yticklabels([])
    axs[4].set_xticklabels([]); axs[4].set_yticklabels([])
    axs[6].set_yticklabels([])
    axs[7].set_yticklabels([])
    axs[8].set_yticklabels([])
    axs[9].set_yticklabels([])

    cbar = fig.colorbar(p0, ax=axs, orientation='vertical', pad=0.03)
    cbar.set_label(r'$\rm B~(\mathit{\mu}G)$')
    plt.savefig("/scratch/jh2/hs9158/results/paper_plots/B_plot.pdf", dpi=100, pad_inches=0.4, 
                metadata={'Creator': None, 'Producer': None})
    plt.savefig("/scratch/jh2/hs9158/results/paper_plots/B_plot.png", dpi=100, bbox_inches='tight')

def make_paper_RMstats(args, i):
    args.Tmin = 1e1; args.Tmax = 1e7
    dataz = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}z.npz".format('RM', 200, i, args.Tmin, args.Tmax, args.TRe_key))
    data = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}.npz".format('RM', 200, i, args.Tmin, args.Tmax, args.TRe_key))
    dataEM = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}.npz".format('EM', 200, i, args.Tmin, args.Tmax, args.TRe_key))
    points = data['data'][:, 0:2]
    pointsz = dataz['data'][:, -1]
    RMdataz = dataz['data'][:, 0:-1]
    RM = data['data'][:, 2]
    EM = dataEM['data'][:, 2]
    rm.get_plane_rotation(args, args.rotation_matrix)
    plane_rot = args.plane_rot
    s_LOS_i = []; s_RM_num = []; s_RM_den = []; s_plane_i = []

    for i in range(len(RM)):
        if EM[i]<100:
            s_LOS = np.sum(pointsz*np.abs(RMdataz[:,i]))/np.sum(np.abs(RMdataz[:,i]))
            s_LOS_i.append(s_LOS)
            s_RM_num.append(np.sum(pointsz*np.abs(RMdataz[:,i])))
            s_RM_den.append(np.sum(np.abs(RMdataz[:,i])))
            s_plane_i.append(plane_rot(points[i][0], points[i][1], True))
    s_RM_master = [s_LOS_i, s_RM_num, s_RM_den, s_plane_i]
    s_RM_master = np.array(s_RM_master)
    np.save("/scratch/jh2/hs9158/results/data/s_RM_master_{}_{}_{}.npy".format(args.resolution, args.scenario, args.key), s_RM_master)
    # s_RM_master = np.load("/scratch/jh2/hs9158/results/data/s_RM_master_{}_{}_{}.npy".format(args.resolution, args.scenario, args.key), allow_pickle=True)
    stop()

def make_paper_plot_ne_B(args, i):
    args.Tmin = 1e1; args.Tmax = 1e7
    data = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}.npz".format('RM', 200, i, args.Tmin, args.Tmax, args.TRe_key))
    dataEM = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}.npz".format('EM', 200, i, args.Tmin, args.Tmax, args.TRe_key))
    dataz = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}z.npz".format('RM', 200, i, args.Tmin, args.Tmax, args.TRe_key))
    datazEM = np.load(args.snappath+"{}_{}_snap_{}_Tmin_{}_Tmax_{}_T{}z.npz".format('EM', 200, i, args.Tmin, args.Tmax, args.TRe_key))
    EM = dataEM['data'][:, 2]
    indEM = np.where(EM<100)[0]
    points = data['data'][:, 0:2][indEM]
    RM = data['data'][:, 2][indEM]
    
    pointsz = dataz['data'][:, -1]
    RMdataz = dataz['data'][:, 0:-1][:, indEM]
    EMz = datazEM['data'][:, 0:-1][:, indEM]
    dz = pointsz[1]-pointsz[0]
    nez = (EMz/dz)**(1/2)
    Bz = RMdataz/0.81/nez/dz
    # convert all the nan in Bz to 0
    Bz[np.isnan(Bz)] = 0

    # load the RMstats
    s_RM_master = np.load("/scratch/jh2/hs9158/results/data/s_RM_master_{}_{}_{}.npy".format(args.resolution, args.scenario, args.key), allow_pickle=True)
    s_LOS_i, s_RM, sigma_LOS, s_plane_i = analyse_RM_stats(args.resolution, args.scenario, args.key)
    s_LOS_plane = s_LOS_i-s_plane_i
    # for the top three points in s_LOS_plane less than 2000 plot the ne and Bz
    ind = np.zeros(3)
    if args.scenario == 0: scale=10
    else: scale=1
    ind[0] = np.where((s_LOS_plane>500/scale) & (s_LOS_plane<1000/scale))[0][0]
    ind[1] = np.where((s_LOS_plane>1000/scale) & (s_LOS_plane<1500/scale))[0][0]
    ind[2] = np.where((s_LOS_plane>1500/scale) & (s_LOS_plane<2000/scale))[0][0]
    ind = ind.astype(int)
    # stop()
    plt.figure(figsize=(6, 6))
    plt.plot(pointsz-s_plane_i[ind[0]], nez[:, ind[0]], linewidth=2, c='slateblue', label=r'$n_{{\rm e}, z};~\langle z \rangle_{\rm LOS, i}-z_{\rm plane, i}=$'+str(np.round(s_LOS_plane[ind[0]]/1e3, 1))+' kpc')
    plt.plot(pointsz-s_plane_i[ind[0]], Bz[:, ind[0]], linewidth=2, c='salmon', label=r'$B_{{\rm LOS}, z};~\langle z \rangle_{\rm LOS, i}-z_{\rm plane, i}=$'+str(np.round(s_LOS_plane[ind[0]]/1e3, 1))+' kpc')
    plt.plot(pointsz-s_plane_i[ind[1]], nez[:, ind[1]], linewidth=2, c='slateblue', linestyle='--', label=r'$n_{{\rm e}, z};~\langle z \rangle_{\rm LOS, i}-z_{\rm plane, i}=$'+str(np.round(s_LOS_plane[ind[1]]/1e3, 1))+' kpc')
    plt.plot(pointsz-s_plane_i[ind[1]], Bz[:, ind[1]], linewidth=2, c='salmon', linestyle='--', label=r'$B_{{\rm LOS}, z};~\langle z \rangle_{\rm LOS, i}-z_{\rm plane, i}=$'+str(np.round(s_LOS_plane[ind[1]]/1e3, 1))+' kpc')
    plt.plot(pointsz-s_plane_i[ind[2]], nez[:, ind[2]], linewidth=2, c='slateblue', linestyle=':', label=r'$n_{{\rm e}, z};~\langle z \rangle_{\rm LOS, i}-z_{\rm plane, i}=$'+str(np.round(s_LOS_plane[ind[2]]/1e3, 1))+' kpc')
    plt.plot(pointsz-s_plane_i[ind[2]], Bz[:, ind[2]], linewidth=2, c='salmon', linestyle=':', label=r'$B_{{\rm LOS}, z};~\langle z \rangle_{\rm LOS, i}-z_{\rm plane, i}=$'+str(np.round(s_LOS_plane[ind[2]]/1e3, 1))+' kpc')
    plt.xlim([-1e3, 2e3])
    plt.yscale('log')
    plt.xlabel(r'$z - z_{\rm plane, i}~(kpc)$')
    plt.legend()
    # have ne on the left y-axis and Bz on the right y-axis
    ax = plt.gca()
    ax2 = ax.twinx()
    # ax2.set_yscale('log')
    ax2.set_ylabel(r'$B_{{\rm LOS}, z}~(\mu G)$', color='salmon')
    ax2.tick_params(axis='y', labelcolor='salmon')
    ax.set_ylabel(r'$n_{{\rm e}, z}~({\rm cm}^{-3})$', color='slateblue')
    ax.tick_params(axis='y', labelcolor='slateblue')
    ax.set_ylim([1e-4, 1e-1])
    ax2.set_ylim([1e-1, 1e3])
    
    plt.savefig("/scratch/jh2/hs9158/results/paper_plots/ne_Bz_{}.pdf".format(i), dpi=100, bbox_inches='tight')

    
    stop()
    

    

def make_phase_plots(args, i):
    # making phase plot of the gas particles density vs temperature and ne/nH vs temperature
    # check if snapshot_processed exists
    if os.path.exists(args.snappath+"snapshot_{}_processed_quantities_T{}.npy".format(str(i).zfill(3), args.TRe_key)):
        print("Processed quantities already exist for this snapshot.")
        arr = np.load(args.snappath+"snapshot_{}_processed_quantities_T{}.npy".format(str(i).zfill(3), args.TRe_key), allow_pickle=True)
        arrdata = arr.item()
        # stop()
        ne_corrected_photoionzed = arrdata['ne_corrected_photoionized']
        ne_corrected = arrdata['ne_corrected']
        # ne = arrdata['ne']
        T_photo = arrdata['grackle_temperature_photo']
        T = arrdata['grackle_temperature']
        nH = arrdata['hydrogen_number_density']
        ind_T = np.where((T>1e1) & (T<2e4))[0]
    # plt.figure(figsize=(10, 5))
    # plt.subplot(121)
    # plt.scatter(T[ind_T], ne_corrected[ind_T]/nH[ind_T], s=0.1, c='b', alpha=0.1)
    # plt.xlabel('T (K)'); plt.ylabel('ne/nH Corrected')
    # plt.yscale('log')
    # plt.subplot(122)
    # plt.scatter(T_photo[ind_T], ne_corrected_photoionzed[ind_T]/nH[ind_T], s=0.1, c='b', alpha=0.1)
    # plt.xlabel('T (K)'); plt.ylabel('ne/nH Corrected Photoionized')
    # plt.yscale('log')
    # plt.show()
    # stop()
    plot_mass_histogram(T_photo, ne_corrected_photoionzed/nH, x_label=' (T (K))', y_label=r'$~(n_{\rm e}/n_{\rm H})$', bins=400, log=True)
    nH_thresh = np.logspace(-4, -2, 20)
    min_ion = np.zeros(len(nH_thresh))
    min_T = np.zeros(len(nH_thresh))
    plt.figure()
    for i in range(len(nH_thresh)):
        ind = np.where(nH<=nH_thresh[i])[0]
        min_ion[i] = np.min(ne_corrected_photoionzed[ind]/nH[ind])
        min_T[i] = np.min(T_photo[ind])
    plt.subplot(121)
    plt.plot(nH_thresh, min_ion)
    plt.xlabel('nH threshold')
    plt.ylabel('min ne/nH below threshold')
    plt.xscale('log')
    plt.yscale('log')
    plt.subplot(122)
    plt.plot(nH_thresh, min_T)
    plt.xlabel('nH threshold')
    plt.ylabel('min T below threshold')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    stop()

    stop()


def VrPlot_maker(args):
    args.filename = 'snapshot_{}.hdf5'.format(str(args.snapnum).zfill(3))
    ds = yt.load(args.snappath+args.filename)
    ad = ds.all_data()
    coord = ad['PartType0','Coordinates'].in_units('kpc')
    v = ad[('PartType0', 'Velocities')]
    v = v.in_units('km/s')
    v_pol = convert_to_cylindrical(coord, v)
    # plot the 3 components of velocity in cylindrical coordinates on y-axis and radius on x-axis in 3 subplots along row
    fig, axs = plt.subplots(1, 3, figsize=(15*0.7, 5*0.7))
    axs[0].scatter(np.linalg.norm(coord, axis=1), v_pol[:, 0], s=0.1)
    axs[0].set_xlabel('r (kpc)')
    axs[0].set_ylabel('Vr (km/s)')
    axs[0].set_ylim(-50, 50)
    axs[1].scatter(np.linalg.norm(coord, axis=1), v_pol[:, 1], s=0.1)
    axs[1].set_xlabel('r (kpc)')
    axs[1].set_ylabel('Vphi (km/s)')
    axs[1].set_ylim(-100, 500)
    axs[2].scatter(np.linalg.norm(coord, axis=1), v_pol[:, 2], s=0.1)
    axs[2].set_xlabel('r (kpc)')
    axs[2].set_ylabel('Vz (km/s)')
    axs[2].set_ylim(-50, 50)
    return plt.savefig(args.plotpath+'V_pol_n_{}_v5.png'.format(args.snapnum), dpi=300)

# write a function to plot the RM map (using imshow) based on points, and RM arguments
def plot_RM_map(args, points, RM, RMDM):
    "This function plots the RM map using the points and RM arrays"
    "using the imshow function from matplotlib"
    # Make two subplots
    points = np.array(points)
    RM = np.array(RM)
    fac = 0.75
    fig, ax = plt.subplots(2, 1, figsize=(5*fac, 10*fac))
    ax1 = plt.subplot(211)
    color = np.repeat('b', len(points))
    color[RM<0] = 'r'
    ax1.scatter(points[:, 0], points[:,1], s=np.abs(RM), c=color, alpha=0.5)
    ax1.set_xlabel('X (pc)')
    ax1.set_ylabel('Y (pc)')
    ax1.set_aspect('equal')

    # Make the other subplot
    ax2 = plt.subplot(212)
    y, x = rm.histogram(RM, 20)
    ax2.plot(x, y)
    if RMDM == 'DM': ax2.set_xlabel(r'$\rm DM~(pc/cm^3)$')
    if RMDM == 'RM': ax2.set_xlabel(r'$\rm RM~(rad/m^2)$')
    ax2.set_ylabel('Number of sources')
    plt.suptitle('Tmin = {}, Tmax = {}, sigmaRM = {}'.format(args.Tmin, args.Tmax, RM.std()))
    # plt.show()
    # stop()
    plt.savefig(args.plotpath+'{}_map_Tmin_{}_Tmax_{}_{}_{}_T{}_res{}.png'.format(RMDM, args.Tmin, args.Tmax, args.key, args.snapnum, args.TRe_key, len(points)), dpi=300)
    

def PlotDiagnostics(args, snaprange):
    #This function plots the diagnostics of the simulation like mass outflow rate, SFR, temperature#
    # if SFR.txt exists in args.snappath, then
    if os.path.exists(args.snappath+'SFR.txt'): SFR = np.loadtxt(args.snappath+'SFR.txt', delimiter=' ', dtype=float)
    else: SFR = calculate_SFR(args, snaprange)
    
    if os.path.exists(args.snappath+'temperature.txt'): temperature = np.loadtxt(args.snappath+'temperature.txt', delimiter=' ', dtype=float)
    else: temperature = calculate_temperature(args, snaprange)

    # if os.path.exists(args.snappath+'mass_outflow.txt') and os.path.exists(args.snappath+'mass_inflow.txt'):
    #     mass_outflow = np.loadtxt(args.snappath+'mass_outflow.txt', delimiter=' ', dtype=float)
    #     mass_inflow = np.loadtxt(args.snappath+'mass_inflow.txt', delimiter=' ', dtype=float)
    # else:
    #     mass_outflow, mass_inflow = calculate_mass_outflow_rate(args, snaprange, 9.6, 2)
    fig, ax = plt.subplots(2, 1, figsize=(10, 20))
    # add a zero in the beginning to the SFR array to make it the same length as the other arrays
    SFR = np.insert(SFR, 0, 0) # to make the length of SFR same as snaprange (required as the SFR for first snapshot is zero)

    ax[0].plot(np.arange(snaprange)*args.tstep/1e6, SFR)
    ax[0].set_ylabel('SFR (Msol/yr)')
    ax[0].set_yscale('log')
    ax[0].set_ylim([1e-2, 1e1])
    ax[1].plot(np.arange(snaprange)*args.tstep/1e6, temperature)
    ax[1].set_ylabel('Temperature (K)')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Time (Myr)')

    # ax[2].plot(mass_outflow, label='mass_outflow')
    # ax[2].plot(mass_inflow, label='mass_inflow')
    # ax[2].set_ylabel('Mass (Msol)')
    # ax[2].legend()
    # make the x-axis common for all and no hspace
    plt.subplots_adjust(hspace=0)
    
    plt.show()

def create_scatterRM_plot(args, data, extent, resolution, R, R_obs):
    "This function creates an RM map for a given snapshot based on the resolution and range and rotation matrix"
    "using the functions from the rayMaker.py code"
    plt.rcParams.update({'figure.constrained_layout.use': False})
    i = args.snapnum
    ptype = "PartType0"
    data_ions = rm.CorrectIons(args, data)
    ad_ion_rot = data_ions.all_data()
    ad_ion_rot = rm.RotateGalaxy(args, ad_ion_rot, R, center=args.center)
    ad_ion_rot = rm.RotateGalaxy(args, ad_ion_rot, R_obs, center=args.center)  # rotate to observer's frame
    coords = ad_ion_rot[ptype, 'Coordinates'].in_units('pc')-(args.center*1e3)*unyt.pc
    rho = ad_ion_rot[ptype, 'Density'].in_units('g/cm**3')
    B = ad_ion_rot[ptype, 'MagneticField']*1e6
    V = ad_ion_rot[ptype, 'Velocities'].in_units('km/s')
    molecular_weight = ad_ion_rot[ptype, 'molecular_weight']*u.mh
    # ParticleID = ad_ion_rot[ptype, 'ParticleIDs']
    # ind_disk = np.where((ParticleID > 0) & (ParticleID < args.ngasdisk))[0]
    # plt.figure(); plt.scatter(coords[ind_disk][:,1], coords[ind_disk][:,2], s=0.1, alpha=0.1); plt.show()
    sml = ad_ion_rot[ptype, 'SmoothingLength'].in_units('pc')
    np.random.seed(0)
    points = np.column_stack((np.random.uniform(low=-extent*1e3/2, high=extent*1e3/2, size=int(resolution)),
                        np.random.uniform(low=-extent*1e3/2, high=extent*1e3/2, size=int(resolution)))) # in pc

    rayLength = 100000 # in parsec
    bins = 10000
    ne = ad_ion_rot[ptype, 'ne']
    BLOS = ad_ion_rot[ptype, 'MagneticField'][:,2]*1e6 # in uG
    args_list = [(point) for point in points]
    # Create a Pool object
    with Pool() as pool:
        results = pool.map(partial(rm.rayInterpolate, 
                                   coordinate=coords, 
                                   rayLength=rayLength, 
                                   bins=bins, 
                                   sml=sml, 
                                   ne=ne, 
                                   BLOS=BLOS), args_list)

    # unpack the results
    points, RM, pointsz, RMz = zip(*results)
    # make a subplot with 4 plots in (2*2 grid) on left and a big plot on right
    # Define the figure and subplots
    fac = 1
    fig = plt.figure(figsize=(10*fac, 15*fac))
    # Create a GridSpec with 3 rows and 2 columns
    gs = gridspec.GridSpec(3, 2, figure=fig)
    # create the subplots
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    axs = np.array([ax0, ax1, ax2, ax3, ax4, ax5]).reshape(3, 2)
    
    # # without miltiprocessing
    # points = [], RM = [], pointsz = [], RMz = []
    # for arguments in args_list:
    #     point, RM, pointz, RMz = rm.rayInterpolate(arguments, coords, rayLength, bins, sml, ne, BLOS)
    #     points = np.vstack((points, point))
    #     RM = np.vstack((RM, RM))
    #     pointsz = np.vstack((pointsz, pointz))
    #     RMz = np.vstack((RMz, RMz))
    # radial velocity
    coord = np.array(coords)
    # Vr = np.sum(coord*V, axis=1)/np.linalg.norm(coord, axis=1)
    LOS = 30e3  # in pc
    res = 50
    X, Y = np.meshgrid(np.linspace(-extent*1e3/1.8, extent*1e3/1.8, res), np.linspace(-extent*1e3/1.8, extent*1e3/1.8, res))
    # plane z
    axis1=0; axis2=1; axisLOS=2
    ind_proj = np.where((np.abs(coord[:, axis1]) <= X.max()) & (np.abs(coord[:, axis2]) <= Y.max()) & (np.abs(coord[:, axisLOS])<=LOS/2))[0]
    dens_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,axis1], coord[ind_proj][:,axis2], bins=res, weights=rho[ind_proj]/molecular_weight[ind_proj], range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    num_counts, xedges, yedges = np.histogram2d(coord[ind_proj][:,axis1], coord[ind_proj][:,axis2], bins=res, range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    num_counts[num_counts==0] = 1
    dens_proj = dens_proj/num_counts
    # make dens_proj = 0 to dens_proj = 1 to avoid division by zero
    # dens_proj[dens_proj==0] = 1

    # plot the ne_proj overlayed with B streamplot in (1, 1), dens_slice in (1, 2), and same quantities in x plane in (2, 1) and (2, 2)
    p1 = axs[0][0].pcolormesh(X, Y, dens_proj.T, norm=colors.LogNorm(vmin=1e-7,vmax=1e2), cmap=cm.viridis)
    axs[0][0].set_aspect('equal')
    axs[0][0].set_title(r'$\rm Density~(particles~cm^{-3})$')
    color = np.repeat('b', len(points))
    color[np.array(RM)<0] = 'r'
    # axs[5].scatter(np.array(points)[:, 0], np.array(points)[:,1], s=np.abs(RM), c=color, alpha=0.5)
    # axs[0].scatter(np.array(points)[:, 0], np.array(points)[:,1], s=0.1, c=RM, cmap='coolwarm')
    print("Density projection plotted")
    # make a projection for ne now
    ne_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,axis1], coord[ind_proj][:,axis2], bins=res, weights=ne[ind_proj], range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    ne_proj = ne_proj/num_counts
    p2 = axs[0][1].pcolormesh(X, Y, ne_proj.T, norm=colors.LogNorm(vmin=1e-7,vmax=1e2), cmap=cm.viridis)
    axs[0][1].set_aspect('equal')
    axs[0][1].set_title(r'$\rm ne~(cm^{-3})$')
    # streamplot of B in x-y plane
    BX_proj = np.histogram2d(coord[ind_proj][:,axis1], coord[ind_proj][:,axis2], bins=res, weights=B[ind_proj][:, axis1], range=[[X.min(), X.max()], [Y.min(), Y.max()]])[0]/num_counts
    BY_proj = np.histogram2d(coord[ind_proj][:,axis1], coord[ind_proj][:,axis2], bins=res, weights=B[ind_proj][:, axis2], range=[[X.min(), X.max()], [Y.min(), Y.max()]])[0]/num_counts
    axs[0][1].streamplot(X, Y, BX_proj.T, BY_proj.T, color='r', linewidth=0.5, density=2, arrowsize=0.75)
    print("B streamplot plotted")
    
    # plane x
    axis1=1; axis2=2; axisLOS=0
    ind_proj = np.where((np.abs(coord[:, axis1]) <= X.max()) & (np.abs(coord[:, axis2]) <= Y.max()) & (np.abs(coord[:, axisLOS])<=LOS/2))[0]
    dens_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,axis1], coord[ind_proj][:,axis2], bins=res, weights=rho[ind_proj]/molecular_weight[ind_proj], range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    num_counts, xedges, yedges = np.histogram2d(coord[ind_proj][:,axis1], coord[ind_proj][:,axis2], bins=res, range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    num_counts[num_counts==0] = 1
    dens_proj = dens_proj/num_counts
    # make dens_proj = 0 to dens_proj = 1 to avoid division by zero
    # dens_proj[dens_proj==0] = 1

    # plot the ne_proj overlayed with B streamplot in (1, 1), dens_slice in (1, 2), and same quantities in x plane in (2, 1) and (2, 2)
    p1 = axs[1][0].pcolormesh(X, Y, dens_proj.T, norm=colors.LogNorm(vmin=1e-7,vmax=1e2), cmap=cm.viridis)
    axs[1][0].set_aspect('equal')
    axs[1][0].set_title(r'$\rm Density~(particles~cm^{-3})$')
    color = np.repeat('b', len(points))
    color[np.array(RM)<0] = 'r'
    # axs[5].scatter(np.array(points)[:, 0], np.array(points)[:,1], s=np.abs(RM), c=color, alpha=0.5)
    # axs[0].scatter(np.array(points)[:, 0], np.array(points)[:,1], s=0.1, c=RM, cmap='coolwarm')
    print("Density projection plotted")
    # make a projection for ne now
    ne_proj, xedges, yedges = np.histogram2d(coord[ind_proj][:,axis1], coord[ind_proj][:,axis2], bins=res, weights=ne[ind_proj], range=[[X.min(), X.max()], [Y.min(), Y.max()]])
    ne_proj = ne_proj/num_counts
    p2 = axs[1][1].pcolormesh(X, Y, ne_proj.T, norm=colors.LogNorm(vmin=1e-7,vmax=1e2), cmap=cm.viridis)
    axs[1][1].set_aspect('equal')
    axs[1][1].set_title(r'$\rm ne~(cm^{-3})$')
    # streamplot of B in x-y plane
    BX_proj = np.histogram2d(coord[ind_proj][:,axis1], coord[ind_proj][:,axis2], bins=res, weights=B[ind_proj][:, axis1], range=[[X.min(), X.max()], [Y.min(), Y.max()]])[0]/num_counts
    BY_proj = np.histogram2d(coord[ind_proj][:,axis1], coord[ind_proj][:,axis2], bins=res, weights=B[ind_proj][:, axis2], range=[[X.min(), X.max()], [Y.min(), Y.max()]])[0]/num_counts
    axs[1][1].streamplot(X, Y, BX_proj.T, BY_proj.T, color='r', linewidth=0.5, density=2, arrowsize=0.75)
    print("B streamplot plotted")
    axs[0][0].set_xlabel("X (pc)"); axs[0][0].set_ylabel("Y (pc)")
    axs[0][1].set_xlabel("X (pc)"); axs[0][1].set_ylabel("Y (pc)")
    axs[1][0].set_xlabel("Y (pc)"); axs[1][0].set_ylabel("Z (pc)")
    axs[1][1].set_xlabel("Y (pc)"); axs[1][1].set_ylabel("Z (pc)")
    # remove whitespace
    # plt.subplots_adjust(hspace=0, wspace=0)
    # plt.show()
    # plt.savefig(args.plotpath+'gasB_observer_{}_c{}.png'.format(i, args.center_method), dpi=300)


    # Make another figure with the in plane and LOS RM measurement
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    color = np.repeat('b', len(points))
    color[np.array(RM)<0] = 'r'
    axs[2][1].scatter(np.array(points)[:, 0], np.array(points)[:,1], s=np.abs(np.array(RM))/3, c=color, alpha=0.5)
    axs[2][1].set_xlabel('X (pc)')
    axs[2][1].set_ylabel('Y (pc)')
    axs[2][1].set_xlim(X.min(), X.max())
    axs[2][1].set_ylim(Y.min(), Y.max())
    axs[2][1].set_aspect('equal')
    # RM plot now
    for n in range(len(np.array(pointsz))):
        axs[2][0].plot(np.array(pointsz)[n][:, 2], np.array(RMz)[n], label='{}'.format(np.round(np.array(points)/1000, 2)[n]))
    # set y scale log
    # axs[4].set_yscale('log')
    axs[2][0].set_xlabel('Z (pc)')
    axs[2][0].set_ylabel('RM')
    axs[2][0].set_xlim(-10000, 10000)
    # axs[2][1].set_aspect('equal')
    axs[2][0].legend(fontsize=4)  # Adjust legend font size
    plt.tight_layout()
    plt.show()
    # save figure with tight layout
    fig.savefig(args.plotpath+'gasplot_scatter_RM_LOS_{}_c{}.png'.format(i, args.center_method), dpi=300)
    stop()

def plotRMz_T(args, extent, resolution, RMDM, ind_cond='all'): # ind_cond can be 'all', 'max', 'min'
    rm.get_plane_rotation(args, args.rotation_matrix)
    f = args.plane_rot
    stop()
    args.Tmin = 1e1; args.Tmax = 1e7
    dataz = np.load("/scratch/jh2/hs9158/results/data/{}{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T37z.npz".format(RMDM, resolution, args.resolution, args.scenario, args.key, args.snapnum, args.Tmin, args.Tmax, args.key))
    data = np.load("/scratch/jh2/hs9158/results/data/{}{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T37.npz".format(RMDM, resolution, args.resolution, args.scenario, args.key, args.snapnum, args.Tmin, args.Tmax, args.key))
    pointsz = dataz['data'][:, -1]
    RM = data['data'][:, 2]
    if ind_cond == 'all':
        for ind_max_RM in range(len(RM)):
            print("i = ", ind_max_RM)
            point = data['data'][:,0:2][ind_max_RM]
            RMz_total = dataz['data'][:, ind_max_RM]
            args.Tmin = 1e1; args.Tmax = 8e3
            dataz_CNM = np.load("/scratch/jh2/hs9158/results/data/{}{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T37z.npz".format(RMDM, resolution, args.resolution, args.scenario, args.key, args.snapnum, args.Tmin, args.Tmax, args.key))
            RMz_CNM = dataz_CNM['data'][:, ind_max_RM]
            RM_CNM = data['data'][:, 2]
            args.Tmin = 8e3; args.Tmax = 2e4
            dataz_WIM = np.load("/scratch/jh2/hs9158/results/data/{}{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T37z.npz".format(RMDM, resolution, args.resolution, args.scenario, args.key, args.snapnum, args.Tmin, args.Tmax, args.key))
            RM_WIM = data['data'][:, 2]
            RMz_WIM = dataz_WIM['data'][:, ind_max_RM]
            args.Tmin = 2e4; args.Tmax = 1e5
            dataz_HIM = np.load("/scratch/jh2/hs9158/results/data/{}{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T37z.npz".format(RMDM, resolution, args.resolution, args.scenario, args.key, args.snapnum, args.Tmin, args.Tmax, args.key))
            RMz_HIM = dataz_HIM['data'][:, ind_max_RM]
            RM_HIM = data['data'][:, 2]
            args.Tmin = 1e5; args.Tmax = 1e7
            dataz_HIIM = np.load("/scratch/jh2/hs9158/results/data/{}{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T37z.npz".format(RMDM, resolution, args.resolution, args.scenario, args.key, args.snapnum, args.Tmin, args.Tmax, args.key))
            RMz_HIIM = dataz_HIIM['data'][:, ind_max_RM]
            RM_HIIM = data['data'][:, 2]
            plane_center = f(point[0], point[1], True)
            z = dataz_HIM['data'][:, -1]
            # plot the RMz_total, RMz_CNM, RMz_WIM, RMz_HIM as a function of z
            fig, ax = plt.subplots()
            ax.plot(z, RMz_total, label='Total')
            ax.plot(z, RMz_CNM, label='CNM')
            ax.plot(z, RMz_WIM, label='WIM')
            ax.plot(z, RMz_HIM, label='HIM')
            ax.plot(z, RMz_HIIM, label='HIIM')
            ax.vlines(plane_center, -1000, 1000, color='k', linestyle='--')
            ax.set_ylim(-3, 3)
            ax.set_xlim(-500+plane_center, 500+plane_center)
            ax.set_xlabel('z (pc)')
            ax.set_ylabel('RMz (rad/m**2)')
            ax.legend()
            plt.title('point = {} pc, Total RM = {}'.format(np.round(point,2), RM[ind_max_RM]))
            plt.savefig(args.plotpath+"RMz_T_{}_{}_snap{}.png".format(ind_cond, ind_max_RM, args.snapnum), dpi=300)
    elif ind_cond == 'max': ind_max_RM = RM.argmax()
    else: ind_max_RM = RM.argmin()
    point = data['data'][:,0:2][ind_max_RM]
    RMz_total = dataz['data'][:, ind_max_RM]
    args.Tmin = 1e1; args.Tmax = 8e3
    dataz_CNM = np.load("/scratch/jh2/hs9158/results/data/{}{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T37z.npz".format(RMDM, resolution, args.resolution, args.scenario, args.key, args.snapnum, args.Tmin, args.Tmax, args.key))
    RMz_CNM = dataz_CNM['data'][:, ind_max_RM]
    RM_CNM = data['data'][:, 2]
    args.Tmin = 8e3; args.Tmax = 2e4
    dataz_WIM = np.load("/scratch/jh2/hs9158/results/data/{}{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T37z.npz".format(RMDM, resolution, args.resolution, args.scenario, args.key, args.snapnum, args.Tmin, args.Tmax, args.key))
    RM_WIM = data['data'][:, 2]
    RMz_WIM = dataz_WIM['data'][:, ind_max_RM]
    args.Tmin = 2e4; args.Tmax = 1e5
    dataz_HIM = np.load("/scratch/jh2/hs9158/results/data/{}{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T37z.npz".format(RMDM, resolution, args.resolution, args.scenario, args.key, args.snapnum, args.Tmin, args.Tmax, args.key))
    RMz_HIM = dataz_HIM['data'][:, ind_max_RM]
    RM_HIM = data['data'][:, 2]
    args.Tmin = 1e5; args.Tmax = 1e7
    dataz_HIIM = np.load("/scratch/jh2/hs9158/results/data/{}{}_{}_scenario{}{}_snap_{}_Tmin_{}_Tmax_{}_{}_T37z.npz".format(RMDM, resolution, args.resolution, args.scenario, args.key, args.snapnum, args.Tmin, args.Tmax, args.key))
    RMz_HIIM = dataz_HIIM['data'][:, ind_max_RM]
    RM_HIIM = data['data'][:, 2]
    plane_center = f(point[0], point[1], True)
    z = dataz_HIM['data'][:, -1]
    # plot the RMz_total, RMz_CNM, RMz_WIM, RMz_HIM as a function of z
    fig, ax = plt.subplots()
    ax.plot(z, RMz_total, label='Total')
    ax.plot(z, RMz_CNM, label='CNM')
    ax.plot(z, RMz_WIM, label='WIM')
    ax.plot(z, RMz_HIM, label='HIM')
    ax.plot(z, RMz_HIIM, label='HIIM')
    ax.vlines(plane_center, -1000, 1000, color='k', linestyle='--')
    ax.set_ylim(-3, 3)
    ax.set_xlim(-500+plane_center, 500+plane_center)
    ax.set_xlabel('z (pc)')
    ax.set_ylabel('RMz (rad/m**2)')
    plt.title('point = {}'.format(np.round(point,2)))
    ax.legend()
    plt.savefig(args.plotpath+"RMz_T_{}_snap{}.png".format(ind_cond, args.snapnum), dpi=300)
    # plt.show()
    stop()

def test_rotation_matrix(args, data, R, R_obs, center):
    v_lmc = np.array([0, 0, np.linalg.norm(args.V_LMC)])
    ad = data.all_data()
    ad = rm.RotateGalaxy(args, ad, R, center=np.array([100, 100, 100]))
    coord = ad['PartType0','Coordinates'].in_units('kpc')
    stop()
    v_LMC_rot = np.dot(R, v_lmc)
    v = ad[('PartType0', 'Velocities')].in_units('km/s')
    v = np.array(v)-v_LMC_rot
    ParticleID = ad['PartType0', 'ParticleIDs']
    ind_disk = np.where((ParticleID > 0) & (ParticleID < args.ngasdisk))[0]
    # choose 5000 random points from the disk
    np.random.seed(0)
    ind_disk_plot = np.random.choice(ind_disk, 500)
    print("Plotting the disk after aligning its angular momentum with z-axis")
    plt.figure(figsize=(7, 7))
    plt.scatter(np.array(coord[ind_disk_plot][:,0]), np.array(coord[ind_disk_plot][:,1]), s=5, alpha=0.25)
    # plot the velocity vectors of ind_disk_plot particles at their positions using quiver
    plt.quiver(np.array(coord[ind_disk_plot][:,0]), np.array(coord[ind_disk_plot][:,1]),
                np.array(v[ind_disk_plot][:,0]), np.array(v[ind_disk_plot][:,1]), scale=8000)
    L = np.cross(np.array(coord[ind_disk_plot])-center, v[ind_disk_plot]); L = L/np.linalg.norm(L, axis=1)[:, np.newaxis]; L_avg = np.mean(L, axis=0)
    plt.title("Average angular momentum vector = {}, V_LMC = {}".format(np.round(L_avg,2), np.round(v_LMC_rot,2)))
    plt.xlim(50, 150); plt.ylim(50, 150); plt.show()
    ad = rm.RotateGalaxy(args, ad, R_obs, center=np.array([100, 100, 100]))
    v_LMC_rot2 = np.dot(R_obs, v_LMC_rot)
    coord = ad['PartType0','Coordinates'].in_units('kpc')
    v = ad[('PartType0', 'Velocities')].in_units('km/s')
    v = np.array(v)-v_LMC_rot2
    ParticleID = ad['PartType0', 'ParticleIDs']
    ind_disk = np.where((ParticleID > 0) & (ParticleID < args.ngasdisk))[0]
    print("Plotting the disk after aligning it in the observer's frame of reference")
    plt.figure(figsize=(7, 7))
    plt.scatter(np.array(coord[ind_disk_plot][:,0]), np.array(coord[ind_disk_plot][:,1]), s=5, alpha=0.25)
    plt.quiver(np.array(coord[ind_disk_plot][:,0]), np.array(coord[ind_disk_plot][:,1]),
                np.array(v[ind_disk_plot][:,0]), np.array(v[ind_disk_plot][:,1]), scale=8000)
    L = np.cross(np.array(coord[ind_disk_plot])-center, v[ind_disk_plot]); L = L/np.linalg.norm(L, axis=1)[:, np.newaxis]; L_avg = np.mean(L, axis=0)
    plt.title("Average angular momentum vector = {}, V_LMC = {}".format(np.round(L_avg,2), np.round(v_LMC_rot2,2)))
    plt.xlim(50, 150); plt.ylim(50, 150); plt.show()
    # calculate angular momentum vectors from positions and velocities
    
    
    
    stop()

def reynolds_test(args):
    # load from args.snappath
    T4 = np.load(args.snappath+'snapshot_100_processed_quantities_T4.npy', allow_pickle=True)
    T4 = T4.item()
    ne4 = T4['ne_corrected_photoionized']
    np.median(ne4[np.where((T4['grackle_temperature_photo']>10) & (T4['grackle_temperature_photo']<2e4))[0]])

    T41 = np.load(args.snappath+'snapshot_100_processed_quantities_T41.npy', allow_pickle=True)
    T41 = T41.item()
    ne41 = T41['ne_corrected_photoionized']
    np.median(ne41[np.where((T41['grackle_temperature_photo']>10) & (T41['grackle_temperature_photo']<2e4))[0]])

    T42 = np.load(args.snappath+'snapshot_100_processed_quantities_T42.npy', allow_pickle=True)
    T42 = T42.item()
    ne42 = T42['ne_corrected_photoionized']
    np.median(ne42[np.where((T42['grackle_temperature_photo']>10) & (T42['grackle_temperature_photo']<2e4))[0]])

    T425 = np.load(args.snappath+'snapshot_100_processed_quantities_T425.npy', allow_pickle=True)
    T425 = T425.item()
    ne425 = T425['ne_corrected_photoionized']
    np.median(ne425[np.where((T425['grackle_temperature_photo']>10) & (T425['grackle_temperature_photo']<2e4))[0]])

    T43 = np.load(args.snappath+'snapshot_100_processed_quantities_T43.npy', allow_pickle=True)
    T43 = T43.item()
    ne43 = T43['ne_corrected_photoionized']

    stop()

def check_slug_luminosities(args, snap):
    # load the processed quantities (essentially your ionising luminosities here)
    data = np.load(args.snappath+'snapshot_{}_processed_quantities_T{}.npy'.format(snap, args.TRe_key), allow_pickle=True)
    data = data.item()
    F = h5py.File(args.snappath+'snapshot_{}.hdf5'.format(snap), 'r')
    age = F['PartType4']['SlugStateDouble'][:, 13]/1e6
    ionising_luminosity = data['ionizing_luminosity']  # in pho
    plt.figure(); plt.scatter(age, ionising_luminosity[:,1], s=1, alpha=0.5); plt.yscale('log'); plt.xlabel("age (Myr)"); plt.ylabel("ionising luminosity");plt.savefig("temp_lum.png", dpi=300)
    stop()


if __name__ == '__main__':
    print("Starting plotting")
