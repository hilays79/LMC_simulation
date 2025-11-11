#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Hilay Shah 2023-

# import all modules being used
import numpy as np
import matplotlib.pyplot as plt
from cfpack import stop, constants
from scipy.interpolate import RegularGridInterpolator
import os
from matplotlib.colors import LogNorm
import math

def make_different_cloudy_runs(T, extinguish=False, ext_den=21):
    # read the base input cloudy script
    # change the temperature
    # save and run the cloudy script
    if extinguish:
        file_path = "/scratch/jh2/hs9158/gizmo-fork/gizmo2/cloudy-master/my_cloudy/my_ism_grid_ext.in"
        key = "ext"
        print("Extinguishing by a column of {} cm**-2".format(ext_den))
    else:
        file_path = "/scratch/jh2/hs9158/gizmo-fork/gizmo2/cloudy-master/my_cloudy/my_ism_grid.in"
        key = "noext"
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            if lines[i].startswith("constant temperature"):
                lines[i] = "constant temperature {}\n".format(np.log10(T))
            if lines[i].startswith("extinguish"):
                lines[i] = "extinguish by a column of {} \n".format(ext_den)

        # save this file with a different name
        if extinguish:
            file_path = "/scratch/jh2/hs9158/gizmo-fork/gizmo2/cloudy-master/my_cloudy/my_ism_grid_T{}_{}den{}.in".format(T, key, ext_den)
        else:
            file_path = "/scratch/jh2/hs9158/gizmo-fork/gizmo2/cloudy-master/my_cloudy/my_ism_grid_T{}_{}.in".format(T, key)
        with open(file_path, 'w') as file:
            file.writelines(lines)
    # run the cloudy script
    os.chdir("/scratch/jh2/hs9158/gizmo-fork/gizmo2/cloudy-master/my_cloudy/")
    print("Running Cloudy for T={} K".format(T))
    os.system("cloudy my_ism_grid_T{}_{}den{}".format(T, key, ext_den))
    print("Cloudy run for T={} K is for {} complete".format(T, key))
    return None

def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

def find_and_read_file(directory, temperature, key, ext_den):
    partial_name = f"my_ism_grid_T{temperature:.1f}"
    
    for filename in os.listdir(directory):
        if filename.startswith(partial_name) and f"_{key}den{ext_den}.ion" in filename:
            full_path = os.path.join(directory, filename)
            return full_path
    
    return None, None

def parse_ionization_file(file_path):
    element_names, abundances, mass_numbers = ISM_abundances()
    ionization_data = []
    for i in range(len(element_names)):
        ionization_data.append([])
    ion_size = 3
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for l in range(len(lines)):
            # Skip empty lines and delimiters
            if lines[l].strip() == "" or "GRID_DELIMIT" in lines[l]:
                continue
            # Extract element name and ionization values using string slicing
            for e in range(len(element_names)):
                if lines[l].startswith(" {}".format(element_names[e])):
                    element = element_names[e]
                    ion_element = np.zeros(ion_size)
                    for k in range(ion_size):
                        ion_element[k] = float(lines[l][11+7*k:11+7*(k+1)].strip())
                    ionization_data[e].append(ion_element)
    return ionization_data

def ISM_abundances():
    # open ISM abundances from file ISM.abn in /scratch/jh2/hs9158/gizmo-fork/gizmo2/cloudy-master/data/abundances
    # then compare it with the abundances from the ISM.ini file in /scratch/jh2/hs9158/gizmo-fork/gizmo2/cloudy-master/my_cloudy directory
    # and return the abundances in the ISM.abn file
    filepath1 = "/scratch/jh2/hs9158/gizmo-fork/gizmo2/cloudy-master/data/abundances/my_ISM.txt"
    filepath2 = "/scratch/jh2/hs9158/gizmo-fork/gizmo2/cloudy-master/my_cloudy/ISM.ini"
    with open(filepath1, 'r') as file:
        lines = file.readlines()
        abundances = []
        element_names = []
        mass_numbers = []
        for line in lines:
            # split the line into elements and their abundances with a space delimiter
            element, abundance, mass = line.split()
            element_names.append(element)
            abundances.append(float(abundance))
            mass_numbers.append(float(mass))
    return element_names, abundances, mass_numbers


def compute_quantities_from_ion(density, temperature, extinguish=False, ext_den=21):
    if extinguish:
        key = "ext"
        file_path = find_and_read_file("/scratch/jh2/hs9158/gizmo-fork/gizmo2/cloudy-master/my_cloudy/", truncate(temperature,1), key, ext_den)
        # stop()
        # file_path = "/scratch/jh2/hs9158/gizmo-fork/gizmo2/cloudy-master/my_cloudy/my_ism_grid_T{}_{}den{}.ion".format(temperature, key, ext_den)
    else:
        key = "noext"
        file_path = find_and_read_file("/scratch/jh2/hs9158/gizmo-fork/gizmo2/cloudy-master/my_cloudy/", truncate(temperature,1), key, ext_den)
        # file_path = "/scratch/jh2/hs9158/gizmo-fork/gizmo2/cloudy-master/my_cloudy/my_ism_grid_T{}_{}.ion".format(temperature, key)

    print("Parsing ionization file")
    ionization_data = parse_ionization_file(file_path)
    print("Ionization file parsed")
    ionization_data = np.array(ionization_data)
    og_ionization_data = ionization_data
    ion_size = 3
    densities = density
    element_names, abundances, mass_numbers = ISM_abundances()
    abundances = np.array(abundances)
    ionization_states = np.arange(ion_size)  # Example ionization states
    ionization_data = 10**ionization_data
    abundances = abundances[:, np.newaxis, np.newaxis]
    # Reshape densities to (1, 61, 1) to match the second dimension and allow broadcasting
    densities = densities[np.newaxis, :, np.newaxis]
    # Reshape ionization_states to (1, 1, 3) to match the third dimension and allow broadcasting
    ionization_states = ionization_states[np.newaxis, np.newaxis, :]
    # Perform the element-wise multiplication
    result = ionization_data * abundances * densities * ionization_states
    ne = np.sum(result, axis=(0, 2))
    weighted_abundances = abundances * np.array(mass_numbers)[:, np.newaxis, np.newaxis]
    mu = np.sum(weighted_abundances)*constants.m_p / (np.sum(abundances)+ne[np.newaxis, :, np.newaxis]/densities)
    n_tot = np.sum(ionization_data * densities * abundances, axis=(0, 2))
    nH = densities[0, :, 0]
    ne_nH = ne / nH; ne_n = ne / n_tot
    return nH,  n_tot, ne, ne_nH, ne_n, mu[0, :, 0]


def ion_density_temperature(density=np.logspace(-3, 3, 61), temperature=np.logspace(1, 4.3, 71), extinguish=True, ext_den=21):
    X = 0.73; Y = 0.25; Z = 0.02 # mass densities
    ne_n_grid = []; mu_grid = []; ne_nH_grid = []; ne_grid = []
    for i in range(len(temperature)):
        nH, n, ne, ne_nH, ne_n, mu = compute_quantities_from_ion(density, temperature[i], extinguish=extinguish, ext_den=ext_den)
        ne_n_grid.append(ne_n); mu_grid.append(mu); ne_nH_grid.append(ne_nH); ne_grid.append(ne)
    ne_n_grid = np.array(ne_n_grid)
    mu_grid = np.array(mu_grid)
    ne_nH_grid = np.array(ne_nH_grid)
    ne_grid = np.array(ne_grid)
    # create a pcolormesh plot of ne_n_grid VS temperature and density
    # plt.figure(figsize=(18, 5))
    # # make a 1X2 subplot
    # plt.subplot(1, 3, 1)
    # plt.pcolormesh(np.log10(temperature), np.log10(density), ne_n_grid.T, cmap='viridis', norm=LogNorm())
    # plt.xlabel('log(Temperature) (K)')
    # plt.ylabel('log(Density) (cm**-3)')
    # plt.colorbar()
    # plt.title('ne/n')
    # # make a 1X2 subplot
    # plt.subplot(1, 3, 2)
    # plt.pcolormesh(np.log10(temperature), np.log10(density), mu_grid.T/constants.m_p, cmap='viridis', norm=LogNorm())
    # plt.xlabel('log(Temperature) (K)')
    # plt.ylabel('log(Density) (cm**-3)')
    # plt.colorbar()
    # plt.title('mu')
    # # make a 1X2 subplot
    # plt.subplot(1, 3, 3)
    # plt.pcolormesh(np.log10(temperature), np.log10(density), ne_nH_grid.T, cmap='viridis', norm=LogNorm())
    # plt.xlabel('log(Temperature) (K)')
    # plt.ylabel('log(Density) (cm**-3)')
    # plt.colorbar()
    # plt.title('ne/nH')
    # plt.suptitle("extinguish = {}, dens={}".format(extinguish, ext_den))
    # plt.show()

    # interpolate ne_grid with RegularGridInterpolator
    ne_interp = RegularGridInterpolator((density, temperature), ne_grid.T)
    # plot the interpolated ne_grid and real ne_grid in two subplots
    # plt.figure(figsize=(18, 5))
    # # make a 1X2 subplot
    # plt.subplot(1, 3, 1)
    # # create mesh out of density and temperature and then create a pcolormesh plot of ne_interp VS temperature and density
    # density, temperature = np.meshgrid(density, temperature)
    # ne_interp_arr = ne_interp((density, temperature))
    # plt.pcolormesh(np.log10(temperature), np.log10(density), ne_interp_arr, cmap='viridis', norm=LogNorm())
    # plt.xlabel('log(Temperature) (K)')
    # plt.ylabel('log(Density) (cm**-3)')
    # plt.colorbar()
    # plt.title('ne_interpolated')
    # # make a 1X2 subplot
    # plt.subplot(1, 3, 2)
    # plt.pcolormesh(np.log10(temperature), np.log10(density), ne_grid, cmap='viridis', norm=LogNorm())
    # plt.xlabel('log(Temperature) (K)')
    # plt.ylabel('log(Density) (cm**-3)')
    # plt.colorbar()
    # plt.title('ne')
    # # make a 1X2 subplot with a log scale colorbar
    # plt.subplot(1, 3, 3)
    # plt.pcolormesh(np.log10(temperature), np.log10(density), ne_nH_grid, cmap='viridis', norm=LogNorm())
    # plt.xlabel('log(Temperature) (K)')
    # plt.ylabel('log(Density) (cm**-3)')
    # plt.colorbar()
    # plt.title('ne/nH')
    # plt.suptitle("extinguish = {}, dens={}".format(extinguish, ext_den))
    # plt.show()
    return ne_interp

if __name__ == "__main__":
    stop()