########################################################################
#
# Cooling rate script - modified by Hilay Shah
#
#
# Copyright (c) 2013-2016, Grackle Development Team.
#
# Distributed under the terms of the Enzo Public Licence.
#
# The full license is in the file LICENSE, distributed with this
# software.
########################################################################

from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import os
import yt
from cfpack import stop
from joblib import Parallel, delayed
import multiprocessing
import dill as pickle
import scipy.optimize as optimize
from scipy import interpolate

from pygrackle import \
    chemistry_data, \
    setup_fluid_container

from pygrackle.utilities.physical_constants import \
    mass_hydrogen_cgs, \
    sec_per_Myr, \
    cm_per_mpc

# loop arg could take 'rho' or 'temp' as input
def find_thermal_equillibrium(dens = np.logspace(-5.5, 3.5, 250, dtype=np.float64)*mass_hydrogen_cgs, 
                              temp = np.logspace(1.5, 4.5, 250), metallicity=0.5, loop='rho',
                              shield_method=0, h2_shield=0, photo_heat=1): # metallicity in units of Zsolar (0.014)
    current_redshift = 0.

    # Set solver parameters
    my_chemistry = chemistry_data()
    my_chemistry.use_grackle = 1
    my_chemistry.with_radiative_cooling = 0
    my_chemistry.primordial_chemistry = 0
    my_chemistry.metal_cooling = 1
    my_chemistry.UVbackground = 1
    my_chemistry.self_shielding_method = shield_method # check
    my_chemistry.H2_self_shielding = h2_shield
    my_chemistry.photoelectric_heating = photo_heat
    my_dir = os.path.dirname(os.path.abspath(__file__))
    grackle_data_file = bytearray(os.path.join(
        "/scratch/jh2/hs9158/gizmo-fork/gizmo2/grackle/input", "CloudyData_UVB=HM2012_shielded.h5"), 'utf-8')
    my_chemistry.grackle_data_file = grackle_data_file

    my_chemistry.use_specific_heating_rate = 1
    my_chemistry.use_volumetric_heating_rate = 1

    # Set units
    my_chemistry.comoving_coordinates = 0 # proper units
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1.0 / (1.0 + current_redshift) / \
        my_chemistry.a_units
    my_chemistry.density_units = mass_hydrogen_cgs # rho = 1.0 is 1.67e-24 g
    my_chemistry.length_units = cm_per_mpc         # 1 Mpc in cm
    my_chemistry.time_units = sec_per_Myr          # 1 Gyr in s
    my_chemistry.set_velocity_units()
     # density in units of proton mass per cm^3
    density_proper = dens / \
        (my_chemistry.a_units *
            my_chemistry.a_value)**(3*my_chemistry.comoving_coordinates)
    Zsolar = 0.014
    metallicity = Zsolar*metallicity
    def cooling_rate_temp(temperature, density):
        fc = setup_fluid_container(my_chemistry,
                                temperature=np.array(temperature), density=np.array(density),
                                converge=True, metal_mass_fraction=metallicity)
        fc.calculate_cooling_time()
        ## look at sound speed and mean molecular weight
        # find the thermal equillibrium density by finding the index of transition from positive to negative cooling times
        cooling_time = fc["cooling_time"][0]
        return 1/cooling_time
    
    def cooling_rate_dens(density, temperature):
        fc = setup_fluid_container(my_chemistry,
                                temperature=np.array(temperature), density=np.array(density),
                                converge=True, metal_mass_fraction=metallicity)
        fc.calculate_cooling_time()

        # find the thermal equillibrium density by finding the index of transition from positive to negative cooling times
        cooling_time = fc["cooling_time"][0]
        return 1/cooling_time
    if loop == 'rho':
        temp_eq_list = []
        for i in range(len(dens)):
            print(i)
            temp_eq_list.append(optimize.brentq(cooling_rate_temp, 1e0, 1e6, args=(dens[i])))
        return dens, np.array(temp_eq_list)
    elif loop == 'temp':
        rho_eq_list = []
        for i in range(len(temp)):
            print(i)
            stop()
            rho_eq_list.append(optimize.brentq(cooling_rate_dens, 1e-6*mass_hydrogen_cgs, 1e4*mass_hydrogen_cgs, args=(temp[i])))
        return np.array(rho_eq_list), temp
    else:
        print("loop arg could take 'rho' or 'temp' as input")

def calculate_internal_energy(density, temperature): # enter density in units of proton mass per cm^3 and temperature in K
    """
    Calculate the internal energy of the fluid in the units of erg/g.
    """
    # Set solver parameters
    current_redshift = 0.

    # Set solver parameters
    my_chemistry = chemistry_data()
    my_chemistry.use_grackle = 1
    my_chemistry.with_radiative_cooling = 0
    my_chemistry.primordial_chemistry = 0
    my_chemistry.metal_cooling = 1
    my_chemistry.UVbackground = 1
    my_chemistry.self_shielding_method = 0 # check
    my_chemistry.H2_self_shielding = 0
    my_chemistry.photoelectric_heating = 1
    my_dir = os.path.dirname(os.path.abspath(__file__))
    grackle_data_file = bytearray(os.path.join(
        "/scratch/jh2/hs9158/gizmo-fork/gizmo2/grackle/input", "CloudyData_UVB=HM2012_shielded.h5"), 'utf-8')
    my_chemistry.grackle_data_file = grackle_data_file

    my_chemistry.use_specific_heating_rate = 1
    my_chemistry.use_volumetric_heating_rate = 1

    # Set units
    my_chemistry.comoving_coordinates = 0 # proper units
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1.0 / (1.0 + current_redshift) / \
        my_chemistry.a_units
    my_chemistry.density_units = mass_hydrogen_cgs # rho = 1.0 is 1.67e-24 g
    my_chemistry.length_units = cm_per_mpc         # 1 Mpc in cm
    my_chemistry.time_units = sec_per_Myr          # 1 Gyr in s
    my_chemistry.set_velocity_units()
    Zsolar = 0.014
    metallicity = Zsolar*0.5
    # set up fluid container
    fc = setup_fluid_container(my_chemistry,
                                temperature=np.array(temperature), density=np.array(density),
                                converge=True, metal_mass_fraction=metallicity)
    fc.calculate_cooling_time()
    return fc["energy"]*my_chemistry.energy_units, fc["cooling_time"]

def calculate_temperature(density, temperature, internal_energy): # enter density in units of proton mass per cm^3 and temperature in K, not important
    # Set solver parameters
    current_redshift = 0.

    # Set solver parameters
    my_chemistry = chemistry_data()
    my_chemistry.use_grackle = 1
    my_chemistry.with_radiative_cooling = 0
    my_chemistry.primordial_chemistry = 0
    my_chemistry.metal_cooling = 1
    my_chemistry.UVbackground = 1
    my_chemistry.self_shielding_method = 0 # check
    my_chemistry.H2_self_shielding = 0
    my_chemistry.photoelectric_heating = 1
    my_dir = os.path.dirname(os.path.abspath(__file__))
    grackle_data_file = bytearray(os.path.join(
        "/scratch/jh2/hs9158/gizmo-fork/gizmo2/grackle/input", "CloudyData_UVB=HM2012_shielded.h5"), 'utf-8')
    my_chemistry.grackle_data_file = grackle_data_file

    my_chemistry.use_specific_heating_rate = 1
    my_chemistry.use_volumetric_heating_rate = 1

    # Set units
    my_chemistry.comoving_coordinates = 0 # proper units
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1.0 / (1.0 + current_redshift) / \
        my_chemistry.a_units
    my_chemistry.density_units = mass_hydrogen_cgs # rho = 1.0 is 1.67e-24 g
    my_chemistry.length_units = cm_per_mpc         # 1 Mpc in cm
    my_chemistry.time_units = sec_per_Myr          # 1 Gyr in s
    my_chemistry.set_velocity_units()
    Zsolar = 0.014
    metallicity = Zsolar*0.5 # For LMC simulations only
    # metallicity = Zsolar*1.0 # for MW simulations use only
    # set up fluid container
    fc = setup_fluid_container(my_chemistry,
                                temperature=np.array(temperature), density=np.array(density), converge=True,
                                metal_mass_fraction=metallicity)
    fc["energy"] = internal_energy*1e10/my_chemistry.energy_units
    fc.calculate_temperature()
    return fc["temperature"]

# def calculate_abundances(density, temperature, metallicity, internal_energy):
#     # Set solver parameters
#     current_redshift = 0.

#     # Set solver parameters
#     my_chemistry = chemistry_data()
#     my_chemistry.use_grackle = 1
#     my_chemistry.with_radiative_cooling = 0
#     my_chemistry.primordial_chemistry = 0
#     my_chemistry.metal_cooling = 1
#     my_chemistry.UVbackground = 1
#     my_chemistry.self_shielding_method = 0 # check
#     my_chemistry.H2_self_shielding = 0
#     my_chemistry.photoelectric_heating = 1
#     my_dir = os.path.dirname(os.path.abspath(__file__))
#     grackle_data_file = bytearray(os.path.join(
#         "/scratch/jh2/hs9158/gizmo-fork/gizmo2/grackle/input", "CloudyData_UVB=HM2012_shielded.h5"), 'utf-8')
#     my_chemistry.grackle_data_file = grackle_data_file

#     my_chemistry.use_specific_heating_rate = 1
#     my_chemistry.use_volumetric_heating_rate = 1

#     # Set units
#     my_chemistry.comoving_coordinates = 0 # proper units
#     my_chemistry.a_units = 1.0
#     my_chemistry.a_value = 1.0 / (1.0 + current_redshift) / \
#         my_chemistry.a_units
#     my_chemistry.density_units = mass_hydrogen_cgs # rho = 1.0 is 1.67e-24 g
#     my_chemistry.length_units = cm_per_mpc         # 1 Mpc in cm
#     my_chemistry.time_units = sec_per_Myr          # 1 Gyr in s
#     my_chemistry.set_velocity_units()
#     my_chemistry.HydrogenFractionByMass = 0.73

#     # set up fluid container
#     fc = setup_fluid_container(my_chemistry,
#                                 temperature=np.array(temperature), density=np.array(density),
#                                 converge=True, metal_mass_fraction=metallicity)
#     fc["energy"] = internal_energy*1e10/my_chemistry.energy_units
#     stop()
#     fc.calculate_temperature()
#     stop()
#     fc.calculate_hydrogen_number_density()
#     stop()


if __name__ == "__main__":
    # calculate_abundances(density = np.array([1e-3]), temperature=np.array([1e4]), metallicity=np.array([0.5]), internal_energy=np.array([100]))
    xrho, yrho = find_thermal_equillibrium(metallicity=0.5, loop='rho')
    data = np.array([xrho, yrho])
    np.savetxt("/scratch/jh2/hs9158/results/data/thermal_equillibrium.txt", data.T, delimiter=',', header='density, temperature')