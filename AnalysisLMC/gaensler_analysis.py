#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Hilay Shah 2023-

import numpy as np
from cfpack import print, hdfio, stop, constants
from astropy.table import Table
import matplotlib.pyplot as plt
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.stats import norm
from scipy.integrate import quad

def calculate_sky_coordinates(ra, dec, center_ra, center_dec, distance_kpc):
    """
    Convert RA/Dec to physical coordinates where:
    x = East-West direction (RA increases eastward)
    y = North-South direction (Dec increases northward)
    
    Parameters:
    ra, dec: coordinates of point (degrees)
    center_ra, center_dec: reference center (degrees)
    distance_pc: distance to object in parsecs
    """
    # Create SkyCoord objects
    point = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    center = SkyCoord(ra=center_ra*u.degree, dec=center_dec*u.degree, frame='icrs')
    
    # Get separation and position angle
    sep = center.separation(point)
    pa = center.position_angle(point)
    
    # Convert to physical distances
    # Total physical distance
    d_kpc = distance_kpc * sep.radian
    
    # Convert to x,y where:
    # x is positive eastward (RA increases)
    # y is positive northward (Dec increases)
    x = d_kpc * np.sin(pa.radian)  # East-West
    y = d_kpc * np.cos(pa.radian)  # North-South
    
    return x, y

def calculate_likelihood(data, data_errors, mu, sigma):
    """
    Calculate log likelihood for given mu (mean) and sigma (true standard deviation)
    
    Parameters:
    data: array of RM measurements
    data_errors: array of measurement uncertainties (1-sigma)
    mu: mean of true distribution
    sigma: standard deviation of true distribution
    """
    def integrand(x, mu_obs, sigma_obs, mu_true, sigma_true):
        # p_true: true distribution of RM values
        p_true = norm.pdf(x, mu_true, sigma_true)
        # G: Gaussian measurement error
        p_obs = norm.pdf(x, mu_obs, sigma_obs)
        return p_true * p_obs
    
    # Calculate likelihood for each data point
    log_L = 0
    for mu_i, sigma_i in zip(data, data_errors):
        # Integrate over all possible true values
        # L_i, _ = quad(integrand, -np.inf, np.inf, 
        #              args=(mu_i, sigma_i, mu, sigma))
        L_i = 1/np.sqrt(2*np.pi*(sigma_i**2 + sigma**2)) * np.exp(-0.5 * (mu_i - mu)**2 / (sigma_i**2 + sigma**2))
        log_L += np.log(L_i)
    
    return log_L

def maximum_likelihood_estimation(data, data_errors, mu_range, sigma_range):
    # Calculate likelihood for each mu, sigma pair and plot the likelihood surface in 2D
    log_L = np.zeros((len(mu_range), len(sigma_range)))
    for i, mu in enumerate(mu_range):
        print(i)
        for j, sigma in enumerate(sigma_range):
            log_L[i, j] = calculate_likelihood(data, data_errors, mu, sigma)

    # Find the mu and sigma that maximise the likelihood
    max_log_L = np.max(log_L)
    max_log_L_idx = np.unravel_index(np.argmax(log_L, axis=None), log_L.shape)
    best_mu = mu_range[max_log_L_idx[0]]
    best_sigma = sigma_range[max_log_L_idx[1]]



    # plot the likelihood surface
    plt.figure(figsize=(10, 10))
    plt.imshow(log_L, origin='lower', extent=[sigma_range[0], sigma_range[-1], mu_range[0], mu_range[-1]])
    plt.colorbar()
    plt.xlabel("Sigma")
    plt.ylabel("Mu")
    plt.title("Log Likelihood: best mu = {:.2f}, best sigma = {:.2f}".format(best_mu, best_sigma))
    plt.show()



def area_under_gaussian(rm, error):
    # norm.cdf(x, loc=rm, scale=error) calculates the probability of values LESS than x
    # in a Gaussian with mean=rm and standard deviation=error
    
    # For positive area: P(RM > 0) = P(RM < ∞) - P(RM < 0)
    positive_area = norm.cdf(np.inf, loc=rm, scale=error) - norm.cdf(0, loc=rm, scale=error)
    
    # For negative area: P(RM < 0)
    negative_area = norm.cdf(0, loc=rm, scale=error)
    return positive_area, negative_area

def get_plane_rotation(rotation_matrix):
    # define an equation of plane with the normal in the direction of the z-axis
    normal = np.array([0,0,1])
    node = np.array([0,1,0])
    # rotate the normal vector to the new frame
    normal_rot = np.dot(rotation_matrix, normal.T).T
    node_rot = np.dot(rotation_matrix, node.T).T
    def plane_rot(coord0, coord1, coord2): # two points on the plane and one point to compute based on the normal, the point to compute is supposed to be True
        # determine which two argments are float and which one is a flag
        if coord0: coord = -(normal_rot[1]*coord1+normal_rot[2]*coord2)/normal_rot[0]
        if coord1: coord = -(normal_rot[0]*coord0+normal_rot[2]*coord2)/normal_rot[1]
        if coord2: coord = -(normal_rot[0]*coord0+normal_rot[1]*coord1)/normal_rot[2]
        return coord
    return node_rot, plane_rot

def LMC_center(rotation_matrix):
    # LMC center coordinates
    center = SkyCoord(ra = '05h23m35s', dec = '-69d45m22s', frame='icrs')
    ra = center.ra.degree
    dec = center.dec.degree
    node_rot, plane_rot = get_plane_rotation(rotation_matrix=rotation_matrix)
    def node_line(x, center):
        return np.abs(node_rot[1]/node_rot[0])*(x-center[0])+center[1]
    return ra, dec, node_rot, node_line

def node_line_test(x, center, m):
    return m*(x-center[0])+center[1]

def txt2npy(file):
    # read a text file and convert it to a numpy array
    data = np.loadtxt(file)
    # make an astropy table with the data to make it easier to work with
    table = Table(data, names=["ID", "RM", "e_RM", "extra", "RA", "Dec", "x-pixel", "y-pixel", "P", "I", "fracpol"])
    # convert the RA from degrees to hours using astropy
    alpha = np.radians(+34.7-180); beta = np.radians(0); gamma = np.radians(90-(180-139.9))
    a = np.cos(alpha); b = np.cos(beta); c = np.cos(gamma)
    d = np.sin(alpha); e = np.sin(beta); f = np.sin(gamma)    
    R_obs_new = np.array([[b*c, d*e*c-a*f, a*e*c+d*f], [b*f, d*e*f+a*c, a*e*f-d*c], [-e, d*b, a*b]])
    ra_center, dec_center, node_rot, node_line = LMC_center(R_obs_new)
    coords = SkyCoord(ra=table["RA"]*u.degree, dec=table["Dec"]*u.degree, frame='icrs')
    x, y = calculate_sky_coordinates(table["RA"], table["Dec"], ra_center, dec_center, 50)
    table["x_kpc"] = x; table["y_kpc"] = y
    # # add a column to the table with the RA in hours
    # table["RA_hours"] = coords.ra.hour

    sign = np.sign(node_line(table["x_kpc"], [0, 0])-table["y_kpc"])
    table["sign"] = sign
    # table["positive_RM_frac"], table["negative_RM_frac"] = area_under_gaussian(table["RM"], table["e_RM"])
    return table

def txt2npy_test(file, center, m):
    # read a text file and convert it to a numpy array
    data = np.loadtxt(file)
    # make an astropy table with the data to make it easier to work with
    table = Table(data, names=["ID", "RM", "e_RM", "extra", "RA", "Dec", "x-pixel", "y-pixel", "P", "I", "fracpol"])
    # convert the RA from degrees to hours using astropy
    # ra_center, dec_center, node_rot, node_line = LMC_center(R_obs_new)
    # coords = SkyCoord(ra=table["RA"]*u.degree, dec=table["Dec"]*u.degree, frame='icrs')
    # x, y = calculate_sky_coordinates(table["RA"], table["Dec"], ra_center, dec_center, 50)
    # table["x_kpc"] = x; table["y_kpc"] = y
    # # add a column to the table with the RA in hours
    # table["RA_hours"] = coords.ra.hour

    # sign = np.sign(node_line_test(table["x_kpc"], [center[0], center[1]], m)-table["y_kpc"])
    # table["sign"] = sign
    # table["positive_RM_frac"], table["negative_RM_frac"] = area_under_gaussian(table["RM"], table["e_RM"])
    return table

def scatter_plot(file):
    # read the data from the text file
    table = txt2npy(file)
    ra_center, dec_center, node_rot, node_line = LMC_center(R_obs_new)
    sign = table["sign"]
    # make a scatter plot of the RMs
    plt.figure(figsize=(5,5))
    color = np.repeat('b', len(table["RM"]))
    color[table["RM"]<0] = 'r'

    edgecolor = np.repeat('b', len(table["RM"]))
    edgecolor[sign<0] = 'r'

    plt.scatter(table["x_kpc"], table["y_kpc"], c=color, s=np.abs(table["RM"]), edgecolor=edgecolor)
    plt.scatter(0, 0, c='k', marker='x', s=15)
    # make a line passing through the center of the LMC pointing towards node_rot vector
    # stop()
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    x_arr = np.linspace(-5, 5, 100)
    y_arr = node_line(x_arr, [0, 0])
    plt.plot(x_arr, y_arr, c='k', linestyle='--')
    plt.xlabel("x (kpc)")
    plt.ylabel("y (kpc)")
    # ra_arr = np.linspace(table['RA'].min(), table['RA'].max(), 100)
    # dec_arr = node_line(ra_arr, [ra_center, dec_center])
    # deg10 = np.rad2deg(10/50)
    # plt.plot(ra_arr, dec_arr, c='k', linestyle='--')

    # invert the x-axis so that it goes from right to left
    
    # plt.xlim([table['RA'].min(), table['RA'].max()])
    # plt.ylim([table['Dec'].min(), table['Dec'].max()])
    plt.gca().invert_xaxis()
    # plt.xlabel("RA (degrees)")
    # plt.ylabel("Dec (degrees)")
    plt.show()
    stop()

def positive_negative_fraction(file, center, m, key='test'):
    # read the data from the text file
    if key == 'test':
        table = txt2npy_test(file, [center[0], center[1]], m)
    else:
        table = txt2npy(file)
    # calculate the fraction of positive and negative RMs on each side of the node_line
    pos_ind = np.where(table["sign"]>0)[0]
    neg_ind = np.where(table["sign"]<0)[0]
    # Assume a gaussian around table['RM'] with a standard deviation of table['e_RM'] and calculate the probability of the RM being positive and negative
    positiveRM_positiveside = table['positive_RM_frac'][pos_ind].mean()
    negativeRM_positiveside = table['negative_RM_frac'][pos_ind].mean()
    positiveRM_negativeside = table['positive_RM_frac'][neg_ind].mean()
    negativeRM_negativeside = table['negative_RM_frac'][neg_ind].mean()
    positiveRM_overall = table['positive_RM_frac'].mean()
    negativeRM_overall = table['negative_RM_frac'].mean()
    # print(f"Positive RM fraction on the positive side of the LMC: {positiveRM_positiveside}")
    # print(f"Negative RM fraction on the positive side of the LMC: {negativeRM_positiveside}")
    # print(f"Positive RM fraction on the negative side of the LMC: {positiveRM_negativeside}")
    # print(f"Negative RM fraction on the negative side of the LMC: {negativeRM_negativeside}")
    # print(f"Overall positive RM fraction: {positiveRM_overall}")
    # print(f"Overall negative RM fraction: {negativeRM_overall}")
    deltapositiveRM_positiveside = len(np.where(table['RM'][pos_ind]>0)[0])/len(pos_ind)
    deltanegativeRM_positiveside = len(np.where(table['RM'][pos_ind]<0)[0])/len(pos_ind)
    deltapositiveRM_negativeside = len(np.where(table['RM'][neg_ind]>0)[0])/len(neg_ind)
    deltanegativeRM_negativeside = len(np.where(table['RM'][neg_ind]<0)[0])/len(neg_ind)
    # print(f"Delta Positive RM fraction on the positive side of the LMC: {deltapositiveRM_positiveside}")
    # print(f"Delta Negative RM fraction on the positive side of the LMC: {deltanegativeRM_positiveside}")
    # print(f"Delta Positive RM fraction on the negative side of the LMC: {deltapositiveRM_negativeside}")
    # print(f"Delta Negative RM fraction on the negative side of the LMC: {deltanegativeRM_negativeside}")
    # print("mean of RM on positive side: ", np.mean(table['RM'][pos_ind]))
    # print("mean of RM on negative side: ", np.mean(table['RM'][neg_ind]))
    return positiveRM_positiveside, negativeRM_negativeside, positiveRM_overall

def maximise_dynamo(centerx_range, centery_range, m_range):
    posposRM = np.zeros((len(centerx_range), len(centery_range), len(m_range)))
    negnegRM = np.zeros((len(centerx_range), len(centery_range), len(m_range)))
    posRM = np.zeros((len(centerx_range), len(centery_range), len(m_range)))
    for ix in range(len(centerx_range)):
        print(ix)
        for iy in range(len(centery_range)):
            center = [centerx_range[ix], centery_range[iy]]
            for im in range(len(m_range)):
                positiveRM_positiveside, negativeRM_negativeside, positiveRM_overall = positive_negative_fraction(filepath+filename, center, m_range[im], key='test')
                posposRM[ix, iy, im] = positiveRM_positiveside
                negnegRM[ix, iy, im] = negativeRM_negativeside
                posRM[ix, iy, im] = positiveRM_overall
    return posposRM, negnegRM, posRM


if __name__ == "__main__":
    filepath = "/scratch/jh2/hs9158/results/Gaensler2005Data/"
    filename = "rm_bkg_list.txt"
    alpha = np.radians(+34.7-180); beta = np.radians(0); gamma = np.radians(90-(180-139.9))
    a = np.cos(alpha); b = np.cos(beta); c = np.cos(gamma)
    d = np.sin(alpha); e = np.sin(beta); f = np.sin(gamma)    
    R_obs_new = np.array([[b*c, d*e*c-a*f, a*e*c+d*f], [b*f, d*e*f+a*c, a*e*f-d*c], [-e, d*b, a*b]])
    table = txt2npy(filepath+filename)
    # Save the table to a file
    # table.write('/scratch/jh2/hs9158/results/data/gaensler_table.dat', format='ascii', overwrite=True)
    
    stop()
    LMC_center(rotation_matrix=R_obs_new)
    # ra_center, dec_center = LMC_center()
    # scatter_plot(filepath+filename)
    positive_negative_fraction(filepath+filename, [0,0], 0.84, key='test')
    m_LMC = 0.8421
    # posposRM, negnegRM, posRM = maximise_dynamo(centerx_range=np.linspace(-0.5, 0.5, 11), centery_range=np.linspace(-0.5, 0.5, 11), m_range=np.array([m_LMC]))
    maximum_likelihood_estimation(data=table["RM"], data_errors=table["e_RM"], mu_range=np.linspace(-20, 20, 200), sigma_range=np.linspace(30, 70, 200))
    stop()
    # plot the posposRM with centerx_range on x-axis, centery_range on y-axis and max of m_range as colorbar
    plt.figure(figsize=(10,10)); plt.imshow(posposRM.max(axis=2), origin='lower', extent=[-0.5, 0.5, -0.5, 0.5]); plt.colorbar(); plt.xlabel("Center x"); plt.ylabel("Center y"); plt.title("Max +RM frac on + side"); plt.show()
    plt.figure(figsize=(10,10)); plt.imshow(negnegRM.max(axis=2), origin='lower', extent=[-0.5, 0.5, -0.5, 0.5]); plt.colorbar(); plt.xlabel("Center x"); plt.ylabel("Center y"); plt.title("Max -RM frac on - side"); plt.show()

    plt.figure(figsize=(10,10)); plt.imshow(posposRM, origin='lower', extent=[-0.5, 0.5, -0.5, 0.5]); plt.colorbar(); plt.xlabel("Center x"); plt.ylabel("Center y"); plt.title("+RM frac on + side"); plt.show()
    plt.figure(figsize=(10,10)); plt.imshow(negnegRM, origin='lower', extent=[-0.5, 0.5, -0.5, 0.5]); plt.colorbar(); plt.xlabel("Center x"); plt.ylabel("Center y"); plt.title("-RM frac on - side"); plt.show()

    stop()