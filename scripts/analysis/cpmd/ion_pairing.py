#=========================================
# Script to compute layer rdfs and fraction of ions paired in each layer
#     Maddy Murphy
#     May 2025
#=========================================

# Import libraries
import numpy as np
import MDAnalysis as mda # type: ignore
import pandas as pd # type: ignore
import os
import re
from scipy import interpolate # type: ignore
from scipy.spatial.distance import cdist # type: ignore
from scipy import stats # type: ignore
import glob
from scipy import integrate
from scipy.signal import savgol_filter

def run_rdf(path, ion_pair_distances, no3_dists_from_electrode, area, cutoffs, dr=0.08, m_vdw=2.95, rerun=False):
    """
    Compute the radial distribution function (rdf) and potential of mean force (pmf)
        between ions in the system.

    Parameters:
        path (str): Path to the directory where output files will be saved.
        ion_pair_distances (numpy.ndarray): Array of ion pair distances.
        no3_dists_from_electrode (numpy.ndarray): Array of NO3 distances from the electrode.
        area (float): Area of the electrode.
        cutoffs (float): used to define the z-dimension of the electrolyte (cell_width).
        dr (float, optional): Bin width for PMF calculation. Default is 0.08 Angstrom.

    Returns:
        tuple: A tuple containing the rdf, pmf, and corresponding bin edges.
    """

    def get_volume_normalization(ion_pair_distances, no3_dists_from_electrode, cutoffs, m_vdw=2.95):
        """
        Calculate volume normalization factors for potential of mean force (PMF) calculation (A_shell).

        Parameters:
            ion_pair_distances (numpy.ndarray): Array of ion pair distances.
            no3_dists_from_electrode (numpy.ndarray): Array of NO3 distances from the electrode.
            m_vdw (float): van der waals radius of the metal. Default is 1.6 Angstroms.

        Returns:
            numpy.ndarray: Volume normalization factors.
        """       
        r = ion_pair_distances.flatten()

        no3_dist_cathode = np.subtract(no3_dists_from_electrode, m_vdw) # subtract vdw radii
        h_top = no3_dist_cathode.flatten() # distance between nitrate and cathode

        cell_width = cutoffs[0] - cutoffs[4] - 2 * m_vdw 
        h_bot = cell_width - h_top # distance between nitrate and anode

        cos_theta_top = np.where(h_top < r, h_top/r, 1) # if r > h_top compute the cos(theta) between r and h_top
        cos_theta_bot = np.where(h_bot < r, h_bot/r, 1) # if r > h_bot compute the cos(theta) between r and h_bot

        hist_vol_na = (
            4 * np.pi * r**2
            - 2 * np.pi * r**2 * (1 - cos_theta_top)
            - 2 * np.pi * r**2 * (1 - cos_theta_bot)
        )

        return hist_vol_na
    
    filename = path + 'rdf.npy'
    if rerun==True or not os.path.exists(filename):
        # initialize bins for the histogram
        bins = np.arange(1, np.sqrt(area), dr)

        # define volume normalization weights based on the cut circle due to the electrode
        weights = get_volume_normalization(ion_pair_distances, no3_dists_from_electrode, cutoffs)

        dens, bins = np.histogram(
            ion_pair_distances.flatten(), weights=1 / weights, bins=bins, density=False
        )

        cell_width = cutoffs[0] - cutoffs[4] - 2 * m_vdw
        bulk_dens = len(ion_pair_distances.flatten()) / (area * cell_width)
        rdf = dens / dr / bulk_dens
        pmf = -np.log(rdf)

        # take the midpoint of the bins
        bins = bins[:-1] + (bins[1] - bins[0]) / 2

        np.save(filename, rdf)
        np.save(path + 'pmf.npy', pmf)
        np.save(path + 'bins.npy', bins)

    return None

def fraction_paired(ion_pair_dists, no3_dists, cip, ssip, cutoffs):
    """ 
    Compute the fraction of nitrate ions paired in each layer.
    
    Parameters:
        ion_pair_dists (np.array): all the ion pair distances from the simulation (timesteps, nitrate, cation)
        no3_dists (np.array): distances from the cathode to the nitrate ion
        cip (float): cutoff for cip pairing
        ssip (float): cutoff for ssip pairing
        cutoffs (np.array): cutoff distances from the cathode for different layers

    returns:
        no3_pairing (np.array): fraction paired data averaged across each timestep
        n_nitrate (np.array): number of nitrate in the interface averaged across each timestep
    """

    no3_pairing = np.zeros((len(ion_pair_dists[:,0,0]), 3, 3)) # timesteps, (stern, diffuse, bulk), (cip, ssip, free)
    n_nitrate = np.zeros((len(ion_pair_dists[:,0,0]), 3)) # timesteps, (stern, diffuse, bulk)

    for x in range(len(ion_pair_dists[:,0,0])): #timesteps
        # initialize values
        n_stern = np.zeros(4) # [total number, cip, ssip, free]
        n_diffuse = np.zeros(4)
        n_bulk = np.zeros(4)

        for y in range(len(ion_pair_dists[0,:,0])): # nitrate ions
            # check nitrate ion layer -- cutoff distance = cutoff[0] - cutoff[layer]
            if no3_dists[x,y] <= (cutoffs[0] - cutoffs[1]): # stern layer
                n_stern[0] += 1
                if any(dist <= cip for dist in ion_pair_dists[x,y,:]):
                    n_stern[1] += 1
                elif any(dist <= ssip for dist in ion_pair_dists[x,y,:]):
                    n_stern[2] += 1
                else:
                    n_stern[3] += 1
            

            elif no3_dists[x,y] <= (cutoffs[0] - cutoffs[2]): # diffuse layer
                n_diffuse[0] += 1
                if any(dist <= cip for dist in ion_pair_dists[x,y,:]):
                    n_diffuse[1] += 1
                elif any(dist <= ssip for dist in ion_pair_dists[x,y,:]):
                    n_diffuse[2] += 1
                else:
                    n_diffuse[3] += 1

            elif no3_dists[x,y] <= (cutoffs[0] - cutoffs[3]): # bulk layer
                n_bulk[0] += 1
                if any(dist <= cip for dist in ion_pair_dists[x,y,:]):
                    n_bulk[1] += 1
                elif any(dist <= ssip for dist in ion_pair_dists[x,y,:]):
                    n_bulk[2] += 1
                else: 
                    n_bulk[3] += 1

        # compute fraction paired
        # stern layer
        n_nitrate[x,0] = n_stern[0]

        # diffuse layer
        n_nitrate[x,1] = n_diffuse[0]

        # bulk layer
        n_nitrate[x,2] = n_bulk[0]

        for i in range(3):
            no3_pairing[x,0,i] = n_stern[i+1]/n_stern[0]
            no3_pairing[x,1,i] = n_diffuse[i+1]/n_diffuse[0]
            no3_pairing[x,2,i] = n_bulk[i+1]/n_bulk[0]

    # average across timesteps
    no3_pairs = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            no3_pairs[i,j] = np.nanmean([no3_pairing[x,i,j] for x in range(len(ion_pair_dists[:,0,0]))])
                                     
    n_nitrate_std = np.std(n_nitrate, axis=0)
    n_nitrate = np.mean(n_nitrate, axis=0)

    print(f'pairing data: {no3_pairs}')
    print(f'number of nitrates: {n_nitrate}, std: {n_nitrate_std}')
    return no3_pairs, n_nitrate

def ion_cuts(pmf, bins):
    pmf = pmf[38:] # cut the infinite portion
    bins_trunc = bins[38:]
    max_1 = np.where(pmf == np.max(pmf[0:25]))[0][0] # 4 to 6.08
    max_2 = np.where(pmf == np.max(pmf[25:40]))[0][0] # 6.08 to 8.08

    cip = bins_trunc[max_1]
    ssip = bins_trunc[max_2]
    return cip, ssip 
    

def main():
    # 1. initialize data path, cations, potentials, etc

    # 2. iterate through each potential, cation, and rep

    # 3. load cutoffs and raw data

    # 4. run the fraction paired analysis

    cations = ['Cs', 'K', 'Na', 'Li']
    potentials = ['00', '10', '20']
    reps = ['0', '1', '2', '3', '4']
    layers = ['stern', 'diffuse', 'bulk']

    base_dir = '/home/mamurphy/nitrate-reduction/cpmd-simulations/'
    base_folder = f'{base_dir}data/'

    rerun = True

    for i, cat in enumerate(cations):
        for j, pot in enumerate(potentials):
            for k, rep in enumerate(reps):
                if cat=='Cs' and pot=='00' and rep=='2':
                    continue # missing data
                else:
                    path = f'{base_folder}{cat}/wca/{pot}/rep{rep}/'
                    area = np.load(f'{path}area.npy', allow_pickle=True)
                    ion_pair_dists = np.load(f'{path}ion_pair_distances.npy', allow_pickle=True)
                    no3_dists = np.load(f'{path}no3_dists_from_electrode.npy', allow_pickle=True)

                    # iterate through each layer and run rdf analysis
                    print(f'running {cat} potential: {pot} rep{rep}.')
                    cutoffs = np.load(path + 'cutoffs.npy', allow_pickle=True)
                    print(cutoffs)

                    for layer in layers:
                        layer_folder = f'{path}layer-analysis/{layer}/'
                        no3_dists = np.load(layer_folder + 'no3_dists.npy', allow_pickle=True)
                        ion_pair_dists = np.load(layer_folder + 'ion_pair_dists.npy', allow_pickle=True)
                        
                        run_rdf(layer_folder, ion_pair_dists, no3_dists, area, cutoffs, rerun=rerun)

                    layer_folder = f'{path}layer-analysis/'
                    pmf = np.load(f'{layer_folder}bulk/pmf.npy', allow_pickle=True)
                    bins = np.load(f'{layer_folder}bulk/bins.npy', allow_pickle=True)
                    cip, ssip = ion_cuts(pmf, bins)

                    no3_pairs, n_nitrate = fraction_paired(ion_pair_dists, no3_dists, cip, ssip, cutoffs)
                    np.save(f'{layer_folder}no3_pairing.npy', no3_pairs)
                    np.save(f'{layer_folder}no3_counts.npy', n_nitrate)




if __name__ == "__main__":
    main()








                            
        





                










