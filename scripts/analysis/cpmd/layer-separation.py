#=========================================
# Script to identify layers based on EDL definitions
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
from scipy.signal import savgol_filter, find_peaks # type: ignore

def layer_separation(path, density, edges, anode_charges, cathode_charges, n_atoms, area, rerun=False):

    def electrode_charge_density(density, anode_charges, cathode_charges, n_atoms, area):
        """ 
        Compute the charge denisty of each layer of the electrode.
        
        Parameters:
            density (dictionary): dictionary of the density profile of each atom type
            anode_charges (numpy array): anode charges at each timestep
            cathode_charges (numpy array): cathode charges at each timestep
            n_atoms (dictionary): dictionary storing the total number of each type of atom
            area (float): cross sectional area of the electrode
            
        Returns:
            charge_density (numpy array): charge density of only the electrode atoms
        """
        
        # average the charge on each electrode atom across the trajectory # charges = (x, 1690, 2) = (timestep, atom, (atom_id, charge))
        # omit the first 1000 steps (equilibration time)
        anode_charges = np.mean(anode_charges[1000:], axis=0)
        cathode_charges = np.mean(cathode_charges[1000:], axis=0)

        # each metal has 10 layers -- 169 atoms per layer
        anode = np.zeros((10,169))
        cathode = np.zeros((10,169))
        for i in range(10):
            start = i*169
            end = (i+1)*169
            anode[i] = anode_charges[start:end][:,1]
            cathode[i] = cathode_charges[start:end][:,1]

        # average over all atoms in the layer to get charge per atom
        anode = np.mean(anode, axis=1)
        cathode = np.mean(cathode, axis=1)

        anode_idx = np.where(density['Au'] > 0.1)[0][0:10]
        cathode_idx = np.where(density['Au'] > 0.1)[0][10:]
        anode_charge_density = np.zeros((10))
        cathode_charge_density = np.zeros((10))

        charge_density = np.copy(density['Au'])

        # get the charge density for each layer
        for i in range(10):
            layer_charge = anode[i] # per atom
            layer_density = density['Au'][anode_idx[i]] 
            ion_density = layer_density * n_atoms  / (area * 1e-24)  # number density in atoms/cm^3
            anode_charge_density[i] = ion_density * layer_charge * (1.602e-19)  # charge density in C/cm^3

            layer_charge = cathode[i]
            layer_density = density['Au'][cathode_idx[i]]
            ion_density = layer_density * n_atoms  / (area * 1e-24)
            cathode_charge_density[i] = ion_density * layer_charge * (1.602e-19)

            charge_density[anode_idx[i]] = anode_charge_density[i]
            charge_density[cathode_idx[i]] = cathode_charge_density[i]
        return charge_density
    
    def charge_density_profile(path, density, anode_charges, cathode_charges, n_atoms, area, rerun=False):
        """ 
        Compute the charge denisty of each layer of the electrode.
        
        Parameters:
            path (str): path to folder to save charge_density array
            density (dictionary): dictionary of the density profile of each atom type
            anode_charges (numpy array): anode charges at each timestep
            cathode_charges (numpy array): cathode charges at each timestep
            n_atoms (dictionary): dictionary storing the total number of each type of atom
            area (float): cross sectional area of the electrode
            
        Returns:
            charge_density (numpy array): charge density of total simulation
        """
        ion_names = ['O', 'H', 'C', 'N_NO', 'O_NO']
        partial_charges = {
            'O': -0.8476,
            'H': 0.4238,
            'C': 1.0,
            'N_NO': 0.8603,
            'O_NO': -0.6201
        } 
        filename = f'{path}charge-density.npy'

        if rerun==True or os.path.exists(filename):
            charge_density = np.zeros_like(density['O'])
            
            for ion in ion_names:
                # convert probability density to number density
                ion_density = density[ion] * n_atoms[ion]  / (area * 1e-24)  # number density in atoms/cm^3
                ion_charge_density = ion_density * partial_charges[ion] * (1.602e-19) # charge density in C/cm^3

                charge_density += ion_charge_density

            # add electrode_charge_density
            electrode = electrode_charge_density(density, anode_charges, cathode_charges, n_atoms['Au'], area)
            charge_density += electrode

            # save array
            charge_density_array = np.array(charge_density)
            np.save(filename, charge_density_array)

        else:
            charge_density_array = np.load(filename, allow_pickle=True)

        return charge_density_array
    
    def electric_field(path, charge_density, edges, rerun=False):
        e0_m = 8.854e-12 # F/m
        e0 = e0_m * 1e-2 # F/cm
        m2cm = 1e2 # m to cm
        ang2cm = 1e-8 # angstrom to cm

        filename = f'{path}electric-field.npy'

        if rerun==True or os.path.exists(filename):
            # e_field = (-ang2cm/e0) * integrate.cumulative_trapezoid(charge_density, edges, initial=0)
            reversed_int = integrate.cumulative_trapezoid(charge_density[::-1], edges[::-1], initial=0)
            e_field = (-ang2cm/e0) * reversed_int[::-1]
            np.save(filename, e_field)

        else:
            e_field = np.load(filename, allow_pickle=True)

        return e_field
    
    def electrostatic_potential(path, e_field, edges, rerun=False):
        ang2cm = 1e-8 # angstrom to cm

        filename = f'{path}electrostatic-potential.npy'

        if rerun==True or os.path.exists(filename):
            # e_potential = integrate.cumulative_trapezoid(e_field, edges, initial=0) * ang2cm
            reversed_int = integrate.cumulative_trapezoid(e_field[::-1], edges[::-1], initial=0)
            e_potential = reversed_int[::-1] * ang2cm

            np.save(filename, e_potential)
        else:
            e_potential = np.load(filename, allow_pickle=True)

        return e_potential

    def cutoffs(path, edges, density, e_potential, rerun=False):
        filename = f'{path}cutoffs.npy'

        if rerun==True or os.path.exists(filename):
            # define z-coordinate cutoffs
            cutoffs = np.zeros(5) # cathode, stern layer, diffuse layer, bulk layer, anode

            # cathode and anode
            anode_idx = np.where(density['Au'] > 0.1)[0][9]
            cathode_idx = np.where(density['Au'] > 0.1)[0][10]
            anode = edges[anode_idx]
            cathode = edges[cathode_idx]

            middle = int(len(e_potential)/2)
            edge = cathode_idx - 12

            # smooth
            smoothed_potential = savgol_filter(e_potential[middle:edge], window_length=5, polyorder=2)
            edges_cut = edges[middle:edge]

            # detect peaks
            peaks, properties = find_peaks(smoothed_potential, prominence=1e-3)  # tune prominence

            if len(peaks) < 2:
                print("Not enough peaks found")
                stern = np.nan
            else:
                # second to last peak
                peak_index = peaks[-1]
                stern = edges_cut[peak_index]
                print(f'stern index: {peak_index}, stern cutoff: {stern}')

            #  find where the potential plateaus
            # compute derivative
            derivative = np.gradient(smoothed_potential[:peak_index])
            edges_cut = edges_cut[:peak_index]

            # find where the derivative is close to zero for three consecutive points
            indices = np.where(np.abs(derivative) < 5e-4)[0]

            # find a region with three consecutive points
            if len(indices) < 3:
                diffuse = edges_cut[indices[-1]]
            else:
                for k in range(len(indices)-2):
                    reversed_indices = indices[::-1]
                    if reversed_indices[k] - reversed_indices[k+2] == 2:
        
                        diffuse = edges_cut[reversed_indices[k]]
                        break
                else:
                    # if no consecutive points are found, use the last index
                    print('no consecutive points found')
                    print(f'indices: {indices}')


            bulk = diffuse - 20

            cutoffs = [cathode, stern, diffuse, bulk, anode]
            print(f'cutoffs: {cutoffs}')

            np.save(filename, cutoffs)
        else:
            cutoffs = np.load(filename, allow_pickle=True)

        return cutoffs
    
    ## run the cutoff analysis
    # charge density
    charge_density = charge_density_profile(path, density, anode_charges, cathode_charges, n_atoms, area, rerun=rerun)

    # electric field
    e_field = electric_field(path, charge_density, edges, rerun=rerun)

    # electrostatic potential
    e_potential = electrostatic_potential(path, e_field, edges, rerun=rerun)

    # cutoffs
    cutoff_array = cutoffs(path, edges, density, e_potential, rerun=rerun)
    print(f'cutoff array: {cutoff_array}')

    return cutoff_array

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

        print(f'rdf and pmf calculation complete. file: {filename}')
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

    # 3. define the cutoffs

    # 4. snip the ion pair distances based on the cutoffs
    #     do this in one for loop to save computational time

    cations = ['Cs', 'K', 'Na', 'Li']
    potentials = ['00', '10', '20']
    reps = ['5', '6', '7']
    # rep = 0
    # params = ['lj_a', 'lj_b', 'lj_c']  # different parameter sets for each cation

    layers = ['stern', 'diffuse', 'bulk']
    base_dir = '/anvil/scratch/x-mmurphy4/'
    base_folder = f'{base_dir}data/'

    rerun = True

    for i, cat in enumerate(cations):
        for j, pot in enumerate(potentials):
            for k, rep in enumerate(reps):

                path = f'{base_folder}{cat}/wca/{pot}/rep{rep}/'
                # load data
                density = np.load(f'{path}density.npy', allow_pickle=True).item()
                edges = np.load(f'{path}edges.npy', allow_pickle=True)
                n_atoms = np.load(f'{path}n-atoms.npy', allow_pickle=True).item()
                area = np.load(f'{path}area.npy', allow_pickle=True)

                anode_charges = np.load(f'{path}anode_charges.npy', allow_pickle=True)
                cathode_charges = np.load(f'{path}cathode_charges.npy', allow_pickle=True)

                ion_pair_distances = np.load(f'{path}ion_pair_distances.npy', allow_pickle=True)
                no3_distances_from_electrode = np.load(f'{path}no3_dists_from_electrode.npy', allow_pickle=True)
                # ion_pair_orientations = np.load(f'{path}ion_pair_orientations.npy', allow_pickle=True) # not used in this analysis

                # define the cutoffs
                print(f'running layer separation for {cat} potential: {pot} rep{rep}.')
                cutoffs = layer_separation(path, density, edges, anode_charges, cathode_charges, n_atoms, area, rerun=rerun)

                # define new path to save layer analysis
                folder = f'{path}layer-analysis/'
                os.makedirs(os.path.dirname(folder), exist_ok=True)

                ## Maybe move this to its own function ??
                file = f'{path}layer-analysis/stern/no3_pairing.npy'
                if rerun == True or not os.path.exists(file):
                    print(f'running analysis for {cat} potential: {pot} rep{rep}')
                    # filter out ion pairs that are in the interface
                    no3_dists_stern = []
                    ion_pair_dists_stern = []

                    no3_dists_diffuse = []
                    ion_pair_dists_diffuse = []

                    no3_dists_bulk = []
                    ion_pair_dists_bulk = []

                    for x in range(len(ion_pair_distances[:,0,0])): #timesteps
                        for y in range(len(ion_pair_distances[0,:,0])): # nitrate ions

                            # check nitrate ion layer -- cutoff distance = cutoff[0] - cutoff[layer]
                            if no3_distances_from_electrode[x,y] <= (cutoffs[0] - cutoffs[1]): # stern layer
                                for i in range(len(ion_pair_distances[x,y,:])):
                                    no3_dists_stern.append(no3_distances_from_electrode[x,y])
                                    ion_pair_dists_stern.append(ion_pair_distances[x,y,i])

                            elif no3_distances_from_electrode[x,y] <= (cutoffs[0] - cutoffs[2]): # diffuse layer
                                for i in range(len(ion_pair_distances[x,y,:])):
                                    no3_dists_diffuse.append(no3_distances_from_electrode[x,y])
                                    ion_pair_dists_diffuse.append(ion_pair_distances[x,y,i])

                            elif no3_distances_from_electrode[x,y] <= (cutoffs[0] - cutoffs[3]): # bulk layer
                                for i in range(len(ion_pair_distances[x,y,:])):
                                    no3_dists_bulk.append(no3_distances_from_electrode[x,y])
                                    ion_pair_dists_bulk.append(ion_pair_distances[x,y,i])
                    no3_dists = [np.array(no3_dists_stern), np.array(no3_dists_diffuse), np.array(no3_dists_bulk)]
                    ion_pair_dists = [np.array(ion_pair_dists_stern), np.array(ion_pair_dists_diffuse), np.array(ion_pair_dists_bulk)]
                    print(f'layer separation complete.')

                    for l, layer in enumerate(layers):
                        layer_folder = f'{path}layer-analysis/{layer}/'
                        os.makedirs(os.path.dirname(layer_folder), exist_ok=True)
                        run_rdf(layer_folder, ion_pair_dists[l], no3_dists[l], area, cutoffs, rerun=rerun)

                    layer_folder = f'{path}layer-analysis/'
                    pmf = np.load(f'{layer_folder}bulk/pmf.npy', allow_pickle=True)
                    bins = np.load(f'{layer_folder}bulk/bins.npy', allow_pickle=True)
                    cip, ssip = ion_cuts(pmf, bins)

                    no3_pairs, n_nitrate = fraction_paired(ion_pair_distances, no3_distances_from_electrode, cip, ssip, cutoffs)
                    np.save(f'{layer_folder}no3_pairing.npy', no3_pairs)
                    np.save(f'{layer_folder}no3_counts.npy', n_nitrate)
                    
                else:
                    # load layers
                    print(f'calculation already done for {cat} potential: {pot} rep{rep}')
            

if __name__ == "__main__":
    main()
