""" 
script with functions used to analyze the RDFs between two of the same atom type. 
Madeline Murphy -- Jan 2024
"""

# Import packages
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import MDAnalysis as mda # type: ignore
from MDAnalysis.analysis.rdf import InterRDF # type: ignore
from scipy.signal import argrelextrema # type: ignore
import pandas as pd # type: ignore
import glob
import re
import os

def create_mda(path, data_file, dcd_file, cat_dcd=False):
    if cat_dcd:  
        all_dcd_files = glob.glob(path + dcd_file)
        step = []
        for f in all_dcd_files:
            step.append(int(f.replace('.','_').split('_')[-2]))
        dcd_file = [x for _,x in sorted(zip(step,all_dcd_files))]
        # print(dcd_file)
    else:
        dcd_file = path + dcd_file
    run = mda.Universe(path + data_file,dcd_file, format="LAMMPS", atom_style='id resid type x y z')
    
    atom_names = {'1': 'O',
              '2': 'H',
              '3': 'C',
              '4': 'N_NO',
              '5': 'O_NO'}

    names = []
    for atom in run.atoms:
        names.append(atom_names[atom.type])
    run.add_TopologyAttr('name', names)
    
    return run

def get_positions(u, atoms):
    """ 
    Get the positons of atoms in an MD trajectory
    
    parameters:
        u: MDAnalysis universe object
        atoms: MDAnalysis atom group object
        
    returns:
        atom_positions: a numpy array of shape (n_frames, n_atoms, 3) containing the atom positions
    """
    time = 0
    atom_positions = np.zeros((u.trajectory.n_frames, len(atoms), 3))
    for ts in u.trajectory:
        atom_positions[time, :, :] = atoms.positions
        time += 1
    return atom_positions

def compute_rdf_one_atom(atom_positions, box, r_max, n_bins):
    """ 
    Compute the radial distribution function for the system. 
    
    parameters: 
        atom_positions: a numpy array of the atom positions
        box: a numpy array of shape (3, 3) containing the box vectors
        r_max: the maximum distance to calculate the RDF
        n_bins: the number of bins to use for the RDF

    returns:
        rdf: a numpy array containing the RDF
        bins: a numpy array containing the bin edges
    """
    n_frames = atom_positions.shape[0]
    n_atoms = atom_positions.shape[1]

    dr = r_max / n_bins
    rdf = np.zeros((n_frames, n_bins))
    bins = np.arange(0, r_max + dr, dr)

    for frame in range(n_frames):
        for ref_atom in range(n_atoms):
            for atom in range(n_atoms):
                if ref_atom == atom: # skip if the ref atom is the same as the surrounding atom
                    continue
                delta = atom_positions[frame, ref_atom] - atom_positions[frame, atom] # calculate the distance between the atoms (vector)
                delta = delta - box * np.round(delta / box) # minimum image convention
                r = np.linalg.norm(delta) # determine the distance between the atoms (norm of the vector)
                hist_counts, _ = np.histogram(r, bins) # determine the bin in which the distance falls
                rdf[frame] += hist_counts[:n_bins] # add the counts to the rdf

    # normalize the rdf
    V = box[0] * box[1] * box[2]
    density = n_atoms / V

    for i in range(n_bins):
        r_i = bins[i]
        shell_volume = 4 * np.pi * (r_i**2) * dr
        normalization = density * n_atoms * shell_volume
        rdf[:, i] /= normalization

    rdf = rdf.mean(axis=0)
    if len(bins) > n_bins:
        bins = bins[:n_bins]
    return rdf, bins    
    
def run_rdf_one_atom(path, cations, concentrations, nbins, atom_name, fileid, rerun=False):
    """ 
    Compute the radial distribution function for the system.
    parameters:
        path (str): the path to the directory containing the trajectory files
        cations (list): a list of cation names
        concentrations (list): a list of concentrations
        nbins (int): the number of bins to use for the RDF
        atom_name (str): the name of the atom
        fileid (str): the name of the fileid to save the RDF .npy files
        rerun (bool): whether to rerun the RDF calculation
        
    returns:
        rdf: a numpy array containing the RDF
        bins: a numpy array containing the bin edges
    """
    rdf_all = np.zeros((len(cations), len(concentrations), nbins))
    rdf_std_all = np.zeros((len(cations), len(concentrations), nbins))
    bins_all = np.zeros((len(cations), len(concentrations), nbins))
    for cat in cations:
        for conc in concentrations:
            if conc == '1M' or conc == '0.5M':
                reps = [1,2,3,4,5]
            else:
                reps = [1,2,3,4,5,6,7,8,9,10]

            system_path = path + f'{cat}/{conc}/'
            filename = system_path + f'rdf/rdf-{fileid}.npy'
            if rerun == True or not os.path.exists(filename):
                rdf_sys = np.zeros((len(reps), nbins))
                bins_sys = np.zeros((len(reps), nbins))
                for i,rep in enumerate(reps):
                    print(f'Running {cat} {conc} rep{rep} -------------------')
                    run_path = system_path + f'rep{rep}/'
                    u = create_mda(
                        run_path,
                        data_file="system.data",
                        dcd_file="nvt_unwrapped_*.dcd",
                        cat_dcd=True,
                        )
                    
                    atom_positions_1 = get_positions(u, u.select_atoms(atom_name))
                    atom_positions_2 = get_positions(u, u.select_atoms(atom_name))

                    r_max = u.dimensions[0]
                    box = u.dimensions[:3]

                    rdf_sys[i,:], bins_sys[i,:] = compute_rdf_one_atom(atom_positions_1, atom_positions_2, box, r_max, nbins)
                # save rdf and bins and average
                if not os.path.exists(system_path + '/rdf'):
                    os.makedirs(system_path + '/rdf')
                np.save(filename, rdf_sys)
                np.save(system_path + f'/rdf/bins-{fileid}.npy', bins_sys)

                # average rdf
                rdf_avg = np.mean(rdf_sys, axis=0)
                rdf_std = np.std(rdf_sys, axis=0)
                print(rdf_std.shape)
                bins_avg = bins_sys.mean(axis=0)
                np.save(system_path + f'/rdf/rdf-{fileid}-avg.npy', rdf_avg)
                np.save(system_path + f'/rdf/rdf-{fileid}-std.npy', rdf_std)
                np.save(system_path + f'/rdf/bins-{fileid}-avg.npy', bins_avg)
            else:
                rdf_sys = np.load(filename)
                rdf_avg = np.load(system_path + f'/rdf/rdf-{fileid}-avg.npy') # need to fix path here so it isn't overwriting the file
                rdf_std = np.load(system_path + f'/rdf/rdf-{fileid}-std.npy')
                bins_avg = np.load(system_path + f'/rdf/bins-{fileid}-avg.npy')
            
            rdf_all[cations.index(cat), concentrations.index(conc), :] = rdf_avg
            rdf_std_all[cations.index(cat), concentrations.index(conc), :] = rdf_std
            bins_all[cations.index(cat), concentrations.index(conc), :] = bins_avg

    return rdf_all, rdf_std_all, bins_all