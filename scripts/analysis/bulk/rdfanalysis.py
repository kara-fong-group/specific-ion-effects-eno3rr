""" 
script with functions used to analyze the RDFs of different systems. Includes the necessary function to calculate the pmf, coordination number, and the first peak of the RDF.
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

def compute_rdf(atom_positions_1, atom_positions_2, box, r_max, n_bins):
    """ 
    Compute the radial distribution function for the system. 
    
    parameters: 
        atom_positions_1: a numpy array of the center atom positions, reference atoms
        atom_positions_2: a numpy array of the surrounding atom positions
        box: a numpy array of shape (3, 3) containing the box vectors
        r_max: the maximum distance to calculate the RDF
        n_bins: the number of bins to use for the RDF

    returns:
        rdf: a numpy array containing the RDF
        bins: a numpy array containing the bin edges
    """
    n_frames = atom_positions_1.shape[0]
    n_ref_atoms = atom_positions_1.shape[1]
    n_atoms = atom_positions_2.shape[1]

    dr = r_max / n_bins
    rdf = np.zeros((n_frames, n_bins))
    bins = np.arange(0, r_max + dr, dr)

    for frame in range(n_frames):
        for ref_atom in range(n_ref_atoms):
            for atom in range(n_atoms):
                delta = atom_positions_1[frame, ref_atom] - atom_positions_2[frame, atom] # calculate the distance between the atoms (vector)
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
    
def run_rdf(path, cations, concentrations, nbins, atom1_name, atom2_name, fileid, rerun=False):
    """ 
    Compute the radial distribution function for the system.
    parameters:
        path (str): the path to the directory containing the trajectory files
        cations (list): a list of cation names
        concentrations (list): a list of concentrations
        nbins (int): the number of bins to use for the RDF
        atom1_name (str): the name of the center atom
        atom2_name (str): the name of the surrounding atom
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
                    
                    atom_positions_1 = get_positions(u, u.select_atoms(atom1_name))
                    atom_positions_2 = get_positions(u, u.select_atoms(atom2_name))

                    r_max = u.dimensions[0]
                    box = u.dimensions[:3]

                    rdf_sys[i,:], bins_sys[i,:] = compute_rdf(atom_positions_1, atom_positions_2, box, r_max, nbins)
                # save rdf and bins and average
                if not os.path.exists(system_path + '/rdf'):
                    os.makedirs(system_path + '/rdf')
                np.save(filename, rdf_sys)
                np.save(system_path + f'/rdf/bins-{fileid}.npy', bins_sys)

                # average rdf
                rdf_avg = np.mean(rdf_sys, axis=0)
                rdf_std = np.std(rdf_sys, axis=0)
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


def plot_rdf(rdf, rdf_std, bins, cations, concentrations, 
             path, fileid, today, colors, xlim, ylim,
             save=False):
    """
    Plot the radial distribution function for the system.
    parameters: 
        rdf: a numpy array containing the RDF
        rdf_std: a numpy array containing the standard deviation of the RDF
        bins: a numpy array containing the bin edges
        cations (list): a list of cation names
        concentrations (list): a list of concentrations
        
        """
    fig, ax = plt.subplots(1, 1, figsize=(6,4))

    for i, cat in enumerate(cations):
        for j, conc in enumerate(concentrations):
            ax.plot(bins[i, j,:], rdf[i, j,:], label=f'{cat}', color=colors[i])
            ax.fill_between(bins[i,j,:], rdf[i,j,:] - rdf_std[i,j], rdf[i,j] + rdf_std[i,j], color=colors[i], alpha=0.5)

    ax.set_xlabel('r (Angstroms)')
    ax.set_ylabel('g(r)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()
    if save:
        fig.savefig(f'{path}/rdf/rdf-{fileid}-{today}.png', dpi=300)

    return fig, ax


## rdf for one atom type ----------------------------------------------------------------------------------
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
                    
                    atom_positions = get_positions(u, u.select_atoms(atom_name))

                    r_max = u.dimensions[0]
                    box = u.dimensions[:3]

                    rdf_sys[i,:], bins_sys[i,:] = compute_rdf_one_atom(atom_positions, box, r_max, nbins)
                # save rdf and bins and average
                if not os.path.exists(system_path + '/rdf'):
                    os.makedirs(system_path + '/rdf')
                np.save(filename, rdf_sys)
                np.save(system_path + f'/rdf/bins-{fileid}.npy', bins_sys)

                # average rdf
                rdf_avg = np.mean(rdf_sys, axis=0)
                rdf_std = np.std(rdf_sys, axis=0)
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

## rdf using MDAnalysis ----------------------------------------------------------------------------------
def run_rdf_mdanalysis(path, cations, concentrations, nbins, atom1_name, atom2_name, fileid, rerun=False):
    """ 
    Compute the radial distribution function for the system.
    parameters:
        path (str): the path to the directory containing the trajectory files
        cations (list): a list of cation names
        concentrations (list): a list of concentrations
        nbins (int): the number of bins to use for the RDF
        atom1_name (str): the name of the center atom
        atom2_name (str): the name of the surrounding atom
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
                    
                    group1 =  u.select_atoms(atom1_name)
                    group2 = u.select_atoms(atom2_name)

                    rdf = InterRDF(group1, group2, nbins=nbins)
                    rdf.run()

                    rdf_sys[i,:] = rdf.rdf
                    bins_sys[i,:] = rdf.bins


                # save rdf and bins and average
                if not os.path.exists(system_path + '/rdf'):
                    os.makedirs(system_path + '/rdf')
                np.save(filename, rdf_sys)
                np.save(system_path + f'/rdf/bins-{fileid}.npy', bins_sys)

                # average rdf
                rdf_avg = np.mean(rdf_sys, axis=0)
                rdf_std = np.std(rdf_sys, axis=0)
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


## Potential of Mean Force (PMF) Calculation--------------------------------------------------------------
def pmf(g, T):
    kb = 1.380649e-23 # boltzmann constant in J/K
    w = - kb * T * np.log(g)
    return w

def run_pmf(path, cations, concentrations, nbins, fileid, T=298.15, rerun=False):
    """ 
    Compute the potential of mean force for the system.
    parameters:
        path (str): the path to the directory containing the trajectory files
        cations (list): a list of cation names
        concentrations (list): a list of concentrations
        nbins (int): the number of bins to use for the RDF
        atom1_name (str): the name of the center atom
        atom2_name (str): the name of the surrounding atom
        fileid (str): the name of the fileid to save the RDF .npy files
        rerun (bool): whether to rerun the RDF calculation
        
    returns:
        pmf_all: a numpy array containing the PMF
        bins: a numpy array containing the bin edges
    """
    pmf_all = np.zeros((len(cations), len(concentrations), nbins))
    pmf_std_all = np.zeros((len(cations), len(concentrations), nbins))

    if rerun == True or not os.path.exists(path + f'pmf/pmf-{fileid}-avg.npy'):
        for i, cat in enumerate(cations):
            for j, conc in enumerate(concentrations):
                if conc == '1M' or conc == '0.5M':
                    reps = [1,2,3,4,5]
                else:
                    reps = [1,2,3,4,5,6,7,8,9,10]

                system_path = path + f'{cat}/{conc}/'
                filename = system_path + f'pmf/pmf-{fileid}.npy'
                if rerun == True or not os.path.exists(filename):
                    pmf_sys = np.zeros((len(reps), nbins))
                    
                    rdf_filename = system_path + f'rdf/rdf-{fileid}.npy'
                    if not os.path.exists(rdf_filename):
                        return 'RDF file not found'
                    
                    rdf_sys = np.load(rdf_filename)

                    for k,rep in enumerate(reps):
                        pmf_sys[k,:] = pmf(rdf_sys[k,:], 298)
                    
                    # save pmf and average
                    if not os.path.exists(system_path + 'pmf'):
                        os.makedirs(system_path + 'pmf')
                    np.save(filename, pmf_sys)

                    # average pmf
                    pmf_avg = np.mean(pmf_sys, axis=0)
                    pmf_std = np.std(pmf_sys, axis=0)

                    pmf_all[i,j,:] = pmf_avg
                    pmf_std_all[i,j,:] = pmf_std
       
        if not os.path.exists(path + 'pmf'):
            os.makedirs(path + 'pmf')
        np.save(path + f'pmf/pmf-{fileid}.npy', pmf_all)
        np.save(path + f'pmf/pmf-{fileid}-std.npy', pmf_std_all)
    else:
        pmf_all = np.load(path + f'pmf/pmf-{fileid}.npy')
        pmf_std_all = np.load(path + f'pmf/pmf-{fileid}-std.npy')
    
    return pmf_all, pmf_std_all

## Coordination Number Calculation--------------------------------------------------------------

def cutoff_dist(pmf, r):
    """ 
    Find the cutoff distance for the PMF.
    
    parameters:
        pmf: a numpy array containing the PMF
        r: a numpy array containing the distances corresponding to the PMF
        
    returns:
        cutoff: the cutoff distance
    """
    # determine cutoff distances from pmf
    cutoff_idx = argrelextrema(pmf, np.greater, order=10)[0][0]
    cutoff = r[cutoff_idx]
    return cutoff, cutoff_idx

def coordination_number(pmf, bins):
    """ 
    Compute the coordination number for the system.
    
    parameters:
        pmf: a numpy array containing the PMF
        bins: a numpy array containing the bin edges

    returns:
        cn: the coordination number
    """
    # determine coordination number from pmf
    r = 0.5 * (bins[:-1] + bins[1:])
    _, cutoff_idx = cutoff_dist(pmf, r)
    pmf_cut = pmf[:cutoff_idx]
    r_cut = r[:cutoff_idx]
    cn = np.trapz(np.exp(-pmf_cut), r_cut)

    return cn

def run_cn(path, cations, concentrations, fileid, rerun=False):
    """ 
    Compute the coordination number for atom pairs in the system.
    
    parameters:
        path (str): the path to the directory containing the trajectory files
        cations (list): a list of cation names
        concentrations (list): a list of concentrations
        fileid (str): the name of the fileid to save the RDF .npy files
        rerun (bool): whether to rerun the RDF calculation
        
    returns:
        cn_avg: a numpy array containing the coordination number -- average across reps
        cn_std: a numpy array containing the standard deviation of the coordination number
    """
    cn_all = np.zeros((len(cations), len(concentrations)))
    cn_std_all = np.zeros((len(cations), len(concentrations)))
                          
    if rerun == True or not os.path.exists(path + f'/cn/cn-{fileid}-avg.npy'):
        for cat in cations:
            for conc in concentrations:
                if conc == '1M' or conc == '0.5M':
                    reps = [1,2,3,4,5]
                else:
                    reps = [1,2,3,4,5,6,7,8,9,10]

                system_path = path + f'{cat}/{conc}/'
                filename = system_path + f'cn/cn-{fileid}.npy'
                if rerun == True or not os.path.exists(filename):
                    cn_sys = np.zeros(len(reps))
                    
                    pmf_filename = system_path + f'pmf/pmf-{fileid}.npy'
                    if not os.path.exists(pmf_filename):
                        return 'PMF file not found'
                    
                    pmf_sys = np.load(pmf_filename)
                    bins_filename = system_path + f'rdf/bins-{fileid}.npy'
                    bins_sys = np.load(bins_filename)

                    for i,rep in enumerate(reps):
                        cn_sys[i] = coordination_number(pmf_sys[i,:], bins_sys[i,:])
                    
                    # save cn and average
                    if not os.path.exists(system_path + 'cn'):
                        os.makedirs(system_path + 'cn')
                    np.save(filename, cn_sys)

                    # average cn
                    cn_avg = np.mean(cn_sys)
                    cn_std = np.std(cn_sys)

                    cn_all[cations.index(cat), concentrations.index(conc)] = cn_avg
                    cn_std_all[cations.index(cat), concentrations.index(conc)] = cn_std

        if not os.path.exists(path + 'cn'):
            os.makedirs(path + 'cn')
        np.save(path + f'cn/cn-{fileid}-avg.npy', cn_avg)
        np.save(path + f'cn/cn-{fileid}-std.npy', cn_std)
    else:
        cn_avg = np.load(path + f'cn/cn-{fileid}-avg.npy')
        cn_std = np.load(path + f'cn/cn-{fileid}-std.npy')
    
    return cn_avg, cn_std

## Density of System ----------------------------------------------------------------------------------
def run_system_density(path, cations, concentrations, rerun=False):
    """ 
    Extract the average density from NVT simulations.
    
    parameters:
        path (str): the path to the directory containing the lammps file
        cations (list): a list of cation names
        concentrations (list): a list of concentrations
        fileid (str): the name of the fileid to save the density .npy files
        rerun (bool): whether to rerun the density calculation
        
    returns:
        density_avg: a numpy array containing the density -- average across reps
        density_std: a numpy array containing the standard deviation of the density
    """
    density_all = np.zeros((len(cations), len(concentrations)))
    density_std_all = np.zeros((len(cations), len(concentrations)))
                          
    if rerun == True or not os.path.exists(path + f'/density/density-avg.npy'):
        for cat in cations:
            for conc in concentrations:
                if conc == '1M' or conc == '0.5M':
                    reps = [1,2,3,4,5]
                else:
                    reps = [1,2,3,4,5,6,7,8,9,10]

                system_path = path + f'{cat}/{conc}/'
                filename = system_path + f'density.npy'
                if rerun == True or not os.path.exists(filename):
                    density = []
                    for i, rep in enumerate(reps):
                        # Initialize the data df and the columns
                        columns = ["Time", "Step", "Temp", "Press", "Volume", "Density", "PotEng", "KinEng", "TotEng", "Enthalpy", "Fmax"]
                        data = []

                        # define the data pattern 
                        log_pattern = r"^\s*(\d+)\s+(\d+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)"
                        logfile = system_path + f'rep{rep}/nvt.log'
                        with open(logfile, 'r') as file:
                            for line in file:
                                # Match all the data to its respective property
                                match = re.match(log_pattern, line)
                                if match:
                                    # Append the matched values as a tuple to the data list
                                    data.append(match.groups())

                        # Create a df
                        columns = ["Time", "Step", "Temp", "Press", "Volume", "Density", "PotEng", "KinEng", "TotEng", "Enthalpy", "Fmax"]
                        print(len(data))
                        # remove the first 500 steps
                        data = data[500:]
                        df = pd.DataFrame(data, columns=columns)
                        df = df.astype(float)

                        # Find the average density
                        avg_density = df['Density'].mean()
                        density.append(avg_density)
                    
                    # save density and average
                    np.save(filename, density)

                    # average density across replicates
                    density_avg = np.mean(density)
                    density_std = np.std(density)

                    density_all[cations.index(cat), concentrations.index(conc)] = density_avg
                    density_std_all[cations.index(cat), concentrations.index(conc)] = density_std

        if not os.path.exists(path + 'density'):
            os.makedirs(path + 'density')
        np.save(path + f'density/density-avg.npy', density_all)
        np.save(path + f'density/density-std.npy', density_std_all)
    else:
        density_all = np.load(path + f'density/density-avg.npy')
        density_std_all = np.load(path + f'density/density-std.npy')
    
    return density_all, density_std_all