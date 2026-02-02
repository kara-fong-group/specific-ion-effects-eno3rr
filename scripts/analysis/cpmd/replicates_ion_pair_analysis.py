#=========================================
# Script to run analysis of ion pair statitstics for all systems and replicates
#     Maddy Murphy
#     Feb 2025
#=========================================

# Import libraries
import numpy as np # type: ignore
import MDAnalysis as mda # type: ignore
import pandas as pd # type: ignore
import os
import re
from scipy import interpolate # type: ignore
from scipy.spatial.distance import cdist # type: ignore
from scipy import stats # type: ignore
import glob

def create_mda(path, data_file, dcd_file, cat_dcd=False):
    """
    Create a MDAnalysis Universe object for a simulation.

    Parameters:
        path (str): Path to the directory containing data files.
        data_file (str): Name of the data file.
        dcd_file (str): Name pattern of DCD trajectory files.
        cat_dcd (bool, optional): Whether to concatenate multiple DCD files. Defaults to False.
            use this when runs include extensions like run1a, run1b, etc.

    Returns:
        MDAnalysis Universe: MDAnalysis Universe object.
    """
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
              '5': 'O_NO',
              '6': 'Au',
              '7': 'Au'}

    names = []
    for atom in run.atoms:
        names.append(atom_names[atom.type])
    run.add_TopologyAttr('name', names)

    # print(f"number of atoms: {len(run.atoms)}")
    # print(f"length of trajectory: {len(run.trajectory)}")

    return run

def create_mda_water(path, data_file, dcd_file, cat_dcd=False):
    """
    Create a MDAnalysis Universe object for a simulation.

    Parameters:
        path (str): Path to the directory containing data files.
        data_file (str): Name of the data file.
        dcd_file (str): Name pattern of DCD trajectory files.
        cat_dcd (bool, optional): Whether to concatenate multiple DCD files. Defaults to False.
            use this when runs include extensions like run1a, run1b, etc.

    Returns:
        MDAnalysis Universe: MDAnalysis Universe object.
    """
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
              '3': 'Au',
              '4': 'Au'}

    names = []
    for atom in run.atoms:
        names.append(atom_names[atom.type])
    run.add_TopologyAttr('name', names)

    # print(f"number of atoms: {len(run.atoms)}")
    # print(f"length of trajectory: {len(run.trajectory)}")

    return run

def get_positions(u,atoms):
    """ 
    Get positions of each atom from a MDAnalysis Universe object.

    parameters:
        u (MDAnalysis Universe): MDAnalysis Universe object.
        atoms (list): List of atom names.
        
    returns:
        np.array: Array of atom positions.
    """
    time = 0
    atom_positions = np.zeros((u.trajectory.n_frames, len(atoms), 3)) # time, atom, xyz
    
    for ts in u.trajectory:
        atom_positions[time, :, :] = atoms.positions
        time += 1
    return atom_positions

def get_area(path, run, rerun=False):
    """
    Calculate the area of the electrode (the area of the x-y plane of the simulation box)

    Parameters:
        path (str): Path to the directory where output files will be saved.
        run (MDAnalysis.Universe): MDAnalysis Universe object representing the system trajectory.
        rerun (bool, optional): If True, recompute the area even if the output file exists. Default is False.

    Returns:
        float: Area of the electrode.
    """
    filename = path + f"area.npy"
    if rerun == True or not os.path.exists(filename):
        area = run.dimensions[0] * run.dimensions[1]
        np.save(path + "area.npy", area)
    else:
        area = np.load(path + "area.npy")
    return area 

def compute_density_profile(u, atoms, bins, run_start=0, run_end=None):
    """ 
    Compute the density profile of a system.
    parameters:
        path (str): path to save the ouput density profile
        u (MDAnalysis Universe): MDAnalysis Universe object.
        atoms (list): List of atom names.
        bins (int): Number of bins for the density profile.
        run_start (int): Starting frame of the run.
        
    returns:
        np.array: Density profile.
    """

    # get arrays of the z positions only
    atom_positions = get_positions(u, atoms)[run_start:run_end, :, 2] #% u.dimensions[2]  # get z-position

    density, edges = np.histogram(atom_positions, bins=bins, density=True)

    edges = edges[:-1] + (edges[1] - edges[0]) / 2  # get the center of the bins
    return density, edges

def run_density_analysis(path, u, run_start=0, run_end=None, rerun=False):
    """ 
    Perform density analysis on a run.
    
    Parameters:
        path (str): path to store output data
        u (MDAnalysis Universe): MDAnalysis universe object.
        run_start (int): starting frame of the run
        rerun (bool): whether or not to rerun the analysis
        
    Returns: 
        density, edges: density profile and edges of the bins
        
    """
    print("Starting density analysis.")
    filename = path + "density.npy"
    if rerun==True or not os.path.exists(filename):
        atoms = ['O', 'H', 'C', 'N_NO', 'O_NO', 'Au']
        n_atoms = {}
        binsize = 0.3 # in Angstroms
        print("u dimension: ", u.dimensions[2])
        bins = np.arange(0, u.dimensions[2], binsize)
        density = {}

        for atom in atoms:
            atom_selection = u.select_atoms(f'name {atom}')
            n_atoms[atom] = len(atom_selection)
            density[atom], edges = compute_density_profile(u, atom_selection, bins, run_start=run_start, run_end=run_end)

        if not os.path.exists(path):
            os.makedirs(path)
        np.save(filename, density)
        np.save(path + f'edges.npy', edges)
        np.save(path + f'n-atoms.npy', n_atoms)

    else:
        density = np.load(filename, allow_pickle=True).tolist()
        edges = np.load(path + 'edges.npy', allow_pickle=True)
        n_atoms = np.load(path + 'n-atoms.npy', allow_pickle=True).tolist()

        print("Density analysis complete.")
    return density, edges, n_atoms

def find_interface(density, edges):
    """ 
    Find the interface between the electrolyte and the electrode.
    """
    # find the electrode edge
    idx_electrode = np.where(density['Au'] > 0.1)
    elec_edge_1 = edges[idx_electrode[0][9]]
    elec_edge_2 = edges[idx_electrode[0][10]]
    electrode = [elec_edge_1, elec_edge_2]

    return electrode[1] #cathode side


# get ion pair statistics 
#=========================================
def get_ion_pair_stats(path, run, interface, run_start=0, run_end=-1, skip=1, rerun=False):
    """ 
    Get the ion pair statistics for a given run, wrt to the left electrode(cathode).
    
    Parameters: 
        path (str): The path to the run directory.
        run (MDAnalysis.Universe): The run trajectory.
        interface (list): The z-coordinate of the cathode.
        run_start (int): The starting frame of the run.
        run_end (int): The ending frame of the run.
        skip (int): The number of frames to skip in the trajectroy analysis.
        rerun (bool): Whether to rerun the analysis.

    Returns:
        tuple: A tuple containing arraays of ion pair distances, distances from electrode, orientations, 
            NO3 dist from electrode, and Na dist from electrode.
    """  
    print("Starting ion pairing analysis.")
    filename = path + f"ion_pair_distances.npy"
    if rerun == True or not os.path.exists(filename):
        no3_atoms = run.select_atoms("name N_NO")
        na_atoms = run.select_atoms("name C")

        nitrate_positions = get_positions(run, no3_atoms)
        cation_positions = get_positions(run, na_atoms)

        # initialize arrays 
        ion_pair_distances = np.empty(
            (len(run.trajectory[run_start:run_end:skip]), len(no3_atoms), len(na_atoms)), 
            dtype=float,
            )
        
        no3_dists_from_electrode = np.empty(
            (len(run.trajectory[run_start:run_end:skip]), len(no3_atoms)), 
            dtype=float,
            )
        
        ion_pair_orientations = np.empty(
            (len(run.trajectory[run_start:run_end:skip]), len(no3_atoms), len(na_atoms)), 
            dtype=float,
            )
        
        print("Computing ion pair statistics...")

        print(len(run.trajectory[run_start:run_end:skip]))

        # loop through each frame in the trajectory
        for i,ts in enumerate(run.trajectory[run_start:run_end:skip]):
            no3_positions = nitrate_positions[i,:,:]
            na_positions = cation_positions[i,:,:]

            # loop through each NO3 ion
            for j in range(len(no3_atoms)):
                no3 = no3_positions[j]
                no3_dists_from_electrode[i,j] = np.abs(no3[2] - interface)

                # loop through each Na ion
                for k in range(len(na_atoms)):
                    na = na_positions[k]

                    # compute pair distance, applying the minimum image convention
                    delta = no3 - na
                    for idx in range(3):
                        delta[idx] = delta[idx] - run.dimensions[idx] * np.round(delta[idx] / run.dimensions[idx])

                    ion_pair_distances[i,j,k] = np.linalg.norm(delta)
                    ion_pair_orientations[i,j,k] = np.dot(delta, [0,0,1]) / np.linalg.norm(delta)

        # save the data
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + "ion_pair_distances.npy", ion_pair_distances)
        np.save(path + "ion_pair_orientations.npy", ion_pair_orientations)
        np.save(path + "no3_dists_from_electrode.npy", no3_dists_from_electrode)
    
    else:
        ion_pair_distances = np.load(path + "ion_pair_distances.npy")
        ion_pair_orientations = np.load(path + "ion_pair_orientations.npy")
        no3_dists_from_electrode = np.load(path + "no3_dists_from_electrode.npy")


    print("Ion pairing statistics complete.")
    return (ion_pair_distances, ion_pair_orientations, no3_dists_from_electrode)


### Run analysis for all systems and replicates
cations = ['Na'] #['Cs', 'K', 'Na', 'Li']
potential = ['10', '20']
reps = ['1', '2'] #, '1', '2', '3', '4']

base_dir = os.getcwd()
start = 1000
end = 10001

rerun = True
for cat in cations:
    for pot in potential:
        for rep in reps:
            data_path = f"../data/{cat}/lj_a/{pot}/rep{rep}/"
            simulation_path = f"../simulations/{cat}/cpmd/lj_a/{pot}/rep{rep}/"

            # make sure the data path exists
            if not os.path.exists(data_path):
                print(f"making directory {data_path}")
                os.makedirs(data_path)

            run = create_mda(simulation_path, 
                             'system.data', 
                             'traj_unwrapped_*.dcd',
                             cat_dcd=True)
            
            area = get_area(data_path, run, rerun=rerun)
            density, edges, n_atoms = run_density_analysis(data_path, run, run_start=start, run_end=end, rerun=rerun)
            interface = find_interface(density, edges)
            print(f"Interface found for {cat} {pot} rep{rep}: {interface}")

            (
                ion_pair_distances, 
                ion_pair_orientations, 
                no3_dists_from_electrode,
                ) = get_ion_pair_stats(data_path, run, interface, run_start=start, run_end=end, skip=1, rerun=rerun)

            print(f"Analysis complete for {cat} {pot} rep{rep}.")

            # delete the universe object
            del run

        print(f"Analysis complete for {cat} {pot}.")
print("All analysis complete.")
