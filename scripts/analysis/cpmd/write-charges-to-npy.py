### Script to write charges from a full charges.out file to top-charges.out and bot-charges.out ###
# Maddy Murphy #
# April 2025 #

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
import pandas as pd
import re 
import MDAnalysis as mda
import glob
import os

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

    run = mda.Universe(path + data_file, dcd_file, format="LAMMPS", atom_style='id resid type x y z')
    
    atom_names = {'1': 'O',
              '2': 'H',
              '3': 'C',
              '4': 'N_NO',
              '5': 'O_NO',
              '6': 'anode',
              '7': 'cathode'}

    names = []
    for atom in run.atoms:
        names.append(atom_names[atom.type])
    run.add_TopologyAttr('name', names)

    return run

def electrode_ids(run):
    anode_ids = [atom.id for atom in run.atoms if atom.name == 'anode']
    cathode_ids = [atom.id for atom in run.atoms if atom.name == 'cathode']

    return anode_ids, cathode_ids


def extract_and_save_charges(dump_file, anode_ids, cathode_ids,
                             cathode_charges_all, anode_charges_all, timesteps):
    """
    Extracts charges for anode and cathode from the LAMMPS dump file and saves as .npy files.

    Parameters:
        dump_file (str): Path to the LAMMPS dump file.
        anode_ids (list of int): atom IDs for anode.
        cathode_ids (list of int): atom IDs for cathode.
        anode_output (str): .npy filename for anode.
        cathode_output (str): .npy filename for cathode.
    """
    anode_ids_set = set(anode_ids)
    cathode_ids_set = set(cathode_ids)


    with open(dump_file, 'r') as file:
        lines = file.readlines()

    i = 0 # line index
    # Loop through lines to find the start of each timestep
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            # Skip timestep and number of atoms
            i += 2  # TIMESTEP line + timestep value
            timesteps.append(int(lines[i-1].strip()))
            i += 2  # NUMBER OF ATOMS line + number
            i += 5  # BOX BOUNDS lines (4 lines) + ITEM: ATOMS line


            # Start reading atom data
            timestep_anode_charges = []
            timestep_cathode_charges = []

            while i < len(lines) and not lines[i].startswith("ITEM: TIMESTEP"):
                parts = lines[i].strip().split()
                if len(parts) == 2:
                    atom_id = int(parts[0])
                    charge = float(parts[1])
                    if atom_id in anode_ids_set:
                        timestep_anode_charges.append((atom_id, charge))
                    elif atom_id in cathode_ids_set:
                        timestep_cathode_charges.append((atom_id, charge))
                i += 1

            # Sort charges by atom ID to maintain consistent order
            timestep_anode_charges.sort()
            timestep_cathode_charges.sort()

            anode_charges_all.append(timestep_anode_charges)
            cathode_charges_all.append(timestep_cathode_charges)
        else:
            i += 1

    return cathode_charges_all, anode_charges_all, timesteps

def main():
    # basedir = "/Users/maddymurphy/Documents/caltech/phd/research-projects/nitrate-reduction/cpmd-simulations/"
    # basedir = "/home/mamurphy/nitrate-reduction/cpmd-simulations/"
    basedir = "/anvil/scratch/x-mmurphy4/"
    path = f"{basedir}simulations/"
    data_path = f"{basedir}data/"

    cations = ['Cs','K','Na','Li']
    potentials = ['00','10','20']
    reps = ['5','6','7']


    for cation in cations:
        for potential in potentials:
            for rep in reps:
                print(f"Processing {cation} at {potential} V, rep {rep}")
                foldername = f"{path}{cation}/cpmd/wca/{potential}/rep{rep}/"
                data_folder = f"{data_path}{cation}/wca/{potential}/rep{rep}/"
                dump_file = f"{foldername}charges.lammpstrj"
                anode_file = f"{data_folder}anode_charges.npy"
                cathode_file = f"{data_folder}cathode_charges.npy"

                anode_ids = np.arange(7726,9416,1)
                cathode_ids = np.arange(9416,(9416+1690),1)

                anode_charges_all = []
                cathode_charges_all = []
                timesteps = []

                if os.path.exists(dump_file):
                    cathode_charges_all, anode_charges_all, timesteps = extract_and_save_charges(dump_file, anode_ids, cathode_ids,
                                            cathode_charges_all, anode_charges_all, timesteps)

                # Convert to NumPy arrays
                anode_array = np.array(anode_charges_all)
                cathode_array = np.array(cathode_charges_all)

                # make sure directories exist
                os.makedirs(os.path.dirname(anode_file), exist_ok=True)
                os.makedirs(os.path.dirname(cathode_file), exist_ok=True)

                np.save(anode_file, anode_array)
                np.save(cathode_file, cathode_array)
                print(f"Saved anode charges to '{anode_file}' with shape {anode_array.shape}")
                print(f"Saved cathode charges to '{cathode_file}' with shape {cathode_array.shape}")
                
                print(f"Finished processing {cation} at {potential} V, rep {rep}")
                print("")
if __name__ == "__main__":
    main()