import numpy as np
import MDAnalysis as mda
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import re
import pandas as pd

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

def read_log_file(folder_name,logfile):
    columns = ["Time", "Step", "Temp", "Press", "Volume", "Density", "PotEng", "KinEng", "TotEng", "Enthalpy", "Fmax"]
    data = []

    # define the data pattern 
    log_pattern = r"^\s*(\d+)\s+(\d+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)"
    log_file = os.path.join(folder_name,logfile)
    with open(log_file, 'r') as file:
        for line in file:
            # Match all the data to its respective property
            match = re.match(log_pattern, line)
            if match:
                # Append the matched values as a tuple to the data list
                data.append(match.groups())

    # Create a df
    columns = ["Time", "Step", "Temp", "Press", "Volume", "Density", "PotEng", "KinEng", "TotEng", "Enthalpy", "Fmax"]
    df = pd.DataFrame(data, columns=columns)
    df = df.astype(float)

    volume = df['Volume'].mean()
    return volume

def particle_density(path):
    # create MDA universe
    u = create_mda(
            path,
            data_file="system.data",
            dcd_file="nvt_unwrapped_*.dcd",
            cat_dcd=True,
            )

    # Define atom groups
    ag_O = u.select_atoms('name O')
    ag_H = u.select_atoms('name H')
    ag_C = u.select_atoms('name C')
    ag_N_NO3 = u.select_atoms('name N_NO')
    ag_O_NO3 = u.select_atoms('name O_NO')

    # number of cations = number of nitrate ions
    n_c = len(ag_C)

    # retrieve volume [=] Angstroms^3
    logfile = "nvt.log"
    volume = read_log_file(path,logfile)

    # compute particle density
    particle_density = n_c/volume

    return particle_density

def particle_density_water(path):
    # create MDA universe
    u = create_mda(
            path,
            data_file="system.data",
            dcd_file="nvt_unwrapped_*.dcd",
            cat_dcd=True,
            )

    # Define atom groups
    ag_O = u.select_atoms('name O')
    ag_H = u.select_atoms('name H')
    ag_C = u.select_atoms('name C')
    ag_N_NO3 = u.select_atoms('name N_NO')
    ag_O_NO3 = u.select_atoms('name O_NO')

    # number of cations = number of nitrate ions
    n_w = len(ag_O)

    # retrieve volume [=] Angstroms^3
    logfile = "nvt.log"
    volume = read_log_file(path,logfile)

    # compute particle density
    particle_density = n_w/volume

    return particle_density