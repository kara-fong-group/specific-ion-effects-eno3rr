import numpy as np
import MDAnalysis as mda
import glob
import os
from scipy.signal import argrelextrema

import ionpairing as ion_pairing

samplingperiod = 1

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
              '5': 'O_NO'}

    names = []
    for atom in run.atoms:
        names.append(atom_names[atom.type])
    run.add_TopologyAttr('name', names)
    
    return run


def generate_times(u, run_start, timestep=samplingperiod):
    """
    Generate time array for simulation frames.

    Parameters:
        u (MDAnalysis Universe): MDAnalysis Universe object.
        run_start (int): Index of the starting frame.
        timestep (float, optional): Time step between frames in picoseconds. Defaults to 0.5.

    Returns:
        numpy.ndarray: Array of time values.
    """
    times = []
    current_step = 0
    for ts in u.trajectory[run_start:]:  # omit equilibration time
        times.append(current_step * timestep)
        current_step += 1
    times = np.array(times)
    return times

def run_pmf_analysis(cations, runs, run_start, rerun=False):
    """
    Perform Potential of Mean Force (PMF) analysis.

    Parameters:
        cations (list): List of cation identities.
        runs (list): List of replicate run indices.
        run_start (int): Index of the starting frame.
        rerun (bool, optional): Whether to rerun the analysis even if results exist. Defaults to False.
    """

    filename = "data/pmf_avg_ncat.npy"
    if os.path.exists(filename):
        pmf_avg = np.load(filename)

        # find first maximum of pmf
        cip_dists = np.empty(len(cations))
        ssip_dists = np.empty(len(cations))

        for j, cat in enumerate(cations):
            pmf = pmf_avg[j]
            maximas = argrelextrema(pmf)
            cip_dists = maximas[0] # first maxima
            ssip_dists = maximas[1] # second maxima


def run_ion_pairing_fracs(cations, runs, cip_dists, ssip_dists, run_start, rerun=False):
    """
    Compute ion pairing fractions.

    Parameters:
        cations (list): List of cation identities.
        runs (list): List of replicate run indices.
        cip_dists (numpy.ndarray): Cutoff distances for contact ion pairs.
        ssip_dists (numpy.ndarray): Cutoff distances for solvent-separated ion pairs.
        run_start (int): Index of the starting frame.
        rerun (bool, optional): Whether to rerun the analysis even if results exist. Defaults to False.
    """
    filename = "data/ion_pairing_fracs.npy"
    if rerun == True or not os.path.exists(filename):
        ion_pairing_all = np.empty((len(cations), len(runs), 3), dtype=float)
        for j, sep in enumerate(cations):
            for i, run in enumerate(runs):
                path = f"{cat}/run{run}/"
                print(path)
                u = create_mda(
                    path,
                    data_file="../run_initial/end_equil.data",
                    dcd_file="traj_unwrapped_*.dcd",
                    cat_dcd=True,
                )
                ion_pairing_all[j, i, :], _ = ion_pairing.compute_ionPairFrac(
                    path,
                    u,
                    cip_dist=cip_dists[j],
                    ssip_dist=ssip_dists[j],
                    run_start=run_start,
                    rerun=rerun,
                )
        ion_pairing_avg = np.empty((len(cations), 3), dtype=float)
        ion_pairing_err = np.empty((len(cations), 3), dtype=float)
        for j, cat in enumerate(cations):
            (
                ion_pairing_avg[j, :],
                ion_pairing_err[j, :],
            ) = ion_pairing.average_replicates(ion_pairing_all[j, :])
        np.save("data/ion_pairing_fracs.npy", ion_pairing_avg)
        np.save("data/ion_pairing_fracs_err.npy", ion_pairing_err)


def run_ion_pair_residence_times(cations, runs, run_start, cip_cutoffs, rerun=False):
    """
    Compute ion pair residence times.

    Parameters:
        cations (list): List of cation identities.
        runs (list): List of replicate run indices.
        run_start (int): Index of the starting frame.
        cip_cutoffs (numpy.ndarray): Cutoff distances for contact ion pairs.
        rerun (bool, optional): Whether to rerun the analysis even if results exist. Defaults to False.
    """
    filename = "data/ion_pair_residence_times.npy"
    if rerun == True or not os.path.exists(filename):
        ion_pair_residence_times = np.empty((len(cations), len(runs)), dtype=object)
        ion_pair_acfs = np.empty((len(cations), len(runs)), dtype=object)
        times_all = np.empty((len(cations), len(runs)), dtype=object)
        for j, cat in enumerate(cations):
            for i, run in enumerate(runs):
                path = f"{cat}/run{run}/"
                print(path)
                u = create_mda(
                    path,
                    data_file="../run_initial/end_equil.data",
                    dcd_file="traj_unwrapped_*.dcd",
                    cat_dcd=True,
                )
                times = generate_times(u, run_start)
                times_all[j, i] = times
                (
                    ion_pair_residence_times[j, i],
                    ion_pair_acfs[j, i],
                ) = ion_pairing.calc_neigh_corr(
                    path,
                    u,
                    u.select_atoms("name Na"),
                    times,
                    cutoff_dist=cip_cutoffs[j],
                    run_start=run_start,
                    rerun=rerun,
                )
        ion_pair_residence_avg = np.empty((len(cations)), dtype=object)
        ion_pair_residence_err = np.empty((len(cations)), dtype=object)
        ion_pair_acf = np.empty((len(cations)), dtype=object)
        ion_pair_acf_err = np.empty((len(cations)), dtype=object)
        for j, cat in enumerate(cations):
            (
                ion_pair_residence_avg[j],
                ion_pair_residence_err[j],
            ) = ion_pairing.average_replicates(ion_pair_residence_times[j, :])
            min_len = np.min([len(ion_pair_acfs[j, i]) for i in range(len(runs))])
            ion_pair_acf[j] = np.mean(
                [ion_pair_acfs[j, i][:min_len] for i in range(len(runs))], axis=0
            )
            ion_pair_acf_err[j] = np.std(
                [ion_pair_acfs[j, i][:min_len] for i in range(len(runs))], axis=0
            )

        np.save("data/ion_pair_residence_times.npy", ion_pair_residence_avg)
        np.save("data/ion_pair_residence_times_err.npy", ion_pair_residence_err)
        np.save("data/ion_pair_acf.npy", ion_pair_acf)
        np.save("data/ion_pair_acf_err.npy", ion_pair_acf_err)
        np.save("data/times.npy", times_all)


def run_coordination_analysis(cations, runs, run_start, rerun=False):
    """
    Perform coordination number analysis.

    Parameters:
        cations (list): List of cation identities.
        runs (list): List of replicate run indices.
        run_start (int): Index of the starting frame.
        rerun (bool, optional): Whether to rerun the analysis even if results exist. Defaults to False.
    """
    filename = "data/coordination_numbers.npy"
    if rerun == True or not os.path.exists(filename):
        coordination_numbers = np.empty((len(cations), len(runs), 2), dtype=object)
        coordination_numbers_err = np.empty((len(cations), len(runs), 2), dtype=object)
        coordination_bins = np.empty((len(cations)), dtype=object)
        for j, cat in enumerate(cations):
            for i, run in enumerate(runs):
                path = f"{cat}/run{run}/"
                print(path)
                u = create_mda(
                    path,
                    data_file="../run_initial/end_equil.data",
                    dcd_file="traj_unwrapped_*.dcd",
                    cat_dcd=True,
                )
                (
                    coordination_bins[j],
                    coordination_numbers[j, i, 0],
                    coordination_numbers[j, i, 1],
                    coordination_numbers_err[j, i, 0],
                    coordination_numbers_err[j, i, 1],
                ) = ion_pairing.coordination_analysis(
                    path, u, cations, run_start=run_start, rerun=rerun
                )
        np.save("data/coordination_numbers.npy", coordination_numbers)
        np.save("data/coordination_numbers_err.npy", coordination_numbers_err)
        np.save("data/coordination_bins.npy", coordination_bins)

