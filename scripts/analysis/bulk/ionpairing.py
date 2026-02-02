import numpy as np
import MDAnalysis as mda
import os
import re
from scipy import interpolate
from scipy.spatial.distance import cdist
from scipy import stats


def compute_ionPairFrac(
    path, run, cip_dist=3.85, ssip_dist=6.25, skip=1, run_start=0, rerun=False
):
    """
    Compute the fraction of ions that are in CIPs (Contact Ion Pairs), SSIPs (Solvent-Separated Ion Pairs),
    and free ions based on given cutoff distances.

    C: cation
    N_NO3: anion reference point --> nitrogen atom in nitrate ion

    Parameters:
        path (str): Path to the directory where output files will be saved.
        run (MDAnalysis.Universe): MDAnalysis Universe object representing the system trajectory.
        cip_dist (float, optional): Cutoff distance for CIPs. Default is 3.85 Angstrom. --> UNDERSTAND IF THIS NEEDS TO CHANGE
        ssip_dist (float, optional): Cutoff distance for SSIPs. Default is 6.25 Angstrom.
        skip (int, optional): Number of frames to skip in trajectory analysis. Default is 1.
        run_start (int, optional): Index of the starting frame for analysis. Default is 0.
        rerun (bool, optional): If True, recompute the analysis even if the output file exists. Default is False.


    Returns:
        tuple: A tuple containing average fraction of ions in CIPs, SSIPs, and free ions,
               and an array of fractions for each frame.
    """
    filename = path + f"ion_pairing.npy"
    if rerun == True or not os.path.exists(filename):
        cations = run.select_atoms("name C") # cation atoms

        frac_cip = []
        frac_free = []
        frac_ssip = []

        # iterate through each frame in the trajectory and classifying each ion based on distance from nitrate ion
        for ts in run.trajectory[run_start:-1:skip]:
            free_cat = run.select_atoms(f"name C and not around {ssip_dist} name N_NO3")
            bound_cat = run.select_atoms(f"name C and around {cip_dist} name N_NO3")

            frac_cip.append(bound_cat.atoms.n_atoms / float(cations.atoms.n_atoms))
            frac_free.append(free_cat.atoms.n_atoms / float(cations.atoms.n_atoms))
            frac_ssip.append(1 - frac_cip[-1] - frac_free[-1])

        avg_frac_cip = np.mean(np.asarray(frac_cip))
        avg_frac_ssip = np.mean(np.asarray(frac_ssip))
        avg_frac_free = 1 - avg_frac_cip - avg_frac_ssip

        avg_pairing = [avg_frac_cip, avg_frac_ssip, avg_frac_free]
        pairing = [frac_cip, frac_ssip, frac_free]

        np.save(path + "ion_pairing.npy", pairing)
        np.save(path + "avg_ion_pairing.npy", avg_pairing)

    else:
        pairing = np.load(path + "ion_pairing.npy")
        avg_pairing = np.load(path + "avg_ion_pairing.npy")

    return avg_pairing, pairing


def get_ion_pair_stats(
    path, run, run_start=0, skip=1, cato_dist=3.36045, nh_dist=2.9337, rerun=False
):
    """
    Compute various statistics related to ion pairs such as distances, heights, orientations, and coordinations.

    Parameters:
        path (str): Path to the directory where output files will be saved.
        run (MDAnalysis.Universe): MDAnalysis Universe object representing the system trajectory.
        run_start (int, optional): Index of the starting frame for analysis. Default is 0.
        skip (int, optional): Number of frames to skip in trajectory analysis. Default is 1.
        nao_dist (float, optional): Distance cutoff for Na-O coordination. Default is 3.36045 Angstrom. --> change to cato_dist
        clh_dist (float, optional): Distance cutoff for Cl-H coordination. Default is 2.9337 Angstrom. --> change to nh_dist
        rerun (bool, optional): If True, recompute the analysis even if the output file exists. Default is False.

    Returns:
        tuple: A tuple containing arrays of ion pair distances and orientations, Cation-Ow coordinations, and N_no3 - Hw coordinations.
         --> double check that this is actually what we are interested in
    """

    filename = path + f"ion_pair_distances.npy"
    if rerun == True or not os.path.exists(filename):
        cation_atoms = run.select_atoms("name C")
        n_atoms = run.select_atoms("name N_NO3")
        ion_pair_distances = np.empty(
            (len(run.trajectory[run_start:-1:skip]), len(cation_atoms), len(n_atoms)),
            dtype=float,
        )
        ion_pair_orientations = np.empty(
            (len(run.trajectory[run_start:-1:skip]), len(cation_atoms), len(n_atoms)),
            dtype=float,
        )
        cato_coordination = np.empty(
            (len(run.trajectory[run_start:-1:skip]), len(cation_atoms), len(n_atoms)),
            dtype=float,
        )
        nh_coordination = np.empty(
            (len(run.trajectory[run_start:-1:skip]), len(cation_atoms), len(n_atoms)),
            dtype=float,
        )

        for i, ts in enumerate(run.trajectory[run_start:-1:skip]):
            # NEED TO CHECK that this function gave the same results as below
            # ion_pair_distances[i,:] = MDAnalysis.analysis.distances.distance_array(na_atoms.positions, cl_atoms.positions, box=run.dimensions).flatten()

            # get vector separating each pair of ions using minimum image convention
            dists = np.empty((3, len(cation_atoms), len(n_atoms)))
            for j in range(3):
                dists[j, :, :] = cdist(
                    (cation_atoms.positions[:, j] % run.dimensions[j]).reshape(-1, 1),
                    (n_atoms.positions[:, j] % run.dimensions[j]).reshape(-1, 1),
                )
                dists[j, :, :] = np.where(
                    dists[j, :, :] > (run.dimensions[j] / 2)[..., None],
                    dists[j, :, :] - run.dimensions[j][..., None],
                    dists[j, :, :],
                )

            for j in range(len(cation_atoms)):
                cato_coord = run.select_atoms(
                    f"name O and around {cato_dist} index {cation_atoms[j].index}"
                ).n_atoms
                cato_coordination[i, j, :] = np.repeat(cato_coord, len(n_atoms))
                for k in range(len(n_atoms)):
                    if j == 0:
                        nh_coord = run.select_atoms(
                            f"name H and around {nh_dist} index {n_atoms[k].index}"
                        ).n_atoms
                        nh_coordination[i, :, k] = np.repeat(nh_coord, len(cation_atoms))
                    ion_pair_distances[i, j, k] = np.linalg.norm(dists[:, j, k])
                    ion_pair_orientations[i, j, k] = np.dot(
                        [0, 0, 1], dists[:, j, k]
                    ) / np.linalg.norm(dists[:, j, k])

        path = path + f"ion_pair_stats/"
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)

        np.save(path + "ion_pair_distances.npy", ion_pair_distances)
        np.save(path + "ion_pair_orientations.npy", ion_pair_orientations)
        np.save(path + "cato_coordination.npy", cato_coordination)
        np.save(path + "nh_coordination.npy", nh_coordination)

    else:
        ion_pair_distances = np.load(path + "ion_pair_distances.npy")
        ion_pair_orientations = np.load(path + "ion_pair_orientations.npy")
        cato_coordination = np.load(path + "cato_coordination.npy")
        nh_coordination = np.load(path + "clh_coordination.npy")

    return (
        ion_pair_distances,
        ion_pair_orientations,
        cato_coordination,
        nh_coordination,
    )

def get_pmf(path, run, cation, run_start, c_vdw=1.7, rerun=False):
    """
    Compute the potential of mean force (PMF) between ions in the system.

    Parameters:
        path (str): Path to the directory where output files will be saved.
        run (MDAnalysis.Universe): MDAnalysis Universe object representing the system trajectory.
        cation (float): Cation atom name.
        run_start (int): Index of the starting frame for analysis.
        rerun (bool, optional): If True, recompute the analysis even if the output file exists. Default is False.

    Returns:
        tuple: A tuple containing the PMF values and corresponding bin edges.
    """

    def get_bulk_volume_normalization(run, ion_pair_distances, c_vdw=1.7):
        """
        Calculate volume normalization factors for potential of mean force (PMF) calculation for a bulk electrolyte.
        
        Parameters:
            run (MDAnalysis.Universe): The MDAnalysis Universe object containing the simulation data.
            ion_pair_distances (numpy.ndarray): Array of ion pair distances (in nm).
            c_vdw (float, optional): van der Waals radius, default is 1.7 nm.
        
        Returns:
            numpy.ndarray: Volume normalization factors.
        """
        # Flatten the ion pair distances
        r = ion_pair_distances.flatten()

        # Calculate the volume of a spherical shell centered on each ion pair distance
        hist_vol_na = 4 * np.pi * r**2  # Surface area of the shell at distance r

        # Get the box dimensions (from the MDAnalysis Universe object)
        box = run.dimensions  # [a, b, c, alpha, beta, gamma]
        
        # compute the box volume from the dimensions of the box
        box_volume = box[0] * box[1] * box[2]

        # Normalize the histogram volume by the total box volume
        normalized_hist_vol_na = hist_vol_na / box_volume

        return normalized_hist_vol_na, box_volume
 
    def compute_pmf(path, run, ion_pair_distances, dr=0.08):
        """
        Compute the potential of mean force (PMF) between ions in the system.

        Parameters:
            run (MDAnalysis.Universe): MDAnalysis Universe object representing the system trajectory.
            ion_pair_distances (numpy.ndarray): Array of ion pair distances.
            dr (float, optional): Bin width for PMF calculation. Default is 0.08 Angstrom.

        Returns:
            tuple: A tuple containing the PMF values and corresponding bin edges.
        """
        weights, box_volume = get_bulk_volume_normalization(run, ion_pair_distances, c_vdw=1.7)
        dens, edges = np.histogram(
            ion_pair_distances.flatten(), weights=1 / weights, density=False
        )
        edges = edges[:-1]
        bulk_dens = len(ion_pair_distances.flatten()) / (box_volume)
        rdf = dens / dr / bulk_dens
        pmf_bins = edges
        pmf = -np.log(rdf)
        return pmf, pmf_bins

    filename = path + f"pmf.npy"
    if rerun == True or not os.path.exists(filename):
        ion_pair_distances, *_ = get_ion_pair_stats(path, run, run_start=run_start, rerun=rerun)
        pmf, pmf_bins = compute_pmf(path, run, ion_pair_distances, dr=0.08)
        np.save(path + "pmf.npy", pmf)
        np.save(path + "pmf_bins.npy", pmf_bins)
    else:
        pmf = np.load(path + "pmf.npy")
        pmf_bins = np.load(path + "pmf_bins.npy")
    return pmf, pmf_bins

def calc_neigh_corr(path, run, atoms, times, run_start=0, cutoff_dist=3.6, rerun=False):
    """
    Calculate the ion pairing residence time and autocorrelation function.

    Parameters:
        path (str): Path to the directory where output files will be saved.
        run (MDAnalysis.Universe): MDAnalysis Universe object representing the system trajectory.
        atoms (MDAnalysis.AtomGroup): AtomGroup representing the ions.
        times (np.ndarray): Array of time points corresponding to the trajectory.
        run_start (int, optional): Index of the starting frame for analysis. Default is 0.
        cutoff_dist (float, optional): Cutoff distance for defining ion pairs. Default is 3.6 Angstrom.
        rerun (bool, optional): If True, recompute the analysis even if the output file exists. Default is False.

    Returns:
        tuple: A tuple containing the ion pairing decay time and normalized autocorrelation function.
    """

    def autocorrFFT(x):
        """
        Calculate the autocorrelation function using the fast Fourier transform.

        Parameters:
            x (numpy.ndarray): Array containing data.

        Returns:
            numpy.ndarray: Autocorrelation function.
        """
        N = len(x)
        F = np.fft.fft(x, n=2 * N)
        PSD = F * F.conjugate()
        res = np.fft.ifft(PSD)
        res = (res[:N]).real
        n = N * np.ones(N) - np.arange(0, N)
        return res / n

    def calc_acf(A_values):
        """
        Calculate the autocorrelation function for a set of adjacency matrices.

        Parameters:
            A_values (dict): Dictionary containing adjacency matrices.

        Returns:
            list: List of autocorrelation functions.
        """
        acfs = []
        for atomid, neighbors in A_values.items():
            atomid = int(re.search(r"\d+", atomid).group())
            acfs.append(autocorrFFT(neighbors))
        return acfs

    def get_decay_time(times, data, decay_value=1.0 / np.e):
        """
        Find the time required for the autocorrelation function to decay to a specified value.

        Parameters:
            times (numpy.ndarray): Array of time points.
            data (numpy.ndarray): Array of autocorrelation function values.
            decay_value (float, optional): Value at which the autocorrelation function is considered decayed.
                                        Default is 1.0/np.e.

        Returns:
            float: Time at which the autocorrelation function decays to the specified value.
        """
        f = interpolate.interp1d(times[: len(data)], data[: len(times)])
        times_new = np.linspace(times[0], times[-1], 2000)

        def loss(time):
            return abs(f(time) - decay_value)

        return min(times_new, key=loss)

    def neighbors(run, run_start, atom, cutoff_dist):
        """
        Find neighboring atoms within a specified distance for each atom in the system.

        Parameters:
            run (MDAnalysis.Universe): MDAnalysis Universe object representing the system trajectory.
            run_start (int): Index of the starting frame for analysis.
            atom (MDAnalysis.Atom): Atom for which neighbors are to be found.
            cutoff_dist (float): Cutoff distance for defining neighbors.

        Returns:
            dict: Dictionary containing adjacency matrices for neighboring atoms.
        """
        A_values = {}
        time_count = 0
        for ts in run.trajectory[run_start::]:
            shell = run.select_atoms(
                "(name Cl and around "
                + str(cutoff_dist)
                + " resid "
                + str(atom.resid)
                + ")"
            )
            # for each atom in shell, create/add to dictionary (key = atom id, value = list of values for step function)
            for shell_atom in shell.atoms:
                if str(shell_atom.id) not in A_values:
                    A_values[str(shell_atom.id)] = np.zeros(
                        int((run.trajectory.n_frames - run_start) / 1)
                    )
                A_values[str(shell_atom.id)][time_count] = 1
            time_count += 1

        # account for species exiting then re-entering the shell
        lag_period = 4 # if an atom leaves for less then this many frames and re-enters, it is not counted as leaving
        for key in A_values:
            started = False
            ended = False
            for i in range(len(A_values[key])):
                if A_values[key][i] == 1:
                    started = True
                if started and np.sum(A_values[key][i:i+lag_period]) == 0:
                    ended = True
                if ended:
                    A_values[key][i] = 0

        return A_values

    filename = path + f"ion_pairing_acf.npy"
    if rerun == True or not os.path.exists(filename):
        # Average ACFs for all cations
        acf_all = []
        for atom in atoms[:]:
            adjacency_matrix = neighbors(run, run_start, atom, cutoff_dist)
            acfs = calc_acf(adjacency_matrix)
            [acf_all.append(acf) for acf in acfs]
        acf_avg = np.mean(acf_all, axis=0)
        acf_avg_norm = acf_avg / acf_avg[0]

        decay_time = get_decay_time(times, acf_avg_norm)

        np.save(path + f"ion_pairing_decay_time.npy", decay_time)
        np.save(path + f"ion_pairing_acf.npy", acf_avg_norm)

    else:
        decay_time = np.load(path + f"ion_pairing_decay_time.npy")
        acf_avg_norm = np.load(path + f"ion_pairing_acf.npy")

    return decay_time, acf_avg_norm


def average_replicates(data, run_axis=0):
    """
    Compute the average and standard deviation of replicated data.

    Parameters:
        data (numpy.ndarray): Array of data with replicates along the specified axis
        run_axis (int, optional): The axis along which to compute the mean and standard deviation.
                                  Default is 0.

    Returns:
        tuple: A tuple containing the average and standard deviation of the data.
    """

    data_avg = np.mean(data, axis=run_axis)
    data_err = np.std(data, axis=run_axis)

    return data_avg, data_err
