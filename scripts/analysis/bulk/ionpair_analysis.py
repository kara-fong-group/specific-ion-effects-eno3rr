import numpy as np
import MDAnalysis as mda
import os
import re
from scipy import interpolate
from scipy.spatial.distance import cdist
from scipy import stats

### --- Ion Pair Analysis --- ###

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

def dists(run):
    """
    Generate a array with the distances between ALL ions in the system.
    Parameters:
        run (MDAnalysis Universe): MDAnalysis Universe object.

    returns:
        dists(numpy.ndarray): Array of distances between ions.
    """
    # Select cations and anions
    cat_atoms = run.select_atoms("name C")
    n_atoms = run.select_atoms("name N_NO")

    # initialize the array
    dists = np.empty((3, len(cat_atoms), len(n_atoms)))

    for i in range(3):
        dists[i,:,:] = cdist((cat_atoms.positions[:,i] % run.dimensions[i]).reshape[-1,1],
                              (n_atoms.positions[:,i] % run.dimensions[i]).reshape[-1,1],)
        
        # check that the distance is the minimum distance using the minimum image convention
        dists[i,:,:] = np.where(
                    dists[i,:,:] > (run.dimensions[i] / 2)[..., None],
                    dists[i,:,:] - run.dimensions[i][..., None],
                    dists[i,:,:],
                )
    return dists

def compute_ion_pair_distances(path, run_start=0, skip=1, rerun=False):
    """
    Compute the distances of each ion pairs in the system at each trajectory point.
    
    Parameters:
        path(str): Path to the directory containing data files.

    returns:
        ion_pair_distances(numpy.ndarray): Array of the distances between ion pairs.
    """
    filename = path + f"ion_pair_dists.npy"
    if rerun == True or not os.path.exists(filename):
        run = create_mda(
            path,
            data_file="../run_initial/end_equil.data",
            dcd_file="traj_unwrapped_*.dcd",
            cat_dcd=True,
        )

        # Select cations and anions
        cat_atoms = run.select_atoms("name C")
        n_atoms = run.select_atoms("name N_NO") 

        num_timepoints = len(run.trajectory[run_start:-1:skip])
        ion_pair_distances = np.empty((num_timepoints, len(cat_atoms), len(n_atoms)))

        for i, ts in enumerate(run.trajectory[run_start:-1:skip]):
            dists = dists(run)
            for j in range(len(cat_atoms)):
                for k in range(len(n_atoms)):
                    ion_pair_distances[i, j, k] = dists[:, j, k]

        np.save(filename, ion_pair_distances)

    else:
        ion_pair_distances = np.load(filename)

    return ion_pair_distances

def calc_neigh_corr(path, run, atoms, times, run_start=0, cutoff_dist=3.6, rerun=False):
    """
    Calculate the ion pair residence times and autocorrelation functions.
    
    Parameters:
        path(str): Path to the directory containing data files.
        run (MDAnalysis Universe): MDAnalysis Universe object.
        atoms (MDAnalysis.Atom): Atom for which the correlation of neighbors are to be found. (the ions type of reference)
        times (numpy.ndarray): Array of times for each trajectory point.
        run_start (int): Index of the starting frame for the analysis. Default is 0.
        cutoff_dist (float): Cutoff distance for contact ion pairs determined from pmf analysis. Default is 3.6.
        rerun (bool, optional): Whether to rerun the analysis even if results exist. Defaults to False.

    returns:
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
            shell = run.select_atoms("(name N_NO and around "
                                     + str(cutoff_dist)
                                     + " resid "
                                     + str(atom.resid)
                                     + ")")
            
            # for each atom in the shell, add it to the dictionary
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

        for atom in atoms[:]: # for each cation
            adjacency_matrix = neighbors(run, run_start, atom, cutoff_dist)
            acfs = calc_acf(adjacency_matrix)
            [acf_all.append(acf) for acf in acfs]
        
        acf_avg = np.mean(acf_all, axis=0)
        acf_avg_norm = acf_avg / acf_avg[0]

        decay_time = get_decay_time(times, acf_avg_norm)

        np.save(filename, acf_avg_norm)
        np.save(path + "ion_pairing_decay_time.npy", decay_time)

    else:
        acf_avg = np.load(filename)
        decay_time = np.load(path + "ion_pairing_decay_time.npy")

    return decay_time, acf_avg

def average_replicates(data, run_axis=0):
    """
    Compute the average and standard deviation of a set of replicates.

    Parameters:
        data (numpy.ndarray): Array of ion pair residence times.
        run_axis (int, optional): Axis along which to average the data. Default is 0.

    Returns:
        tuple: A tuple containing the average residence time and standard error.
    """
    avg = np.mean(data, axis=run_axis)
    err = np.std(data, axis=run_axis)

    return avg, err