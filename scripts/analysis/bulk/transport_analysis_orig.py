""" 
Script to analyze transport properties of ions in bulk elyte systems using the Onsager Framework
Madleline A. Murphy, 2025 (adapted from code by Kara D. Fong)
"""

import numpy as np
import MDAnalysis as mda # type: ignore
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats # type: ignore

# Unit Conversions
A2cm = 1e-8
ps2s = 1e-12
e2c = 1.60217662e-19
base_dir = os.path.dirname(os.path.realpath(__file__))

# Constants
kb = 1.3806504e-23  # J/K, Boltzmann constant

# matplotlib settings
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
smallsize = 10
largesize = 10
plt.rcParams.update({"font.size": largesize})
plt.rc("xtick", labelsize=smallsize, direction="in")
plt.rc("ytick", labelsize=smallsize, direction="in")
plt.rc("axes", labelsize=largesize)
plt.rc("axes", titlesize=largesize, linewidth=0.7)
plt.rc("legend", fontsize=largesize)
plt.rc("lines", markersize=8, linewidth=2)
plt.rc("legend", frameon=False)
plt.rcParams["figure.figsize"] = [3.25, 3.25]
plt.rc("text", usetex=False)
matplotlib.rcParams["mathtext.default"] = "regular"
blues = [
    "#edf8b1",
    "#c7e9b4",
    "#7fcdbb",
    "#41b6c4",
    "#1d91c0",
    "#225ea8",
    "#253494",
    "#081d58",
]

##### Preliminary trajectory analysis #####

def generate_times(u, run_start, timestep):
    """ 
    generate an array of times in ps for a given trajectory

    paramters: 
        u: MDAnalysis universe object
        run_start: int, number of ps to skip before starting data collection
        timestep: int, number of ps between data collection
    returns:
        times: np.array, array of times in ps
    """
    # timestep is number of ps between data collection
    times = []
    current_step = 0
    for ts in u.trajectory[run_start:]:  # omit equilibration time
        times.append(current_step * timestep)
        current_step += 1
    times = np.array(times)
    return times


def define_atom_types(run): 
    """ 
    define atom types for cations, anions, and water (solvent) molecules
    
    parameters:
        run: MDAnalysis universe object
    returns: 
        c_ions: MDAnalysis atomgroup object, cations
        n_no3_ions: MDAnalysis atomgroup object, nitrate anions
        o_atoms: MDAnalysis atomgroup object, water (solvent) molecules
    """
    c_ions = run.select_atoms("name C")
    n_no3_ions = run.select_atoms("name N_NO")
    o_no3_ions = run.select_atoms("name O_NO")
    o_atoms = run.select_atoms("name O")
    h_atoms = run.select_atoms("name H")

    return c_ions, n_no3_ions, o_atoms

def create_position_arrays(path, u, anions, cations, times, run_start, in_memory=True, rerun=False):
    """ 
    create arrays of anion and cation positions relative to the center of mass of the system

    parameters: 
        path (str): path to save data
        u (MDAnalysis universe object): universe object
        anions (MDAnalysis atomgroup object): anions
        cations (MDAnalysis atomgroup object): cations
        times (np.array): array of times in ps
        run_start (int): number of ps to skip before starting data collection
        in_memory (bool): whether to load trajectory into memory
        rerun (bool): whether to rerun the analysis

    returns:
        anion_positions (np.array): array of anion positions
        cation_positions (np.array): array of cation positions
    """
    filename = base_dir + "/positions_anion.npy"
    if rerun == True or not os.path.exists(filename):
        if in_memory:
            u.transfer_to_memory() # Warning: this can be memory intensive
            traj = u.trajectory
            masses = u.atoms.masses
            atoms_all = u.select_atoms("not name X") # all atoms
            com = np.einsum('ijk,j', traj.coordinate_array[:,atoms_all.indices,:], masses[atoms_all.indices])/np.sum(masses[atoms_all.indices])
            anion_positions = traj.coordinate_array[:,anions.indices,:] - np.repeat(com[:,np.newaxis,:], len(anions.indices), axis=1)
            cation_positions = traj.coordinate_array[:,cations.indices,:] - np.repeat(com[:,np.newaxis,:], len(cations.indices), axis=1)

        else:
            time = 0
            anion_positions = np.zeros((len(times), len(anions), 3))
            cation_positions = np.zeros((len(times), len(cations), 3))
            for ts in u.trajectory[run_start:]:
                atoms_all = u.select_atoms("not name X") # all atoms
                anion_positions[time, :, :] = anions.positions - atoms_all.center_of_mass(
                    wrap=False
                )
                cation_positions[time, :, :] = cations.positions - atoms_all.center_of_mass(
                    wrap=False
                )
                time += 1

        np.save(filename, np.array(anion_positions))
        filename = path + "/positions_cation.npy"
        np.save(filename, np.array(cation_positions))
    else:
        anion_positions = np.load(filename, allow_pickle=True)
        filename = path + "/positions_cation.npy"
        cation_positions = np.load(filename, allow_pickle=True)
    return anion_positions, cation_positions

##### Computing MSDs (Mean-Squared Displacements) #####

def cross_corr(x, y):
    N = len(x)
    F1 = np.fft.fft(
        x, n=2 ** (N * 2 - 1).bit_length()
    )  # 2*N because of zero-padding, use next highest power of 2
    F2 = np.fft.fft(y, n=2 ** (N * 2 - 1).bit_length())
    PSD = F1 * F2.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real
    n = N * np.ones(N) - np.arange(0, N)  # divide res(m) by (N-m)
    return res / n

def autocorrFFT(x):
    """Calculates the position autocorrelation function using the fast Fourier transform."""
    #  https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft
    N = len(x)
    F = np.fft.fft(x, n=2 * N)  # 2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real 
    n = N * np.ones(N) - np.arange(0, N)  
    return res / n 

def msd_fft_1d(r):
    """Calculates mean square displacement of the array r using the fast Fourier transform."""
    N = len(r)
    D = np.square(r)
    D = np.append(D, 0)
    S2 = autocorrFFT(r)
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    return S1 - 2 * S2

def msd_variance_cross_1d(r1, r2, msd):

    # compute A1, recursive relation with D = r1^2 r2^2
    N = len(r1)
    D = np.square(r1)*np.square(r2)
    D = np.append(D, 0)
    Q = 2 * D.sum()
    A1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        A1[m] = Q / (N - m)

    # compute A2, cross correlation of r1^2r2 and r2
    A2 = cross_corr(np.square(r1)*r2, r2)

    # compute A3, cross correlation of r1^2 and r2^2
    A3 = cross_corr(np.square(r1), np.square(r2))

    # compute A4, cross correlation of (r1r2^2) and r1
    A4 = cross_corr(r1*np.square(r2), r1)

    # compute A5, cross correlation of r1r2 and r1r2
    A5 = cross_corr(r1*r2, r1*r2)

    # compute A6, cross correlation of r1 and r1r2^2
    A6 = cross_corr(r1, np.square(r2)*r1)

    # compute A7, cross correlation of r2^2 and r1^2
    A7 = cross_corr(np.square(r2), np.square(r1))

    # compute A8, cross correlation of r2 and r1^2r2
    A8 = cross_corr(r2, np.square(r1)*r2)    

    var_x = A1 - 2*A2 + A3 - 2*A4 +4*A5 - 2*A6 + A7 - 2*A8 - msd**2
    n_minus_m = N * np.ones(N) - np.arange(0, N)   # divide by (N-m)^2 (Var[E[X]] = Var[X]/n)

    return var_x/n_minus_m

def msd_variance_1d(r, msd):

    # compute A1, recursive relation with D = r^4
    N = len(r)
    D = r**4
    D = np.append(D, 0)
    Q = 2 * D.sum()
    A1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        A1[m] = Q / (N - m)

    # compute A2, autocorrelation of r^2
    A2 = cross_corr(r**2, r**2)

    # compute A3 and A4, cross correlations of r and r^3
    A3 = cross_corr(r, r**3)
    A4 = cross_corr(r**3, r)

    var_x = A1 + 6*A2 - 4*A3 - 4*A4 - msd**2
    n_minus_m = N * np.ones(N) - np.arange(0, N)   # divide by (N-m)^2 (Var[E[X]] = Var[X]/n)

    return var_x/n_minus_m

def calc_Lii_self(atom_positions, times):
    """Calculates self transport coefficient MSD for either cations or anions.
    Returns separate values for x, y, and z directions
    """
    msd = np.zeros([len(times), 3])
    msd_var = np.zeros([len(times), 3])
    n_atoms = np.shape(atom_positions)[1]
    for atom_num in range(n_atoms):
        r = atom_positions[:, atom_num, :]
        for i in range(3):  # x, y, z
            msd_temp = msd_fft_1d(np.array(r[:, i]))
            msd[:, i] += msd_temp
            msd_var[:, i] += msd_variance_1d(r[:, i], msd_temp)
    return msd, msd_var

def calc_Lij(r1, r2):
    """Calculates cation-anion correlations."""

    def msd_fft_cross_1d(r, k):
        """Calculates mean square displacement of the array r using the fast Fourier transform."""

        N = len(r)
        D = np.multiply(r, k)
        D = np.append(D, 0)
        S2 = cross_corr(r, k)
        S3 = cross_corr(k, r)
        Q = 2 * D.sum()
        S1 = np.zeros(N)
        for m in range(N):
            Q = Q - D[m - 1] - D[N - m]
            S1[m] = Q / (N - m)
        return S1 - S2 - S3

    r1_sum = np.sum(r1, axis=1)
    r2_sum = np.sum(r2, axis=1)
    msd = np.transpose(
        [msd_fft_cross_1d(r1_sum[:, i], r2_sum[:, i]) for i in range(3)]
    )
    msd_var = np.transpose(
        [msd_variance_cross_1d(r1_sum[:, i], r2_sum[:, i], msd[:, i]) for i in range(3)]
    )
    return msd, msd_var

def average_directions(msd, dirs='xyz'):
    if dirs=='xyz':
        msd_all = (
            msd[:, 0] + msd[:, 1] + msd[:, 2]
        ) / 3
    elif dirs=='xy':
        msd_all = (
            msd[:, 0] + msd[:, 1]
        ) / 2
    elif dirs=='z':
        msd_all = msd[:, 2]
    else:
        raise ValueError("dirs must be 'xyz', 'xy', or 'z' (others not yet implemented)")
    return msd_all

def get_lij_msds(path, u, run_start, samplingperiod, dirs='xyz', T=300, volume=None, rerun=True):
    filename = path + f"/transport_msd_data.npy"
    if os.path.isfile(filename) and not rerun:
        msds = np.load(filename, allow_pickle=True)
        times = np.load(path + f"/times.npy", allow_pickle=True)
        msd_variances =np.load(path + f"/lij_msd_variance.npy", allow_pickle=True)
    else:
        times = generate_times(u, run_start, timestep=samplingperiod)
        cations, anions, solvent = define_atom_types(u)
        if volume == None:
            volume = u.dimensions[0] * u.dimensions[1] * u.dimensions[2] 
        anion_positions, cation_positions = create_position_arrays(
            path, u, anions, cations, times, run_start, rerun=rerun
        )

        msd_self_cation, var_self_cation = calc_Lii_self(cation_positions, times)
        msd_self_anion, var_self_anion = calc_Lii_self(anion_positions, times)
        msd_total_cation, var_total_cation = calc_Lij(cation_positions, cation_positions)
        msd_total_anion, var_total_anion = calc_Lij(anion_positions, anion_positions)
        msd_distinct_catAn, var_distinct_catAn = calc_Lij(cation_positions, anion_positions)

        msds = np.array(
            [
                msd_self_cation,
                msd_self_anion,
                msd_total_cation,
                msd_total_anion,
                msd_distinct_catAn,
            ]
        )/ 2.0 / kb / T / volume

        msd_variances = np.array(
            [
                var_self_cation,
                var_self_anion,
                var_total_cation,
                var_total_anion,
                var_distinct_catAn,
            ]
        ) * (1/2.0 / kb / T / volume)**2

        msds = np.array([average_directions(msd, dirs) for msd in msds])
        msd_variances = np.array([average_directions(var, dirs) for var in msd_variances])

        np.save(path + f"/transport_msd_data.npy", msds)
        np.save(path + f"/lij_msd_variance.npy", msd_variances)
        np.save(path + f"/times.npy", times)
    return msds, msd_variances, times

def get_cond_msd(path, u, run_start, samplingperiod, dirs='xyz', T=300, volume=None, rerun=False):
    filename = path + f"/cond_msd_var.npy"
    if os.path.isfile(filename) and not rerun:
        msd = np.load(path + f'/cond_msd.npy')
        msd_var = np.load(path + f'/cond_msd_var.npy')
        times = np.load(path + f"/times.npy")
    else:
        if volume == None:
            volume = u.dimensions[0]*u.dimensions[1]*u.dimensions[2]
        times = generate_times(u, run_start, timestep=samplingperiod)
        cations, anions, solvent = define_atom_types(u)

        anion_positions, cation_positions = create_position_arrays(
                path, u, anions, cations, times, run_start, rerun=rerun
            )

        ion_positions = -anion_positions + cation_positions
        r = np.sum(ion_positions,axis=1)
        msd = np.array([msd_fft_1d(np.array(r[:,i])) for i in range(3)])
        msd_var = np.array([msd_variance_1d(r[:,i], msd[i]) for i in range(3)])
        msd = average_directions(np.transpose(msd), dirs)
        msd_var = average_directions(np.transpose(msd_var), dirs)
        msd = msd/ 2.0 / kb / T / volume
        msd_var = msd_var*(1/ 2.0 / kb / T / volume)**2
        np.save(path + f'/cond_msd.npy', msd)
        np.save(path + f'/cond_msd_var.npy', msd_var)
        np.save(path + f'/times.npy', times)
    return msd, msd_var, times

##### Fitting MSDs #####

def fit_data(msd, times, start, end, weighted=False, msd_var=[], units="mS/cm"):
    if units == "mS/cm":
        convert = e2c * e2c / ps2s / A2cm * 1000
    elif units == "cm2/s":
        convert = A2cm * A2cm / ps2s
    else:
        raise ValueError("Units must be 'mS/cm' or 'cm2/s'")
    if weighted == False:
        # slope_avg, _, _, _, _ = stats.linregress(times[start:end], msd[start:end])
        # lij = slope_avg * convert 
        lij = 0.0 # placeholder
    else:
        if msd_var.size == 0:
            raise ValueError("For weighted linear regression, you must supply MSD variance")
        line = np.polynomial.polynomial.polyfit(times[start:end], msd[start:end], 1, w=1/msd_var[start:end])
        lij = (line[1] * convert) 
    msd_slope = np.gradient(np.log(np.abs(msd[start:end])), np.log(times[start:end]))
    beta = np.nanmean(np.array(msd_slope))

    return lij, beta

def fit_all_msds(msds, times, start=100, end=1000, weighted=True, msd_var=[]):
    lij = np.empty([len(msds)], dtype=float)
    beta = np.empty([len(msds)], dtype=float)
    for l in range(len(msds)):
        if weighted: var = msd_var[l]
        else: var = []
        if l ==4: 
            lij[l], beta[l] = fit_data(msds[l], times, start, end, weighted, var)
        else:
            lij[l], beta[l] = fit_data(msds[l], times, start, end, weighted, var)
    # lij[-2] = lij[2] - lij[0]  # cation distinct
    # lij[-1] = lij[3] - lij[1]  # anion distinct

    return lij, beta

##### Plotting MSDs #####

def make_one_msd_plot(times, msd, start, end, color="k", label="", save=False):
    plt.plot(times, np.abs(msd), color=color, label=label)
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time (ps)")
    plt.ylabel("MSD")
    plt.ylim(min(np.abs(msd[1:])) * 0.9, max(np.abs(msd)) * 1.1)

    i = int(len(msd) / 5)
    slope_guess = (msd[i] - msd[5]) / (times[i] - times[5])
    plt.plot(times[start:end], times[start:end] * slope_guess * 2, "k--", alpha=0.5)
    plt.tight_layout()
    if save:
        os.makedirs(base_dir + "/test/figures/msds", exist_ok=True)
        plt.savefig(base_dir + f"/test/igures/msds/{label}.png", format="png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

##### Miscellaneous #####

def compute_water_msd(path, u, run_start, samplingperiod, dirs='xyz', T=300, volume=None, rerun=False):
    filename = f"{path}/water_msd_var.npy"
    if os.path.isfile(filename) and not rerun:
        msd = np.load(f'{path}/water_msd.npy')
        msd_var = np.load(f'{path}/water_msd_var.npy')
        times = np.load(f"{path}/times.npy")
    else:
        if volume == None:
            volume = u.dimensions[0]*u.dimensions[1]*u.dimensions[2]
        times = generate_times(u, run_start, timestep=samplingperiod)
        _, _, waters = define_atom_types(u)

        u.transfer_to_memory() # Warning: this can be memory intensive
        traj = u.trajectory
        masses = u.atoms.masses
        not_carbon = u.select_atoms("not name Ca") # in confined systems, don't include solid in center of mass calculation
        com = np.einsum('ijk,j', traj.coordinate_array[:,not_carbon.indices,:], masses[not_carbon.indices])/np.sum(masses[not_carbon.indices])
        water_positions = traj.coordinate_array[:,waters.indices,:] - np.repeat(com[:,np.newaxis,:], len(waters.indices), axis=1)
        msd, msd_var = calc_Lii_self(water_positions, times)
        msd = average_directions(msd, dirs)
        msd_var = average_directions(msd_var, dirs)
        msd = msd / 2.0 / len(waters)
        msd_var = msd_var * (1 / 2.0 / len(waters))**2

        np.save(f'{path}/water_msd.npy', msd)
        np.save(f'{path}/water_msd_var.npy', msd_var)
        np.save(f'{path}/times.npy', times)

    return msd, msd_var, times

