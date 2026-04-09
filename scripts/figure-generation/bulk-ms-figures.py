# ================================
# Code to create figures for the manuscript
# Madeline Murphy -- Oct 2025
# ================================ 

"""
The following code creates the transport subfigures (5a, 5b, 5c).
The code is organized by figure number.

files needed to run (for each cation and concentration):
total-transport/lij.npy
total-transport/lij_sol.npy

"""

import os
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

# color map
cations = ['Cs', 'K', 'Na', 'Li']
cation_labels = [r'Cs$^+$', r'K$^+$', r'Na$^+$', r'Li$^+$']
markers = ['o', 's', '^', 'D']
colors = ["#1c4f7e", "#E05656",  "#4db4e8", "#8d0a0e"]
colors2 = ["#A27514", "#049ea9", "#e6c352"]

# Figure 5a: conductivity
def conductivity(avg, std, fig_folder):
    conc_vals = [1.0, 0.5, 0.1, 0.01]

    fig, ax = plt.subplots(1, 1, figsize=[3.25, 3.4])
    for k, cation in enumerate(cations):
        ax.plot(conc_vals, avg[k,:], marker=markers[k], color=colors[k], label=f'{cation}', linestyle='--')
        ax.errorbar(conc_vals, avg[k,:], yerr=std[k,:], fmt=markers[k], color=colors[k], capsize=5)
    ax.set_xlabel('Concentration (mol/kg)', fontsize=11)
    ax.set_ylabel(r'Conductivity (mS/cm)', fontsize=11)
    ax.set_ylim(-5, 110)
    ax.set_yticks(np.arange(0, 120, 20))
    ax.set_yticklabels([f'{y:.0f}' for y in np.arange(0, 120, 20)], fontsize=10)

    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_xticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.2)], fontsize=10)
    # ax.legend(fontsize=10, frameon=False, ncols=2, loc='lower right')
    ax.grid()
    plt.tight_layout()
    fig.savefig(f'{fig_folder}conductivity.tiff', dpi=300)
    return None

# Figure 5b: self diffusion coefficients
def self_diffusion(avg, std, fig_folder):
    cation_xvals = [4, 3, 2, 1]

    fig, ax = plt.subplots(1, 1, figsize=[3.25,3.25])

    ax.scatter(cation_xvals, avg[:,0,0]*1e4, marker='o', label='Cation', color=colors2[0])
    ax.scatter(cation_xvals, avg[:,0,1]*1e4, marker='^', label='Nitrate', color=colors2[1])

    ax.plot(cation_xvals, avg[:,0,0]*1e4, color=colors2[0], linestyle='-')
    ax.plot(cation_xvals, avg[:,0,1]*1e4, color=colors2[1], linestyle='-')

    for k, cation in enumerate(cations):
        ax.errorbar(cation_xvals[k], avg[k,0,0]*1e4, yerr=std[k,0,0]*1e4, fmt='o', capsize=5, color=colors2[0])
        ax.errorbar(cation_xvals[k], avg[k,0,1]*1e4, yerr=std[k,0,1]*1e4, fmt='^', capsize=5, color=colors2[1])

    ax.set_xticks(cation_xvals)
    ax.set_xticklabels(cation_labels, fontsize=11)
    ax.set_ylabel(r'Diffusion Coefficient($10^{-5}$ cm$^2$/s)', fontsize=11)
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(1.0e-5, 2.0e-5)
    ax.set_yticks(np.arange(1.0e-5, 2.1e-5, 0.2e-5))
    ax.set_yticklabels([f'{y*1e5:.1f}' for y in np.arange(1.0e-5, 2.1e-5, 0.2e-5)], fontsize=10)

    ax.grid()
    # ax.legend(fontsize=10, frameon=False, loc='lower right')
    plt.tight_layout()
    fig.savefig(f'{fig_folder}/self-diffusion-finite-size.tiff', dpi=300)
    return None

# Figure 5c: transference number and electrolyte diffusivity
def transference_and_diffusivity(avg_tn, std_tn, avg_D_el, std_D_el, fig_folder):
    fig, ax1 = plt.subplots(1, 1, figsize=[4,3.25])
    ax2 = ax1.twinx()
    colors3 = ["#4E3706", "#2282c2", "#47453e"]

    cation_labels = [r'Cs$^+$', r'K$^+$', r'Na$^+$', r'Li$^+$']
    cation_xvals = [4, 3, 2, 1]

    ax1.scatter(cation_xvals, avg_tn[:,0], marker='o', label='Nitrate TN', color=colors3[1])
    ax1.plot(cation_xvals, avg_tn[:,0], color=colors3[1], linestyle='-')
    ax1.errorbar(cation_xvals, avg_tn[:,0], yerr=std_tn[:,0], fmt='o', capsize=5, color=colors3[1])
    ax1.set_xticks(cation_xvals)
    ax1.set_xticklabels(cation_labels, fontsize=11)
    ax1.set_xlim(0.5, 4.5)
    ax1.set_ylabel('Nitrate Transference Number', fontsize=11)
    ax1.set_ylim(0.3, 0.7)
    ax1.set_yticks(np.arange(0.3, 0.8, 0.1))
    ax1.set_yticklabels([f'{y:.1f}' for y in np.arange(0.3, 0.8, 0.1)], fontsize=10, color=colors3[1])
    ax1.set_ylabel(r'NO$_3^-$ Transference Number (t$_-$)', fontsize=11, color=colors3[1])

    ax2.scatter(cation_xvals, avg_D_el[:,0]*1e4, marker='^', label='Electrolyte Diffusion Coeff.', color=colors3[0])
    ax2.plot(cation_xvals, avg_D_el[:,0]*1e4, color=colors3[0], linestyle='-')
    ax2.errorbar(cation_xvals, avg_D_el[:,0]*1e4, yerr=std_D_el[:,0]*1e4, fmt='^', capsize=5, color=colors3[0])
    ax2.set_ylim(4e-5, 1.2e-4)
    ax2.set_yticks(np.arange(4e-5, 1.3e-4, 2e-5))
    ax2.set_yticklabels([f'{y*1e4:.1f}' for y in np.arange(4e-5, 1.3e-4, 2e-5)], fontsize=10, color=colors3[0])
    ax2.set_ylabel(r'D$_{\mathrm{el}}$ ($10^{-5}$ cm$^2$/s)', fontsize=11, color=colors3[0])
    ax1.grid()
    plt.tight_layout()

    fig.savefig(f'{fig_folder}/tn-diff-combined.tiff', dpi=300)
    return None

# function to read in transport data
def read_transport_data(data_path):
    # code to read in the raw transport data
    concentrations = ['1M', '0.5M', '0.1M', '0.01M']
    conc_vals = [1.0, 0.5, 0.1, 0.01] 
    num_ion = [6, 3, 1, 1]
    num_water = [332, 332, 560, 5600]
    cations = ['Cs', 'K', 'Na', 'Li']
    Mw = {'Cs': 132.905, 'K': 39.0983, 'Na': 22.9897, 'Li': 6.94}
    lij_names = ['Cation self', 'Anion self', 'Cation total', 'Anion total', 'Cation-Anion distinct']

    avg_lij = np.empty((len(cations), len(concentrations), len(lij_names)))
    std_lij = np.empty((len(cations), len(concentrations), len(lij_names)))

    avg_cond = np.empty((len(cations), len(concentrations)))
    std_cond = np.empty((len(cations), len(concentrations)))

    avg_D_el = np.empty((len(cations), len(concentrations)))
    std_D_el = np.empty((len(cations), len(concentrations)))

    avg_diff = np.empty((len(cations), len(concentrations), 2))
    std_diff = np.empty((len(cations), len(concentrations), 2))

    avg_tn = np.empty((len(cations), len(concentrations)))
    std_tn = np.empty((len(cations), len(concentrations)))

    diff_intercepts = np.empty((len(cations), len(concentrations), 2))

    def cond(lij):
        return lij[2,:] + lij[3,:] - 2 * lij[4,:]

    def diff_el(lij, conc, M_cat, n_ion, n_wat):
        M_water = 18.01528  # g/mol
        M_nit = 62.0049  # g/mol
        c = conc * 1e3  # mol/L to mol/m^3 conversion
        c_wat = c * (n_wat / n_ion)  # mol/m^3
        R = 8.314  # J/(mol*K)
        T = 298.15  # K
        F = 96485  # C/mol
        
        denom = cond(lij) 
        numer = (lij[2] * lij[3] - lij[4]**2) * (R * T / c)
        correction = (1 + (c * (M_cat + M_nit) / (c_wat * M_water)))

        return (numer / denom) * correction / F**2  # m^2/s

    def self_diffusion_coeff(lij, conc):
        c = conc * 1e3  # mol/L to mol/m^3 conversion
        R = 8.314  # J/(mol*K)
        T = 298.15  # K
        return (lij * 0.1 / 96485**2) * (R * T /c) # m^2/s

    def transference_number(lij):
        return (lij[3,:] - lij[4,:]) / (lij[2,:] + lij[3,:] - 2*lij[4,:])

    for k, cation in enumerate(cations):
        for j, conc in enumerate(concentrations):

            lij_com = np.load(f'{data_path}/{cation}/bulk/{conc}/total-transport/lij.npy')
            lij_sol = np.load(f'{data_path}/{cation}/bulk/{conc}/total-transport/lij_sol.npy')

            avg_lij[k,j,:] = np.mean(lij_sol, axis=1)
            std_lij[k,j,:] = np.std(lij_sol, axis=1)

            avg_cond[k,j] = np.mean(cond(lij_sol))
            std_cond[k,j] = np.std(cond(lij_sol))

            avg_diff[k,j,0] = np.mean(self_diffusion_coeff(lij_com[0,:], conc_vals[j]))
            avg_diff[k,j,1] = np.mean(self_diffusion_coeff(lij_com[1,:], conc_vals[j]))
            std_diff[k,j,0] = np.std(self_diffusion_coeff(lij_com[0,:], conc_vals[j]))
            std_diff[k,j,1] = np.std(self_diffusion_coeff(lij_com[1,:], conc_vals[j]))

            avg_tn[k,j] = np.mean(transference_number(lij_sol))
            std_tn[k,j] = np.std(transference_number(lij_sol))

            avg_D_el[k,j] = np.mean(diff_el(lij_sol, conc_vals[j], Mw[cation], num_ion[j], num_water[j]))
            std_D_el[k,j] = np.std(diff_el(lij_sol, conc_vals[j], Mw[cation], num_ion[j], num_water[j]))

            # load inifinite size limit data
            path = f'{data_path}/finite-size/'
            diff_intercepts[:,j,:] = np.load(f'{path}{conc}/infinite-size-limit-diffusion.npy')
    return avg_lij, std_lij, avg_cond, std_cond, avg_D_el, std_D_el, avg_diff, std_diff, avg_tn, std_tn, diff_intercepts

def make_all_figures():
    transport_data_path = '../../data'
    compressed_data_path = '../../data-ms-figs/bulk'
    fig_folder = '../../figures/ms/'
    # make sure the figure folder exists
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    # save needed data for the figures
    avg_cond = np.load(f'{compressed_data_path}/avg_cond.npy', allow_pickle=True)
    std_cond = np.load(f'{compressed_data_path}/std_cond.npy', allow_pickle=True)
    avg_D_el = np.load(f'{compressed_data_path}/avg_D_el.npy', allow_pickle=True)
    std_D_el = np.load(f'{compressed_data_path}/std_D_el.npy', allow_pickle=True)
    diff_intercepts = np.load(f'{compressed_data_path}/avg_diff.npy', allow_pickle=True)
    std_diff = np.load(f'{compressed_data_path}/std_diff.npy', allow_pickle=True)
    avg_tn = np.load(f'{compressed_data_path}/avg_tn.npy', allow_pickle=True)
    std_tn = np.load(f'{compressed_data_path}/std_tn.npy', allow_pickle=True)

    conductivity(avg_cond, std_cond, fig_folder)
    self_diffusion(diff_intercepts, std_diff, fig_folder)
    transference_and_diffusivity(avg_tn, std_tn, avg_D_el, std_D_el, fig_folder)

    return None

if __name__ == "__main__":
    make_all_figures()