# ================================
# Code to create figures for the supporting information (Bulk properties)
# Madeline Murphy -- Oct 2025
# ================================ 

"""
The following code creates the bulk transport property figures for the supporting information.
Includes bulk validation, Onsager flux figures, finite size analysis, and reference velocity analysis. 
The code is organized by figure number. (S6, S7, S8, S9, S10, S11).

files needed to run (for each cation and concentration):
total-transport/lij.npy
total-transport/lij_sol.npy
total-transport/beta.npy

finite-size/box{size}/total-transport/lij.npy (for each box size)
finite-size/box{size}/box_sizes.npy (for each box size)

"""

import os
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from scipy.stats import linregress

# color map
cations = ['Cs', 'K', 'Na', 'Li']
cation_labels = [r'Cs$^+$', r'K$^+$', r'Na$^+$', r'Li$^+$']
markers = ['o', 's', '^', 'D']
colors = ["#1c4f7e", "#E05656",  "#4db4e8", "#8d0a0e"]
colors2 = ["#A27514", "#049ea9", "#e6c352"]

# Figure S6: self diffusion coefficients
def diffusion(conc_vals, avg_D_el_all, std_D_el_all, folder, colors=colors):
    # plot electrolyte diffusion coefficient
    fig, ax = plt.subplots(1, 1, figsize=[3.25,3.25])
    for k, cation in enumerate(cations):
        ax.plot(conc_vals, avg_D_el_all[k,:]*1e4, marker=markers[k], color=colors[k], label=f'{cation}', linestyle='--')
        ax.errorbar(conc_vals, avg_D_el_all[k,:]*1e4, yerr=std_D_el_all[k,:]*1e4, fmt=markers[k], color=colors[k], capsize=5)
    ax.set_xlabel('Concentration (mol/kg)', fontsize=11)
    ax.set_ylabel(r'Diffusion Coefficient (cm$^2$/s)', fontsize=11)
    ax.set_ylim(4e-5, 9.8e-5)
    ax.grid()
    plt.tight_layout()
    # ax.legend(fontsize=10, frameon=True, ncols=2)
    fig.savefig(f'{folder}electrolyte_diff.tiff', dpi=300)
    return None

def transference(conc_vals, avg_tn_all, std_tn_all, folder, colors=colors):
    # plot the transference number of nitrate
    fig, ax = plt.subplots(1, 1, figsize=[3.25, 3.25])

    for k, cation in enumerate(cations):
        ax.scatter(conc_vals, avg_tn_all[k,:], marker=markers[k], color=colors[k], label=f'{cation}')
        ax.plot(conc_vals, avg_tn_all[k,:], color=colors[k], linestyle='--')
        ax.errorbar(conc_vals, avg_tn_all[k,:], yerr=std_tn_all[k,:], fmt=markers[k], color=colors[k], capsize=5)

    ax.set_xlabel('Concentration (mol/kg)', fontsize=11)
    ax.set_ylabel('Nitrate Transference Number', fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.3, 0.7)
    # ax.legend(fontsize=10, frameon=True, ncols=2, loc='lower right')
    ax.grid()
    plt.tight_layout()
    fig.savefig(f'{folder}transference-number.tiff', dpi=300)
    return None

def beta_fs_effects(path, fig_folder, colors=colors):
    fig, ax = plt.subplots(5,4, figsize=(7,7), sharex=True)
    concentration = ['1M', '0.5M', '0.1M', '0.01M']
    lij_names = [r'M$^+$ - self', r'NO$_3^-$ - self', r'M$^+$ - total', r'NO$_3^-$ - total', r'Cation-Anion']

    for j, cation in enumerate(['Cs', 'K', 'Na', 'Li']):
        for k, conc in enumerate(concentration):
            beta = np.load(f'{path}{cation}/bulk/{conc}/total-transport/beta.npy')
            # lij = np.load(f'{path}{cation}/{conc}/total-transport/lij.npy')
            # print("size of beta:", beta.shape)
            # print(lij)
            length = beta.shape[1]

            for i in range(5):
                # plot a black horizontal line at y=1
                ax[i,k].axhline(y=1, color='k', linestyle='--', linewidth=1)
                ax[i,k].scatter(j, np.mean(beta[i]), color=colors[j])
                ax[i,k].errorbar(j, np.mean(beta[i]), yerr=np.std(beta[i]), color=colors[j], capsize=5)
                # if i == 0:
                    # ax[i,k].set_title(f"box {size}", fontsize=11, y=0.87, x=0.5)
                # ax[k,i].set_title(f"box {size}, {lij_names[i]}")
                ax[i,k].grid()
                ax[i,k].set_xticks(range(4))
                ax[i,k].set_xticklabels([r'Cs$^+$', r'K$^+$', r'Na$^+$', r'Li$^+$'], fontsize=11)
                if k == 0:
                    ax[i,k].set_ylabel(r'$\beta$, ' + f'{lij_names[i]}', fontsize=11)

                if j == 0:
                    ax[i,k].grid()

                if i != 4:
                    ax[i,k].set_ylim(0.9, 1.1)
                else:
                    ax[i,k].set_ylim(-0.5, 3)
    plt.tight_layout()
    fig.savefig(f'{fig_folder}transport-beta.tiff', dpi=300)
    return None

def finite_size_effects(concentrations, box_sizes_all, avg_diff_all, std_diff_all, path):
    # fit the diffusion coefficients to a line
    conc = '1M'
    j = concentrations.index(conc)
    # print(f'Fitting diffusion coefficients for {conc}, j={j}')
    colors = ['#1f77b4', '#ff7f0e']

    def fit_diffusion_data(box_sizes, diff_values):
        slope, intercept, r_value, p_value, std_err = linregress(box_sizes, diff_values)
        return slope, intercept, r_value**2
    infinite_size_limit = np.zeros((len(cations), 2))
    avg_slopes = np.zeros((len(cations), 2))

    # plot the self diffusion coefficients
    fig, axes = plt.subplots(2, 2, figsize=[7,7])
    ax = axes.flatten()
    for k, cation in enumerate(cations):
        # print(f'Plotting self diffusion coefficients for {cation}')
        ax[k].scatter(1/box_sizes_all[k,j,:], avg_diff_all[k,j,:,0], color=colors[0])
        ax[k].errorbar(1/box_sizes_all[k,j,:], avg_diff_all[k,j,:,0], yerr=std_diff_all[k,j,:,0], fmt='o', capsize=5, color=colors[0])
        slope, intercept, r_squared = fit_diffusion_data(1/box_sizes_all[k,j,:], avg_diff_all[k,j,:,0])
        infinite_size_limit[k,0] = intercept
        avg_slopes[k,0] = slope
        # print(f'Slope: {slope}, Intercept: {intercept} for Cation self diffusion')
        ax[k].plot(1/box_sizes_all[k,j,:], slope * (1/box_sizes_all[k,j,:]) + intercept, linestyle='--', color=colors[0], label=f'Cation, $r^2$={r_squared:.2f}')

        ax[k].scatter(1/box_sizes_all[k,j,:], avg_diff_all[k,j,:,1], color=colors[1])
        ax[k].errorbar(1/box_sizes_all[k,j,:], avg_diff_all[k,j,:,1], yerr=std_diff_all[k,j,:,1], fmt='o', capsize=5, color=colors[1])
        slope, intercept, r_squared = fit_diffusion_data(1/box_sizes_all[k,j,:], avg_diff_all[k,j,:,1])
        ax[k].plot(1/box_sizes_all[k,j,:], slope * (1/box_sizes_all[k,j,:]) + intercept, linestyle='--', color=colors[1], label=f'Anion, $r^2$={r_squared:.2f}')
        # print(f'Slope: {slope}, Intercept: {intercept} for Anion self diffusion')
        infinite_size_limit[k,1] = intercept
        avg_slopes[k,1] = slope

        ax[k].set_xlabel(r'Box Size ($\AA^{-1}$)', fontsize=11)
        ax[0].set_ylabel(r'Self Diffusion Coefficient ($m^2/s$)', fontsize=11)
        ax[k].set_title('cation: {cation}'.format(cation=cation) + r'$^+$', fontsize=10, y=0.87, x=0.8)
        ax[k].set_xlim(0.03, 0.05)
        ax[k].set_ylim(0.8e-9, 2.0e-9)
        ax[k].grid()
        if k == 0 or k == 1:
            ax[k].legend(fontsize=10, frameon=True, loc='lower left')
        else:
            ax[k].legend(fontsize=10, frameon=True, loc='center right', bbox_to_anchor=(1.0, 0.7))
    plt.tight_layout()

    fig.savefig(f'{path}self-diffusion-fits-{conc}.tiff', dpi=300)
    return None

def conductivity_com_vs_sol(conc_vals, avg_cond_all, std_cond_all, com_avg_cond_all, com_std_cond_all, path):
    # plot conductivity comparison
    fig, ax = plt.subplots(1, 1, figsize=[3.25,3.25])
    for k, cation in enumerate(cations):
        ax.plot(conc_vals, avg_cond_all[k,:], marker=markers[k], color=colors[k], label=f'{cation} - COM', linestyle='-', alpha=0.6, linewidth=2)
        ax.errorbar(conc_vals, avg_cond_all[k,:], yerr=std_cond_all[k,:], fmt=markers[k], color=colors[k], capsize=5)
        ax.plot(conc_vals, com_avg_cond_all[k,:], marker=markers[k], color=colors[k], label=f'{cation} - SOL', linestyle=':')
        ax.errorbar(conc_vals, com_avg_cond_all[k,:], yerr=com_std_cond_all[k,:], fmt=markers[k], color=colors[k], capsize=5)

    ax.set_xlabel('Concentration (mol/kg)', fontsize=11)
    ax.set_ylabel(r'Conductivity (mS/cm)', fontsize=11)
    ax.set_ylim(-5, 110)
    ax.set_yticks(np.arange(0, 120, 20))
    ax.set_yticklabels([f'{y:.0f}' for y in np.arange(0, 120, 20)], fontsize=10)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_xticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.2)], fontsize=10)
    ax.grid()
    plt.tight_layout()
    fig.savefig(f'{path}conductivity-com-vs-sol.tiff', dpi=300)
    return None

def l_plus_minus(conc_vals, avg_lij_all, std_lij_all, path):
    # plot conductivity comparison
    fig, ax = plt.subplots(1, 1, figsize=[4.5,4.7])
    for k, cation in enumerate(cations):
        ax.plot(conc_vals, avg_lij_all[k,:], marker=markers[k], color=colors[k], label=f'{cation} - COM', linestyle='-', alpha=0.6, linewidth=2)
        ax.errorbar(conc_vals, avg_lij_all[k,:], yerr=std_lij_all[k,:], fmt=markers[k], color=colors[k], capsize=5)

    ax.set_xlabel('Concentration (mol/kg)', fontsize=14)
    ax.set_ylabel(r'$L^{+-}$ (mS/cm)', fontsize=14)
    ax.set_ylim(-2, 13)
    ax.set_yticks(np.arange(0, 13, 2))
    ax.set_yticklabels([f'{y:.0f}' for y in np.arange(0, 13, 2)], fontsize=12)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_xticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.2)], fontsize=12)
    ax.grid()
    plt.tight_layout()
    fig.savefig(f'{path}l-plus-minus.tiff', dpi=300)
    return None

def lij_com_vs_sol(conc_vals, avg_lij_all, std_lij_all, com_avg_lij_all, com_std_lij_all, path):
    # plot the lij values
    fig, ax = plt.subplots(1, 3, figsize=[7, 3.25])
    lij_labels = [r'$L^{++}$', r'$L^{--}$', r'$L^{+-}$']
    cation_labels = [r'$Cs^+$', r'$K^+$', r'$Na^+$', r'$Li^+$']
    markers = ['o', 's', '^', 'D']

    for cat, i in enumerate(cations):
        for j, label in enumerate(lij_labels):
            ax[j].plot(conc_vals, com_avg_lij_all[cat,:,j+2], marker=markers[cat], color=colors[cat], linestyle='-', label=f'{cation_labels[cat]} - com', alpha=0.6, linewidth=2)
            ax[j].errorbar(conc_vals, com_avg_lij_all[cat,:,j+2], yerr=com_std_lij_all[cat,:,j+2], fmt=markers[cat], color=colors[cat], capsize=5)

            ax[j].plot(conc_vals, avg_lij_all[cat,:,j+2], marker=markers[cat], color=colors[cat], label=f'{cation_labels[cat]} - sol', linestyle=':')
            ax[j].errorbar(conc_vals, avg_lij_all[cat,:,j+2], yerr=std_lij_all[cat,:,j+2], fmt=markers[cat], color=colors[cat], capsize=5)

            if cat==0:
                ax[j].set_xlabel('Concentration (mol/kg)', fontsize=11)
                ax[j].set_ylabel(rf'{label} (mS/cm)', fontsize=11)
                ax[j].grid()

    ax[0].set_ylim(-3, 65)
    ax[1].set_ylim(-3, 65)
    ax[2].set_ylim(-15, 15)
    # ax[2].legend(fontsize=12, frameon=True, loc='lower left', ncols =2)
    plt.tight_layout()

    fig.savefig(f'{path}lij-comparison-to-com.tiff', dpi=300)
    return None

def supporting_electrolyte(data_folder, figure_folder):
    # import raw lij data, diffusion data, and concentration array
    n_molecules = [1110, 21, 20, 1] # water, cation, perchlorate, nitrate
    cations = ['Cs', 'K', 'Na', 'Li']
    lij_names = ['Cation self', 'Anion self']
    cation_masses = [132.905, 39.0983, 22.989769, 6.939]  # g/mol
    # color map
    colors = ["#2066a8", "#d47f72",  "#4db4e8", "#ae282c"]
    lij_all_array = np.empty((len(cations), len(lij_names)), dtype=object)
    avg_lij_all = np.empty((len(cations), len(lij_names)))
    std_lij_all = np.empty((len(cations), len(lij_names)))

    cond_all = np.empty((len(cations)), dtype=object)
    avg_cond_all = np.empty((len(cations)))
    std_cond_all = np.empty((len(cations)))

    diff_all = np.empty((len(cations), len(lij_names)), dtype=object)
    avg_diff_all = np.empty((len(cations), len(lij_names)))
    std_diff_all = np.empty((len(cations), len(lij_names)))

    def self_diffusion_coeff(lij, conc):
        c = conc  # mol/m^3
        R = 8.314  # J/(mol*K)
        T = 298.15  # K
        return (lij * 0.1 / 96485**2) * (R * T /c) # m^2/s

    def transference_number(lij):
        return (lij[3,:] - lij[4,:]) / (lij[2,:] + lij[3,:] - 2*lij[4,:])


    for k, cation in enumerate(cations):
            lij = np.load(f'{data_folder}{cation}/bulk/supporting-elyte/total-transport/lij.npy')
            cond = np.load(f'{data_folder}{cation}/bulk/supporting-elyte/total-transport/cond.npy')
            lij_beta_all = np.load(f'{data_folder}{cation}/bulk/supporting-elyte/total-transport/beta.npy')
            cond_beta_all = np.load(f'{data_folder}{cation}/bulk/supporting-elyte/total-transport/cond_beta.npy')

            # determine avg volume
            for r in range(3):
                path = f'{data_folder}{cation}/bulk/supporting-elyte/rep{r}/volume.npy'
                if r == 0:
                    volumes = np.empty(3)
                volumes[r] = np.load(path)
            avg_volume = np.mean(volumes)  # in Å^3
            print(f'Average volume for {cation}: {avg_volume} Å^3')

            # calculate concentration values in mol/m^3
            cation_conc = n_molecules[1] / (avg_volume * 1e-30)  # in molecules/m^3
            cation_conc = cation_conc  / 6.022e23  # convert to mol/m^3
            print(f' cation concentration: {cation_conc * 1e-3} mol/L')

            nitrate_conc = n_molecules[3]/ (avg_volume * 1e-30)  # in molecules/m^3
            nitrate_conc = nitrate_conc  / 6.022e23  # convert to mol/m^3
            print(f' nitrate concentration: {nitrate_conc * 1e-3} mol/L')


            for i in range(len(lij_names)):
                lij_all_array[k,i] = lij[i,:]

            print(f'{cation} lij :', lij.shape)
            print(np.mean(lij, axis=1))
            avg_lij_all[k,:] = np.mean(lij, axis=1)
            std_lij_all[k,:] = np.std(lij, axis=1)

            cond_all[k] = cond
            print(f'{cation} cond :', cond)
            avg_cond_all[k] = np.mean(cond)
            std_cond_all[k] = np.std(cond)


            diff_all[k,0] = self_diffusion_coeff(lij[0,:], cation_conc)
            diff_all[k,1] = self_diffusion_coeff(lij[1,:], nitrate_conc)
            avg_diff_all[k,0] = np.mean(diff_all[k,0])
            avg_diff_all[k,1] = np.mean(diff_all[k,1])
            std_diff_all[k,0] = np.std(diff_all[k,0])
            std_diff_all[k,1] = np.std(diff_all[k,1])

    # diffusion coefficients
    # plot the self diffusion coefficients wrt to cation identity
    cation_labels = [r'Cs$^+$', r'K$^+$', r'Na$^+$', r'Li$^+$']
    cation_xvals = [4, 3, 2, 1]
    colors2 = ["#A27514", "#049ea9", "#e6c352"]
    fig, ax = plt.subplots(1, 1, figsize=[3.25,3.25])
    ax.scatter(cation_xvals, avg_diff_all[:,0]*1e4, marker='o', label='Cation', color=colors2[0])
    ax.scatter(cation_xvals, avg_diff_all[:,1]*1e4, marker='^', label='Nitrate', color=colors2[1])
    ax.plot(cation_xvals, avg_diff_all[:,0]*1e4, color=colors2[0], linestyle='--')
    ax.plot(cation_xvals, avg_diff_all[:,1]*1e4, color=colors2[1], linestyle='--')

    for k, cation in enumerate(cations):
        ax.errorbar(cation_xvals[k], avg_diff_all[k,0]*1e4, yerr=std_diff_all[k,0]*1e4, fmt='o', capsize=5, color=colors2[0])
        ax.errorbar(cation_xvals[k], avg_diff_all[k,1]*1e4, yerr=std_diff_all[k,1]*1e4, fmt='^', capsize=5, color=colors2[1])

    ax.set_xticks(cation_xvals)
    ax.set_xticklabels(cation_labels, fontsize=11)
    ax.set_ylabel(r'Diffusion Coefficient ($\mathrm{cm^2/s}$)', fontsize=11)
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(0.6e-5, 1.7e-5)
    ax.grid()
    # ax.legend(fontsize=10, frameon=True, loc='lower right')
    plt.tight_layout()
    # figure_folder = '/Users/maddymurphy/Documents/GitRepos/phd/specific-ion-effects-eno3rr/figures/si/'
    fig.savefig(f'{figure_folder}self-diffusion-supporting-elyte.tiff', dpi=300)

    # conductivity
    # plot the conductivity
    cation_labels = [r'Cs$^+$', r'K$^+$', r'Na$^+$', r'Li$^+$']
    cation_xvals = [4, 3, 2, 1]

    fig, ax = plt.subplots(1, 1, figsize=[3.25,3.25])

    ax.scatter(cation_xvals, avg_cond_all[:], marker='o', label='Conductivity', color=colors2[0])
    ax.plot(cation_xvals, avg_cond_all[:], color=colors2[0], linestyle='--')
    for k, cation in enumerate(cations):
        ax.errorbar(cation_xvals[k], avg_cond_all[k], yerr=std_cond_all[k], fmt='o', capsize=5, color=colors2[0])

    ax.set_xticks(cation_xvals)
    ax.set_xticklabels(cation_labels, fontsize=11)
    ax.set_ylabel(r'Conductivity (mS/cm)', fontsize=11)
    ax.set_xlim(0.5, 4.5)
    ax.grid()
    plt.tight_layout()

    fig.savefig(f'{figure_folder}conductivity-supporting-elyte.tiff', dpi=300)
    return None

def main():
    fig_folder = '../../figures/si/'
    # make sure the figure folder exists
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    # load and average data
    # import raw lij data, diffusion data, and concentration array
    concentrations = ['1M', '0.5M', '0.1M', '0.01M']
    conc_vals = [1.0, 0.5, 0.1, 0.01]  # Molar concentrations
    num_ion = [6, 3, 1, 1]
    num_water = [332, 332, 560, 5600]
    cations = ['Cs', 'K', 'Na', 'Li']
    lij_names = ['Cation self', 'Anion self', 'Cation total', 'Anion total', 'Cation-Anion distinct']
    cation_masses = [132.905, 39.0983, 22.989769, 6.939]  # g/mol
    # color map
    # colors = ["#2066a8", "#d47f72",  "#4db4e8", "#ae282c"]
    colors = ["#1c4f7e", "#E05656",  "#4db4e8", "#8d0a0e"]
    lij_all_array = np.empty((len(cations), len(concentrations), len(lij_names)), dtype=object)
    avg_lij_all = np.empty((len(cations), len(concentrations), len(lij_names)))
    std_lij_all = np.empty((len(cations), len(concentrations), len(lij_names)))

    com_lij_all_array = np.empty((len(cations), len(concentrations), len(lij_names)), dtype=object)
    com_avg_lij_all = np.empty((len(cations), len(concentrations), len(lij_names)))
    com_std_lij_all = np.empty((len(cations), len(concentrations), len(lij_names)))

    cond_all = np.empty((len(cations), len(concentrations)), dtype=object)
    avg_cond_all = np.empty((len(cations), len(concentrations)))
    std_cond_all = np.empty((len(cations), len(concentrations)))

    com_cond_all = np.empty((len(cations), len(concentrations)), dtype=object)
    com_avg_cond_all = np.empty((len(cations), len(concentrations)))
    com_std_cond_all = np.empty((len(cations), len(concentrations)))

    D_el_all = np.empty((len(cations), len(concentrations)), dtype=object)
    avg_D_el_all = np.empty((len(cations), len(concentrations)))
    std_D_el_all = np.empty((len(cations), len(concentrations)))

    diff_all = np.empty((len(cations), len(concentrations), 2), dtype=object)
    avg_diff_all = np.empty((len(cations), len(concentrations), 2))
    std_diff_all = np.empty((len(cations), len(concentrations), 2))

    tn_all = np.empty((len(cations), len(concentrations)), dtype=object)
    avg_tn_all = np.empty((len(cations), len(concentrations)))
    std_tn_all = np.empty((len(cations), len(concentrations)))
    sizes = ['0', '1', '2', '3']
    box_sizes_all = np.empty((len(cations), len(concentrations), len(sizes)))

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

    folder = f'../../data/'

    for k, cation in enumerate(cations):
        for j, conc in enumerate(concentrations):

            lij_com= np.load(f'{folder}{cation}/bulk/{conc}/total-transport/lij.npy')
            lij_sol = np.load(f'{folder}{cation}/bulk/{conc}/total-transport/lij_sol.npy')
            lij_beta_all = np.load(f'{folder}{cation}/bulk/{conc}/total-transport/beta.npy')

            for i in range(len(lij_names)):
                lij_all_array[k,j,i] = lij_sol[i,:]
                com_lij_all_array[k,j,i] = lij_com[i,:]

            avg_lij_all[k,j,:] = np.mean(lij_sol, axis=1)
            std_lij_all[k,j,:] = np.std(lij_sol, axis=1)
            com_avg_lij_all[k,j,:] = np.mean(lij_com, axis=1)
            com_std_lij_all[k,j,:] = np.std(lij_com, axis=1)

            cond_all[k,j] = cond(lij_sol)
            avg_cond_all[k,j] = np.mean(cond(lij_sol))
            std_cond_all[k,j] = np.std(cond(lij_sol))

            com_cond_all[k,j] = cond(lij_com)
            com_avg_cond_all[k,j] = np.mean(cond(lij_com))
            com_std_cond_all[k,j] = np.std(cond(lij_com))

            diff_all[k,j,0] = self_diffusion_coeff(lij_com[0,:], conc_vals[j])
            diff_all[k,j,1] = self_diffusion_coeff(lij_com[1,:], conc_vals[j])
            avg_diff_all[k,j,0] = np.mean(self_diffusion_coeff(lij_com[0,:], conc_vals[j]))
            avg_diff_all[k,j,1] = np.mean(self_diffusion_coeff(lij_com[1,:], conc_vals[j]))
            std_diff_all[k,j,0] = np.std(self_diffusion_coeff(lij_com[0,:], conc_vals[j]))
            std_diff_all[k,j,1] = np.std(self_diffusion_coeff(lij_com[1,:], conc_vals[j]))

            tn_all[k,j] = transference_number(lij_sol)
            avg_tn_all[k,j] = np.mean(transference_number(lij_sol))
            std_tn_all[k,j] = np.std(transference_number(lij_sol))

            D_el_all[k,j] = diff_el(lij_sol, conc_vals[j], cation_masses[k], num_ion[j], num_water[j])
            # print('cation, conc, D_el_all:', cation, conc, D_el_all[k,j])
            avg_D_el_all[k,j] = np.mean(diff_el(lij_sol, conc_vals[j], cation_masses[k], num_ion[j], num_water[j]))
            std_D_el_all[k,j] = np.std(diff_el(lij_sol, conc_vals[j], cation_masses[k], num_ion[j], num_water[j]))

    # create figures
    diffusion(conc_vals, avg_D_el_all, std_D_el_all, fig_folder)
    transference(conc_vals, avg_tn_all, std_tn_all, fig_folder)
    conductivity_com_vs_sol(conc_vals, avg_cond_all, std_cond_all, com_avg_cond_all, com_std_cond_all, fig_folder)
    lij_com_vs_sol(conc_vals, avg_lij_all, std_lij_all, com_avg_lij_all, com_std_lij_all, fig_folder)
    l_plus_minus(conc_vals, avg_lij_all[:,:,4], std_lij_all[:,:,4], fig_folder)

    # finite size effects
    concentrations = ['1M', '0.5M', '0.1M', '0.01M']
    conc_vals = [1.0, 0.5, 0.1, 0.01]  # molality values for the concentrations
    cations = ['Cs', 'K', 'Na', 'Li']
    sizes = ['0', '1', '2', '3']
    lij_names = ['Cation self', 'Anion self', 'Cation total', 'Anion total', 'Cation-Anion distinct']

    lij_all_array = np.empty((len(cations), len(concentrations), len(lij_names), len(sizes)), dtype=object)
    beta_all_array = np.empty((len(cations), len(concentrations), len(lij_names), len(sizes)), dtype=object)
    avg_lij_all = np.empty((len(cations), len(concentrations), len(lij_names), len(sizes)))
    std_lij_all = np.empty((len(cations), len(concentrations), len(lij_names), len(sizes)))

    cond_all = np.empty((len(cations), len(concentrations), len(sizes)), dtype=object)
    avg_cond_all = np.empty((len(cations), len(concentrations), len(sizes)))
    std_cond_all = np.empty((len(cations), len(concentrations), len(sizes)))

    diff_all = np.empty((len(cations), len(concentrations), len(sizes), 2), dtype=object)
    avg_diff_all = np.empty((len(cations), len(concentrations), len(sizes), 2))
    std_diff_all = np.empty((len(cations), len(concentrations), len(sizes), 2))

    box_sizes_all = np.empty((len(cations), len(concentrations), len(sizes)))

    def cond(lij):
        return lij[2,:] + lij[3,:] - 2 * lij[4,:]

    def self_diffusion_coeff(lij, conc):
        c = conc * 1e3  # mol/L to mol/m^3 conversion
        R = 8.314  # J/(mol*K)
        T = 298.15  # K
        return (lij * 0.1 / 96485**2) * (R * T /c) # m^2/s

    for k, cation in enumerate(cations):
        for j, conc in enumerate(concentrations):
            for s, size in enumerate(sizes):
                if s == 0:
                    lij_all = np.load(f'{folder}{cation}/bulk/{conc}/total-transport/lij.npy')
                    lij_beta_all = np.load(f'{folder}{cation}/bulk/{conc}/total-transport/beta.npy')

                    for i in range(len(lij_names)):
                        lij_all_array[k,j,i,s] = lij_all[i,:]
                        beta_all_array[k,j,i,s] = lij_beta_all[i,:]
                    avg_lij_all[k,j,:,s] = np.mean(lij_all, axis=1)
                    std_lij_all[k,j,:,s] = np.std(lij_all, axis=1)

                    cond_all[k,j,s] = cond(lij_all)
                    avg_cond_all[k,j,s] = np.mean(cond(lij_all))
                    std_cond_all[k,j,s] = np.std(cond(lij_all))

                    diff_all[k,j,s,0] = self_diffusion_coeff(lij_all[0,:], conc_vals[j])
                    diff_all[k,j,s,1] = self_diffusion_coeff(lij_all[1,:], conc_vals[j])
                    avg_diff_all[k,j,s,0] = np.mean(self_diffusion_coeff(lij_all[0,:], conc_vals[j]))
                    avg_diff_all[k,j,s,1] = np.mean(self_diffusion_coeff(lij_all[1,:], conc_vals[j]))
                    std_diff_all[k,j,s,0] = np.std(self_diffusion_coeff(lij_all[0,:], conc_vals[j]))
                    std_diff_all[k,j,s,1] = np.std(self_diffusion_coeff(lij_all[1,:], conc_vals[j]))


                else:
                    lij_all = np.load(f'{folder}{cation}/bulk/{conc}/finite-size/box{size}/lij.npy')
                    lij_beta_all = np.load(f'{folder}{cation}/bulk/{conc}/finite-size/box{size}/beta.npy')

                    for i in range(len(lij_names)):
                        lij_all_array[k,j,i,s] = lij_all[i,:]
                        beta_all_array[k,j,i,s] = lij_beta_all[i,:]
                        # if conc == '0.01M':
                            # print(f'Box size {size} for {cation} {conc} has beta values: {lij_beta_all[i,:]}')
                    avg_lij_all[k,j,:,s] = np.mean(lij_all, axis=1)
                    std_lij_all[k,j,:,s] = np.std(lij_all, axis=1)

                    cond_all[k,j,s] = cond(lij_all)
                    avg_cond_all[k,j,s] = np.mean(cond(lij_all))
                    std_cond_all[k,j,s] = np.std(cond(lij_all))

                    diff_all[k,j,s,0] = self_diffusion_coeff(lij_all[0,:], conc_vals[j])
                    diff_all[k,j,s,1] = self_diffusion_coeff(lij_all[1,:], conc_vals[j])
                    avg_diff_all[k,j,s,0] = np.mean(self_diffusion_coeff(lij_all[0,:], conc_vals[j]))
                    avg_diff_all[k,j,s,1] = np.mean(self_diffusion_coeff(lij_all[1,:], conc_vals[j]))
                    std_diff_all[k,j,s,0] = np.std(self_diffusion_coeff(lij_all[0,:], conc_vals[j]))
                    std_diff_all[k,j,s,1] = np.std(self_diffusion_coeff(lij_all[1,:], conc_vals[j]))

                # read in the box sizes
                box_sizes = np.load(f'{folder}{cation}/bulk/{conc}/finite-size/box{size}/box_sizes.npy')
                avg_box_size = np.mean(box_sizes)

                box_sizes_all[k,j,s] = avg_box_size

    # create finite size figures
    beta_fs_effects(folder, fig_folder=fig_folder, colors=colors)
    finite_size_effects(concentrations, box_sizes_all, avg_diff_all, std_diff_all, fig_folder)
    supporting_electrolyte(folder, fig_folder)
    return None

if __name__ == "__main__":
    main()