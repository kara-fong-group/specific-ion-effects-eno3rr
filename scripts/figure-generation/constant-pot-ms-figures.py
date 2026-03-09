""" 
Script to generate figures for manuscript from processed data.
Madeline Murphy, June 2024

Figures included:
2a: cation density profiles at different potentials
2c: nitrate free energy of adsorption at all potentials
3: water density and potential profiles
4a: bulk layer rdf
4b: ion pairing
4d: stern layer pmf for ion pairs

files needed to run (for each cation and potential and metal parameter set):

density / structure analysis:
    density.npy
    edges.npy
    electrostatic-potential.npy
    area.npy
    n-atoms.npy

EDL layer analysis:
    layer-analysis/{layer}/rdf.npy
    layer-analysis/{layer}/pmf.npy
    layer-analysis/{layer}/bins.npy

    layer-analysis/{layer}/no3_pairing.npy
    layer-analysis/{layer}/no3_counts.npy
    


"""

from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re 
import MDAnalysis as mda
import glob
import os

cations = ['Cs', 'K', 'Na', 'Li']
potentials = ['00','10','20']
reps = ['0','1','2', '3', '4', '5', '6', '7', '8', '9']

data_folder = f'../../data/'
compressed_data_folder = f'../../data-ms-figs/constant-pot'
figure_folder = f'../../figures/ms/'

# create figure folder
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)

def plot_cation_density_profiles(edges, avg_density, std_density, colors):
    # plot charge density profiles
    fig, ax = plt.subplots(1,3, figsize=(7,3), sharey=True)
    pot_labels = ['0.0V', '1.0 V', '2.0 V']

    for i, cation in enumerate(cations):
        if cation == 'Cs' or cation == 'Li': # only plot Cs and Li for main text
            for j, potential in enumerate(pot_labels):
                ax[j].plot(edges[i,j,0], avg_density[i,j]['cation'], label=f'{cation}', color=colors[i])
                ax[j].fill_between(edges[i,j,0], avg_density[i,j]['cation']-std_density[i,j]['cation'], avg_density[i,j]['cation']+std_density[i,j]['cation'], alpha=0.3, color=colors[i])

                ax[j].set_xlim(1, -26)
                xticks = np.arange(-25, 1, 5)
                xlabels = [str(-x) for x in xticks]
                ax[j].set_xticks(xticks)
                ax[j].set_xticklabels(xlabels, fontsize=10)
                # ax[j].set_ylim(0, 0.3)
                # ax[j].set_yticks(np.arange(0, 0.1, 0.01))
                # ax[j].set_yticklabels(np.round(np.arange(0, 0.1, 0.01), 3), fontsize=10)

                # ax[j].set_title(r'$\phi_{\mathrm{app}}$' + f' = {pot_labels[j]}', fontsize=12, x=0.74, y=0.85)
                ax[j].set_xlabel(r'z ($\mathrm{\AA}$)', fontsize=12)
                ax[0].set_ylabel(r'Cation Density (1/$\mathrm{nm^3}$)', fontsize=12)
                # ax[0].legend(fontsize=12, ncols=1, loc='lower right')
                if i == 0:
                    ax[j].grid()
    plt.tight_layout()
    fig.savefig(f'{figure_folder}cation-density-profiles-cs-and-li.tiff', dpi=300)

    return None

def plot_water_potential(edges, avg_density, std_density, avg_epot, std_epot):
    fig, ax1 = plt.subplots(1,1, figsize=(3.75, 3.25))
    ax2 = ax1.twinx()
    j = 1
    colors1 = ["#0c2944", "#0254A1", "#3d8ee0", "#9acaf7"]
    colors2 = [ "#5e1316",  "#B72D31",  "#D64454",  "#E6A7B8"]

    for i, cation in enumerate(cations):
        ax1.plot(-edges[i,j,0], avg_epot[i,j], label=f'{cation}', color=colors1[i])
        ax1.fill_between(-edges[i,j,0], avg_epot[i,j]-std_epot[i,j], avg_epot[i,j]+std_epot[i,j], alpha=0.3, color=colors1[i])

        ax2.plot(-edges[i,j,0], avg_density[i,j]['water'], label=f'{cation}' , color=colors2[i])
        ax2.fill_between(-edges[i,j,0], avg_density[i,j]['water']-std_density[i,j]['water'], avg_density[i,j]['water']+std_density[i,j]['water'], alpha=0.3, color=colors2[i])

    ax1.set_xlabel(r'z ($\mathrm{\AA}$)', fontsize=12)
    ax1.set_ylabel('$\Phi$ (V)', fontsize=12, color=colors1[1])
    ax2.set_ylabel('Oxygen Density (g/cm³)', fontsize=12, color=colors2[1])
    ax1.grid()

    ax1.set_xlim(-5, 15)
    ax1.set_xticks(np.arange(-5, 16, 5))
    ax1.set_xticklabels([str(x) for x in np.arange(-5, 16, 5)], fontsize=10)

    ax1.set_ylim(-0.4, 1.4)
    ax1.set_yticks(np.arange(-0.4, 1.6, 0.4))
    ax1.set_yticklabels([str(round(y,1)) for y in np.arange(-0.4, 1.6, 0.4)], fontsize=10, color=colors1[1])
    ax1.spines['left'].set_color(colors1[1])


    ax2.set_ylim(0, 1.8)
    ax2.set_yticks(np.arange(0, 1.81, 0.4))
    ax2.set_yticklabels([str(round(y,1)) for y in np.arange(0, 1.81, 0.4)], fontsize=10, color=colors2[1])
    ax2.spines['right'].set_color(colors2[1])
    # ax1.legend(fontsize=10, loc='upper right')
    # ax2.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    fig.savefig(f'{figure_folder}potential-waterdensity-1V.tiff', dpi=300)
    return None

def bulk_rdf(bins, rdf_avg, rdf_std, colors):
    # plot nvt rdf
    fig, ax = plt.subplots(1,1, figsize=(3.5, 2.7))

    for i, cation in enumerate(cations):
        ax.plot(bins, rdf_avg[i,0,2], label=f'{cation}', color=colors[i])
        ax.fill_between(bins, rdf_avg[i,0,2]-rdf_std[i,0,2], rdf_avg[i,0,2]+rdf_std[i,0,2], alpha=0.3, color=colors[i])
        # ax.set_title('1.0M Bulk', fontsize=14, y=0.82, x=0.8)

        ax.set_xlabel(r'r ($\mathrm{\AA}$)', fontsize=11)
        ax.set_ylabel('g(r)', fontsize=11)
        ax.set_xlim(2,10)
        ax.set_ylim(0,7.5)

        x_ticks = np.arange(2, 11, 2)
        x_labels = [str(x) for x in x_ticks]
        ax.set_xticks(x_ticks)  
        ax.set_xticklabels(x_labels, fontsize=10)
        y_ticks = np.arange(0, 7.1, 2)
        y_labels = [str(y) for y in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=10)
        
    ax.grid()   
    # ax.legend(loc='right', fontsize=12, ncols=2, bbox_to_anchor=(1.5, 0.5))
    plt.tight_layout()
    fig.savefig(f'{figure_folder}rdf-ion-pairs-bulk.tiff', dpi=300)
    return None

def frac_ion_pairs(no3_pairing_avg, no3_pairing_std):
    # plot only the stern layer 
    fig, ax = plt.subplots(1,1, figsize=(3.25,2.5))
    pot_labels = ['0.0V', '1.0V', '2.0V']
    potentials = ['00', '10', '20']
    markers = ['o', 's', '^', 'D']
    # colors = ['#0d7d87', '#c31e23', '#ff5a5e', '#99c6cc']
    colors = ["#A27514", "#55bbbb", "#016e76", "#e6c352"]
    cat_labels = [r'Li$^+$', r'Na$^+$', r'K$^+$', r'Cs$^+$']
    x_array = np.array([3, 2, 1, 0])

    ax.scatter(x_array, no3_pairing_avg[:,0,2,0], label=f'Bulk layer', color=colors[3], marker=markers[0])
    ax.errorbar(x_array, no3_pairing_avg[:,0,2,0], yerr=no3_pairing_std[:,0,2,0], fmt=markers[0], capsize=5, color=colors[3], markersize=2)

    # for j, pot in enumerate(potentials):
    #     ax.scatter(x_array, no3_pairing_avg[:,j,0,0], label=f'{pot_labels[j]} - Stern', color=colors[j], marker=markers[j+1])
    #     ax.errorbar(x_array, no3_pairing_avg[:,j,0,0], yerr=no3_pairing_std[:,j,0,0], fmt=markers[j+1], capsize=5, color=colors[j])


    ax.set_ylim(0,0.65)
    ax.set_ylabel(r'$f_{\mathrm{paired}, \mathrm{NO^{-}_3}}$', fontsize=12)
    y_ticks = np.arange(0, 0.7, 0.1)
    y_ticklabels = [str(round(y,1)) for y in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels, fontsize=10)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(cat_labels, fontsize=12)
    ax.set_xlim(-0.5, 3.5)
    ax.grid()
    # ax.legend(ncols=2, fontsize=10, frameon=True, loc='upper left')
    plt.tight_layout()
    fig.savefig(f'{figure_folder}ion-pair-fraction-bulk.tiff', dpi=300)
    return None

def nitrate_free_energy_inset(edges, avg_pmf):
    # plot one pmfs on one plot
    fig, ax = plt.subplots(1,1, figsize=(1.6,1.2))
    pot_labels = ['0.0 V', '1.0 V', '2.0 V']
    linestyles = ['-', '-', '-', '-.']

    i = 0  # Cs
    j = 0  # 0.0 V

    ax.plot(-edges[i,j,0], avg_pmf[i,j]['nitrate'], color='k', linestyle=linestyles[j])
    # ax.fill_between(-edges[i,j,0], avg_pmf[i,j]['nitrate']-std_pmf[i,j]['nitrate'], avg_pmf[i,j]['nitrate']+std_pmf[i,j]['nitrate'], alpha=0.3, color=colors[i])
    ax.set_xlim(2, 12)
    ax.set_ylim(-0.5, 3)

    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='y', labelleft=False)

    # x_ticks = np.arange(2, 13, 2)
    # x_labels = [str(x) for x in x_ticks]
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels(x_labels, fontsize=10)
    # ax.set_yticks(np.arange(0, 5.1, 1))
    # ax.set_yticklabels(np.arange(0, 5.1, 1), fontsize=10)

    ax.set_xlabel(r'z ($\mathrm{\AA}$)', fontsize=10)
    ax.set_ylabel(r'F ($\mathrm{k_BT}$)', fontsize=10)
    # ax.legend(fontsize=10, ncol=2)
    ax.grid()
    plt.tight_layout()
    fig.savefig(f'{figure_folder}nitrate-surface-pmf-inset.tiff', dpi=300)
    return None

def delta_f_ads(avg_delta_f_ads, std_delta_f_ads, colors):
    # plot delta_f_ads vs applied potential as a bar chart
    fig, ax = plt.subplots(1,1, figsize=(3.5,2.75))
    ax.grid()
    ax.set_axisbelow(True)
    x = np.arange(len(potentials))

    width = 0.2 # the width of the bars
    for j, cation in enumerate(cations):
        ax.bar(x + (j-1.5)*width, avg_delta_f_ads[j,:], yerr=std_delta_f_ads[j,:], width=width, label=f'{cation}', color=colors[j], capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(potential)/10:.1f} V' for potential in potentials], fontsize=12)
    ax.set_ylabel(r'$\Delta \mathrm{F}_{\mathrm{ads, NO_3^-}}$ $(\mathrm{k_{B} T})$', fontsize=11)
    # ax.legend(title='Cation', fontsize=10)
    ax.set_ylim(0, 5.0)
    plt.tight_layout()
    fig.savefig(f'{figure_folder}delta-f-ads-bar-chart.tiff', dpi=300)
    return None

def load_density(data_folder, cations, potentials, reps):
    # load density data
    density = np.empty((len(cations),len(potentials),len(reps)), dtype=object)
    pmf_array = np.empty((len(cations),len(potentials),len(reps)), dtype=object)
    edges = np.empty((len(cations),len(potentials),len(reps)), dtype=object)
    cell_width = np.empty((len(cations),len(potentials)), dtype=object)

    # compute units
    # determine density profile in g/cm^3
    atom_plot = ['water', 'nitrate', 'cation']
    atom_name = ['O', 'N_NO', 'C']
    Mw = {'water': 18.015, 'nitrate': 62.0049, 'Cs': 132.905, 'K': 39.0983, 'Na': 22.9897, 'Li': 6.94}
    Na = 6.02214076e23  # Avogadro's number

    def pmf(density):
        pmf_raw = -np.log(density)
        bulk = pmf_raw[len(pmf_raw)//2]
        pmf_corrected = pmf_raw - bulk
        return pmf_corrected

    for i, cation in enumerate(cations):
        for j, potential in enumerate(potentials):
            for k, rep in enumerate(reps):
                folder = f'{data_folder}{cation}/constant-potential/wca/{potential}/rep{rep}/'
                density_raw = np.load(f'{folder}density.npy', allow_pickle=True).item()
                edges[i,j,k] = np.load(f'{folder}edges.npy', allow_pickle=True)

                area = np.load(f'{folder}area.npy', allow_pickle=True)
                n_atoms = np.load(f'{folder}n-atoms.npy', allow_pickle=True).item()
                density[i,j,k] = {}
                pmf_array[i,j,k] = {}
                for l, atom in enumerate(atom_plot):
                    pmf_array[i,j,k][atom] = pmf(density_raw[atom_name[l]])

                    if atom == 'cation':
                        density[i,j,k][atom] = (density_raw[atom_name[l]] / (area * 1e-3)) * n_atoms[atom_name[l]] # * Mw[cation] / Na
                    else:
                        density[i,j,k][atom] = (density_raw[atom_name[l]] / (area * 1e-24)) * n_atoms[atom_name[l]] * Mw[atom] / Na

                # find the electrode edge
                idx_electrode = np.where(density_raw['Au'] > 0.1)
                elec_edge_1 = edges[i,j,k][idx_electrode[0][9]]
                elec_edge_2 = edges[i,j,k][idx_electrode[0][10]]
                cell_width[i,j] = elec_edge_2 - elec_edge_1
                edges[i,j,k] = edges[i,j,k] - elec_edge_2

    # average density profiles
    avg_density = np.empty((len(cations),len(potentials)), dtype=object)
    std_density = np.empty((len(cations),len(potentials)), dtype=object)

    avg_pmf = np.empty((len(cations), len(potentials)), dtype=object)
    std_pmf = np.empty((len(cations), len(potentials)), dtype=object)

    atoms = ['cation', 'nitrate', 'water']

    for i, cation in enumerate(cations):
        for j, potential in enumerate(potentials):
            avg_density[i,j] = {}
            std_density[i,j] = {}
            avg_pmf[i,j] = {}
            std_pmf[i,j] = {}

            for atom in atoms:
                avg_density[i,j][atom] = np.mean([density[i,j,k][atom] for k in range(len(reps))], axis=0)
                std_density[i,j][atom] = np.std([density[i,j,k][atom] for k in range(len(reps))], axis=0)
                avg_pmf[i,j][atom] = np.mean([pmf_array[i,j,k][atom] for k in range(len(reps))], axis=0)
                std_pmf[i,j][atom] = np.std([pmf_array[i,j,k][atom] for k in range(len(reps))], axis=0)

    # compute the free energy of adsorption for each cation at each potential
    z_min_values = np.empty((len(cations)))
    delta_f_ads = np.empty((len(cations), len(potentials), len(reps)))

    avg_delta_f_ads = np.empty((len(cations), len(potentials)))
    std_delta_f_ads = np.empty((len(cations), len(potentials)))
    for i, cation in enumerate(cations):
        for j, potential in enumerate(potentials):
            if j == 0:
                # determine the z-location of the first well in the pmf
                pmf_profile = avg_pmf[i,j]['nitrate']
                edges_profile = edges[i,j,0]
                # find the indices of edges > -5
                valid_indices = np.where(edges_profile > -6)[0]
                
                min_idx = np.argmin(pmf_profile[valid_indices])
                z_min = edges_profile[valid_indices][min_idx]
                z_min_values[i] = z_min

            for k, rep in enumerate(reps):
                # find the index closest to z_min
                edges_profile = edges[i,j,k]
                pmf_profile = pmf_array[i,j,k]['nitrate']
                idx_closest = (np.abs(edges_profile - z_min_values[i])).argmin()
                delta_f_ads[i,j,k] = pmf_profile[idx_closest]  # in kT units

            avg_delta_f_ads[i,j] = np.mean(delta_f_ads[i,j])
            std_delta_f_ads[i,j] = np.std(delta_f_ads[i,j])
    return edges, avg_density, std_density, avg_pmf, std_pmf, avg_delta_f_ads, std_delta_f_ads

def load_epot(data_folder, cations, potentials, reps):
    # load electric field and electrostatic potential data
    epot = np.empty((len(cations),len(potentials),len(reps)), dtype=object)
    cutoffs = np.empty((len(cations),len(potentials),len(reps)), dtype=object)

    for i, cation in enumerate(cations):
        for j, potential in enumerate(potentials):
            for k, rep in enumerate(reps):
                folder = f'{data_folder}{cation}/constant-potential/wca/{potential}/rep{rep}/'
                epot[i,j,k] = np.load(f'{folder}electrostatic-potential.npy', allow_pickle=True)

    # average profiles
    avg_epot = np.empty((len(cations),len(potentials)), dtype=object)
    std_epot = np.empty((len(cations),len(potentials)), dtype=object)

    for i, cation in enumerate(cations):
        for j, potential in enumerate(potentials):
            avg_epot[i,j] = np.mean([epot[i,j,k] for k in range(len(reps))], axis=0)
            std_epot[i,j] = np.std([epot[i,j,k] for k in range(len(reps))], axis=0)

    return avg_epot, std_epot

def load_ion_pairing(data_folder, cations, potentials, reps):
    layers = ['stern', 'diffuse', 'bulk']
    rdf = np.empty((len(cations),len(potentials),len(reps), len(layers)), dtype=object)
    rdf_avg = np.empty((len(cations),len(potentials), len(layers)+1), dtype=object)
    rdf_std = np.empty((len(cations),len(potentials), len(layers)+1), dtype=object)

    pmf_d = np.empty((len(cations),len(potentials),len(reps), len(layers)), dtype=object)
    pmf_avg = np.empty((len(cations),len(potentials), len(layers)+1), dtype=object)
    pmf_std = np.empty((len(cations),len(potentials), len(layers)+1), dtype=object)

    for i, cation in enumerate(cations):
        for j, potential in enumerate(potentials):
            for l, layer in enumerate(layers):
                for k, rep in enumerate(reps):
                    folder = f'{data_folder}{cation}/constant-potential/wca/{potential}/rep{rep}/layer-analysis/{layer}/'
                    bins = np.load(f'{folder}bins.npy', allow_pickle=True)
                    rdf[i,j,k,l] = np.load(f'{folder}rdf.npy', allow_pickle=True)
                    pmf_d[i,j,k,l] = np.load(f'{folder}pmf.npy', allow_pickle=True)

                rdf_avg[i,j,l] = np.mean([rdf[i,j,k,l] for k in range(len(reps))], axis=0)
                rdf_std[i,j,l] = np.std([rdf[i,j,k,l] for k in range(len(reps))], axis=0)

                pmf_avg[i,j,l] = np.mean([pmf_d[i,j,k,l] for k in range(len(reps))], axis=0)
                pmf_std[i,j,l] = np.std([pmf_d[i,j,k,l] for k in range(len(reps))], axis=0)

    no3_pairing = np.empty((len(cations),len(potentials),len(reps),3,3), dtype=object)
    no3_pairing_avg = np.empty((len(cations),len(potentials),3,3), dtype=object)
    no3_pairing_std = np.empty((len(cations),len(potentials),3,3), dtype=object)

    n_no3 = np.empty((len(cations),len(potentials),len(reps),3), dtype=object)
    n_no3_avg = np.empty((len(cations),len(potentials),3), dtype=object)
    n_no3_std = np.empty((len(cations),len(potentials),3), dtype=object)

    for i, cation in enumerate(cations):
        for j, potential in enumerate(potentials):
                for k, rep in enumerate(reps):
                    folder = f'{data_folder}{cation}/constant-potential/wca/{potential}/rep{rep}/layer-analysis/'
                    no3_pairing[i,j,k,:,:] = np.load(f'{folder}no3_pairing.npy', allow_pickle=True)
                    n_no3[i,j,k,:,] = np.load(f'{folder}no3_counts.npy', allow_pickle=True)
                for x in range(3): # layer
                    for y in range(3): # pair type
                        no3_pairing_avg[i,j,x,y] = np.mean([no3_pairing[i,j,k,x,y] for k in range(len(reps))]) # cation, pot, pair type, layer
                        no3_pairing_std[i,j,x,y] = np.std([no3_pairing[i,j,k,x,y] for k in range(len(reps))])

                    n_no3_avg[i,j,x] = np.mean([n_no3[i,j,k,x] for k in range(len(reps))])
                    n_no3_std[i,j,x] = np.std([n_no3[i,j,k,x] for k in range(len(reps))])

    return bins, rdf_avg, rdf_std, pmf_avg, pmf_std, no3_pairing_avg, no3_pairing_std


def main():
    # load data for figures
    avg_density = np.load(f'{compressed_data_folder}/avg_density.npy', allow_pickle=True)
    std_density = np.load(f'{compressed_data_folder}/std_density.npy', allow_pickle=True)
    edges = np.load(f'{compressed_data_folder}/edges.npy', allow_pickle=True)
    avg_epot = np.load(f'{compressed_data_folder}/avg_epot.npy', allow_pickle=True)
    std_epot = np.load(f'{compressed_data_folder}/std_epot.npy', allow_pickle=True)
    bins = np.load(f'{compressed_data_folder}/bins.npy', allow_pickle=True)
    rdf_avg = np.load(f'{compressed_data_folder}/rdf_avg.npy', allow_pickle=True)
    rdf_std = np.load(f'{compressed_data_folder}/rdf_std.npy', allow_pickle=True)
    no3_pairing_avg = np.load(f'{compressed_data_folder}/no3_pairing_avg.npy', allow_pickle=True)
    no3_pairing_std = np.load(f'{compressed_data_folder}/no3_pairing_std.npy', allow_pickle=True)
    avg_delta_f_ads = np.load(f'{compressed_data_folder}/avg_delta_f_ads.npy', allow_pickle=True)
    std_delta_f_ads = np.load(f'{compressed_data_folder}/std_delta_f_ads.npy', allow_pickle=True)
    avg_pmf = np.load(f'{compressed_data_folder}/avg_pmf.npy', allow_pickle=True)

    # color map
    colors = ["#1c4f7e", "#E05656",  "#4db4e8", "#8d0a0e"]

    # generate figures
    plot_cation_density_profiles(edges, avg_density, std_density, colors)
    plot_water_potential(edges, avg_density, std_density, avg_epot, std_epot)
    bulk_rdf(bins, rdf_avg, rdf_std, colors)
    frac_ion_pairs(no3_pairing_avg, no3_pairing_std)
    delta_f_ads(avg_delta_f_ads, std_delta_f_ads, colors)
    nitrate_free_energy_inset(edges, avg_pmf)
    
if __name__ == "__main__":
    main()
    