""" 
Script to generate figures for SI from saved, preprocessed data.
Madeline Murphy, June 2024
"""

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
figure_folder = f'../../figures/si/'

# create figure folder
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)

def electric_field(edges, avg_epot, std_epot, colors, figure_folder):
    # plot the electric field
    fig, axes = plt.subplots(1,3, figsize=(7, 2.5))
    pot_labels = ['0.0 V', '1.0 V', '2.0 V']

    for i, cation in enumerate(cations):
        for j, potential in enumerate(potentials):
            axes[j].plot(-edges[i,j,0], avg_epot[i,j], label=f'{cation}', color=colors[i])
            axes[j].fill_between(-edges[i,j,0], avg_epot[i,j]-std_epot[i,j], avg_epot[i,j]+std_epot[i,j], alpha=0.3, color=colors[i])
            if i == 0:
                axes[j].set_xlim(-10, 65)
                #axes[j].set_ylim(0, 0.05)
                # axes[j].set_title(f'potential: {pot_labels[j]}', fontsize=14)
                axes[j].set_xlabel(r'z ($\mathrm{\AA}$)', fontsize=11)
                axes[0].set_ylabel(r'$\phi$ (V)', fontsize=11)
                axes[j].grid()

            # axes[j].legend(ncols=2)
    plt.tight_layout()
    fig.savefig(f'{figure_folder}electrostatic-potential-profiles.tiff', dpi=300)
    return None

def nitrate_density(edges, avg_density, std_density, colors, figure_folder):
    # plot charge density profiles
    fig, ax = plt.subplots(1,3, figsize=(7,3), sharey=True)
    pot_labels = ['0.0 V', '1.0 V', '2.0 V']

    for i, cation in enumerate(cations):
        for j, potential in enumerate(pot_labels):
            ax[j].plot(edges[i,j,0], avg_density[i,j]['nitrate'], label=f'{cation}' , color=colors[i])
            ax[j].fill_between(edges[i,j,0], avg_density[i,j]['nitrate']-std_density[i,j]['nitrate'], avg_density[i,j]['nitrate']+std_density[i,j]['nitrate'], alpha=0.3, color=colors[i])

            ax[j].set_xlim(0, -25)
            xticks = np.arange(-25, 1, 5)
            xlabels = [str(-x) for x in xticks]
            ax[j].set_xticks(xticks)
            ax[j].set_xticklabels(xlabels, fontsize=10)
            ax[j].set_ylim(0, 1.0)

            # ax[j].set_title(r'$\Delta\phi$' + f' = {pot_labels[j]}', fontsize=12, x=0.7, y=0.85)
            ax[j].set_xlabel(r'z ($\mathrm{\AA}$)', fontsize=12)
            ax[0].set_ylabel(r'Density (1/$\mathrm{nm^3}$)', fontsize=12)
            # ax[0].legend(fontsize=12)
            if i == 0:
                ax[j].grid()
    plt.tight_layout()
    fig.savefig(f'{figure_folder}nitrate-density.tiff', dpi=300)

def nitrate_free_energy(edges, avg_pmf, std_pmf, colors, figure_folder):
    # plot charge density profiles
    fig, ax = plt.subplots(1,3, figsize=(7,3), sharey=True)
    pot_labels = ['0.0 V', '1.0 V', '2.0 V']

    for i, cation in enumerate(cations):
        for j, potential in enumerate(pot_labels):
            ax[j].plot(edges[i,j,0], avg_pmf[i,j]['nitrate'], label=f'{cation}' , color=colors[i])
            ax[j].fill_between(edges[i,j,0], avg_pmf[i,j]['nitrate']-std_pmf[i,j]['nitrate'], avg_pmf[i,j]['nitrate']+std_pmf[i,j]['nitrate'], alpha=0.3, color=colors[i])

            ax[j].set_xlim(0, -25)
            xticks = np.arange(-25, 1, 5)
            xlabels = [str(-x) for x in xticks]
            ax[j].set_xticks(xticks)
            ax[j].set_xticklabels(xlabels, fontsize=10)
            ax[j].set_ylim(-1, 4.0)

            # ax[j].set_title(r'$\Delta\phi$' + f' = {pot_labels[j]}', fontsize=12, x=0.7, y=0.85)
            ax[j].set_xlabel(r'z ($\AA$)', fontsize=11)
            ax[0].set_ylabel(r'Free Energy (k$_\mathrm{B}$T)', fontsize=11)
            # ax[0].legend(fontsize=12)
            if i == 0:
                ax[j].grid()
    plt.tight_layout()
    fig.savefig(f'{figure_folder}nitrate-free-energy.tiff', dpi=300)

def full_density(edges, avg_density, std_density, colors, cat_labels, figure_folder):
    pot_labels = ['0V', '1V', '2V']
    for j, potential in enumerate(pot_labels):
        # plot all three atom types -- 1.0V only
        fig, ax = plt.subplots(2,2, figsize=(3.25, 3.25))
        axes = [ax[0,0], ax[0,1], ax[1,0], ax[1,1]]
            
        for i, cation in enumerate(cations):
            axes[i].plot(-edges[i,j,0], avg_density[i,j]['water'], label='water', color=colors[0])
            axes[i].fill_between(-edges[i,j,0], avg_density[i,j]['water']-std_density[i,j]['water'], avg_density[i,j]['water']+std_density[i,j]['water'], alpha=0.3, color=colors[0])

            axes[i].plot(-edges[i,j,0], avg_density[i,j]['nitrate'], label=r'NO$_{3}^{-}$', color=colors[1])
            axes[i].fill_between(-edges[i,j,0], avg_density[i,j]['nitrate']-std_density[i,j]['nitrate'], avg_density[i,j]['nitrate']+std_density[i,j]['nitrate'], alpha=0.3, color=colors[1])

            axes[i].plot(-edges[i,j,0], avg_density[i,j]['cation'], label=f'{cat_labels[i]}', color=colors[2])
            axes[i].fill_between(-edges[i,j,0], avg_density[i,j]['cation']-std_density[i,j]['cation'], avg_density[i,j]['cation']+std_density[i,j]['cation'], alpha=0.3, color=colors[2])

            axes[i].set_xlim(-5,25)
            # axes[i].set_ylim(0, 0.04)
            # axes[i].set_yticks([0, 0.01, 0.02, 0.03, 0.04])
            # axes[i].legend()
            axes[i].grid()

        ax[0,0].set_ylabel(r'Density 1/($\mathrm{\AA}$)', fontsize=11)
        ax[1,0].set_xlabel(r'z ($\mathrm{\AA}$)', fontsize=11)
        ax[1,1].set_xlabel(r'z ($\mathrm{\AA}$)', fontsize=11)
        ax[1,0].set_ylabel(r'Density 1/($\mathrm{\AA}$)', fontsize=11)
        # ax[0,0].set_title('1.0V Density Profiles', fontsize=14)

        plt.tight_layout()
        plt.savefig(f'{figure_folder}all-density-profiles-surface-{pot_labels[j]}.tiff', dpi=300)
    return None

def rdf_layers(bins, rdf_avg, rdf_std, colors, figure_folder):
    # plot the rdf of the ion pairs
    fig, ax = plt.subplots(3,3, figsize=(7,5), sharex=True, sharey=True)
    pot_labels = ['0.0 V', '1.0 V', '2.0 V']
    layers = ['stern', 'diffuse', 'bulk']

    for i, cation in enumerate(cations):
        for j, potential in enumerate(pot_labels):
            for l, layer in enumerate(layers):

                ax[j,l].plot(bins, rdf_avg[i,j,l], label=f'{cation}', color=colors[i])
                ax[j,l].fill_between(bins, rdf_avg[i,j,l]-rdf_std[i,j,l], rdf_avg[i,j,l]+rdf_std[i,j,l], alpha=0.3, color=colors[i])
                ax[j,l].set_xlim(2,10)

                if i == 1:
                    ax[j,l].grid()

                ax[j,0].set_ylabel('g(r)', fontsize=11)
                ax[2,l].set_xlabel(r'r ($\AA$)', fontsize=11)
                ax[j,l].set_ylim(0,8)
    # ax[2,2].legend(loc='lower right', ncols=2, fontsize=12)
    plt.tight_layout()
    fig.savefig(f'{figure_folder}rdf-ion-pairs.tiff', dpi=300)
    return None

def rdf_bulk(colors, figure_folder):
    fig, ax = plt.subplots(1,1, figsize=(3.25, 2.5))
    # import 1M bulk rdf data
    rdf_avg = np.empty((len(cations)), dtype=object)
    rdf_std = np.empty((len(cations)), dtype=object)
    for i, cation in enumerate(cations):
        rdf_avg[i] = np.load(f'{data_folder}{cation}/bulk/1M/rdf/rdf-cat-no3-avg.npy', allow_pickle=True)
        rdf_std[i] = np.load(f'{data_folder}{cation}/bulk/1M/rdf/rdf-cat-no3-std.npy', allow_pickle=True)
        bins_nvt = np.load(f'{data_folder}{cation}/bulk/1M/rdf/bins-cat-no3.npy', allow_pickle=True)[0,:]

    for i, cation in enumerate(cations):
        ax.plot(bins_nvt, rdf_avg[i], label=f'{cation}', color=colors[i])
        ax.fill_between(bins_nvt, rdf_avg[i]-rdf_std[i], rdf_avg[i]+rdf_std[i], alpha=0.3, color=colors[i])
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

def pmf_layers(bins, pmf_avg, pmf_std, colors, figure_folder):
    # plot pmfs of ion pairs
    fig, ax = plt.subplots(3,3, figsize=(7,5), sharex=True, sharey=True)
    pot_labels = ['0.0 V', '1.0 V', '2.0 V']
    layers = ['stern', 'diffuse', 'bulk']

    for i, cation in enumerate(cations):
        for j, potential in enumerate(pot_labels):
            for l, layer in enumerate(layers):

                ax[j,l].plot(bins, pmf_avg[i,j,l], label=f'{cation}', color=colors[i])
                ax[j,l].fill_between(bins, pmf_avg[i,j,l]-pmf_std[i,j,l], pmf_avg[i,j,l]+pmf_std[i,j,l], alpha=0.3, color=colors[i])
                # ax[j,l].set_title(r'$\phi_{app}$' +f' = {pot_labels[j]}  \n    layer = {layer} ', loc='right', y=0.8, fontsize=9)
                ax[j,l].set_xlim(2,10)

                if i == 1:
                    ax[j,l].grid()

                ax[j,0].set_ylabel(r'Free Energy ($\mathrm{k_B}$T)', fontsize=11)
                ax[2,l].set_xlabel(r'r ($\AA$)', fontsize=11)
                ax[j,l].set_ylim(-3,4)
    # ax[2,2].legend(loc='lower right', ncols=1, fontsize=10)
    plt.tight_layout()
    fig.savefig(f'{figure_folder}pmf-ion-pairs.tiff', dpi=300)
    return None

def cutoffs(edges, avg_epot, std_epot, avg_cutoffs, figure_folder, colors):
    fig, axes = plt.subplots(1, 3, figsize=(7, 3), sharey=True)
    pot_labels = ['0.0 V', '1.0 V', '2.0 V']
    for i, cation in enumerate(cations):
        for j, potential in enumerate(potentials):
            axes[j].plot(-edges[i, j, 0], avg_epot[i, j], label=f'{cation}', color=colors[i])
            axes[j].fill_between(-edges[i, j, 0], avg_epot[i, j] - std_epot[i,j], avg_epot[i, j] + std_epot[i,j], alpha=0.3, color=colors[i])

            # add cutoffs as vertical lines
            for k in range(3):
                cutoff = avg_cutoffs[i,j][k+1] - avg_cutoffs[i,j][0]
                print(f'cutoff {k} for {cation} at {potential}: {cutoff}')
                axes[j].axvline(x=-cutoff, color=colors[i], linestyle='--', alpha=0.5)
            # Axis formatting (only needs to happen once per subplot)
            if i == 0:
                axes[j].set_xlim(-2, 27)
                xticks = np.arange(0, 26, 5)
                xlabels = [str(x) for x in xticks]
                axes[j].set_xticks(xticks)
                axes[j].set_xticklabels(xlabels, fontsize=10)
                # axes[j].set_title(f'potential: {pot_labels[j]}', fontsize=14)
                axes[j].set_xlabel(r'z ($\mathrm{\AA}$)', fontsize=11)

                axes[j].set_ylim(-1.0, 1.5)
                y_ticks = np.arange(-1.0, 1.6, 0.5)
                y_ticklabels = [str(round(y,1)) for y in y_ticks]
                axes[j].set_yticks(y_ticks)
                axes[j].set_yticklabels(y_ticklabels, fontsize=10)
                axes[0].set_ylabel(r'$\phi$ (V)', fontsize=11)
                axes[j].grid()
    # axes[0].legend()

    plt.tight_layout()
    fig.savefig(f'{figure_folder}electrostatic-cutoffs.tiff', dpi=300)


def hydrogen_density_profiles(edges, avg_density, std_density, figure_folder, colors):
    # plot charge density profiles
    fig, ax = plt.subplots(1,3, figsize=(7,3), sharey=True)
    pot_labels = ['0.0 V', '1.0 V', '2.0 V']

    for i, cation in enumerate(cations):
        for j, potential in enumerate(pot_labels):
            ax[j].plot(edges[i,j,0], avg_density[i,j]['H'], label=f'{cation}' , color=colors[i])
            ax[j].fill_between(edges[i,j,0], avg_density[i,j]['H']-std_density[i,j]['H'], avg_density[i,j]['H']+std_density[i,j]['H'], alpha=0.3, color=colors[i])

            ax[j].set_xlim(5, -25)
            xticks = np.arange(-25, 1, 5)
            xlabels = [str(-x) for x in xticks]
            ax[j].set_xticks(xticks)
            ax[j].set_xticklabels(xlabels, fontsize=10)
            # ax[j].set_ylim(0, 100)

            # ax[j].set_title(r'$\Delta\phi$' + f' = {pot_labels[j]}', fontsize=12, x=0.7, y=0.85)
            ax[j].set_xlabel(r'z ($\AA$)', fontsize=11)
            ax[0].set_ylabel(r'Density (1/nm³)', fontsize=11)
            # ax[0].legend(fontsize=12)
            if i == 0:
                ax[j].grid()
    plt.tight_layout()
    fig.savefig(f'{figure_folder}hydrogen-water-density.tiff', dpi=300)

def one_pot_water_density(edges, avg_epot, std_epot, avg_density, std_density, figure_folder):
    # limaye et al data
    water_density_peak = [7.9006 - 4.479167, 1.3114]  # adjusted for our electrode position
    water_density_trough = [9.4917 - 4.479167, 0.9162]
    potential_peak = [5.8333 - 4.479167, 0.0066 + 0.5]  # adjusted for our electrode position and zeroing at bulk
    potential_trough = [7.4479 - 4.479167, -0.9571 + 0.5]

    fig, ax1 = plt.subplots(1,1, figsize=(4,3.25))
    ax2 = ax1.twinx()
    j = 1
    colors1 = ["#0c2944", "#0356a3", "#3984cf", "#90bce6",]
    colors2 = [ "#5e1316",  "#ae282c",  "#cd3f44",  "#c47172"]

    for i, cation in enumerate(cations):
        ax1.plot(-edges[i,j,0], avg_epot[i,j], label=f'{cation}', color=colors1[i])
        ax1.fill_between(-edges[i,j,0], avg_epot[i,j]-std_epot[i,j], avg_epot[i,j]+std_epot[i,j], alpha=0.3, color=colors1[i])

        ax2.plot(-edges[i,j,0], avg_density[i,j]['water'], label=f'{cation}' , color=colors2[i])
        ax2.fill_between(-edges[i,j,0], avg_density[i,j]['water']-std_density[i,j]['water'], avg_density[i,j]['water']+std_density[i,j]['water'], alpha=0.3, color=colors2[i])

    # plot limaye et al data
    ax1.scatter(potential_peak[0], potential_peak[1], color='black', marker='o', label='Limaye et al Peak')
    ax1.scatter(potential_trough[0], potential_trough[1], color='black', marker='o', label='Limaye et al Trough')

    ax2.scatter(water_density_peak[0], water_density_peak[1], color='black', marker='x', label='Limaye et al Peak')
    ax2.scatter(water_density_trough[0], water_density_trough[1], color='black', marker='x', label='Limaye et al Trough')

    ax1.set_xlabel(r'z ($\mathrm{\AA}$)', fontsize=11)
    ax1.set_ylabel('$\Phi$ (V)', fontsize=12, color=colors1[1])
    ax2.set_ylabel('Water Density (g/cm³)', fontsize=11, color=colors2[1])
    ax1.grid()

    ax1.set_xlim(-5, 15)
    ax1.set_xticks(np.arange(0, 16, 5))
    ax1.set_xticklabels([str(x) for x in np.arange(0, 16, 5)], fontsize=10)

    ax1.set_ylim(-0.4, 2.0)
    ax1.set_yticks(np.arange(-0.4, 2.1, 0.4))
    ax1.set_yticklabels([str(round(y,1)) for y in np.arange(-0.4, 2.1, 0.4)], fontsize=10, color=colors1[1])
    ax1.spines['left'].set_color(colors1[1])


    ax2.set_ylim(0, 1.8)
    ax2.set_yticks(np.arange(0, 1.81, 0.4))
    ax2.set_yticklabels([str(round(y,1)) for y in np.arange(0, 1.81, 0.4)], fontsize=10, color=colors2[1])
    ax2.spines['right'].set_color(colors2[1])
    plt.tight_layout()
    fig.savefig(f'{figure_folder}potential-waterdensity-1V-limaye-comparison.tiff', dpi=300)

def all_pot_water(edges, avg_epot, std_epot, avg_density, std_density, figure_folder):
    fig, ax1 = plt.subplots(1,1, figsize=(3.25,3.25))
    ax2 = ax1.twinx()
    j = 2
    colors1 = ["#0c2944", "#0254A1", "#3d8ee0", "#9acaf7"]
    colors2 = [ "#5e1316",  "#B72D31",  "#D64454",  "#E6A7B8"]

    for i, cation in enumerate(cations):
        ax1.plot(-edges[i,j,0], avg_epot[i,j], label=f'{cation}', color=colors1[i])
        ax1.fill_between(-edges[i,j,0], avg_epot[i,j]-std_epot[i,j], avg_epot[i,j]+std_epot[i,j], alpha=0.3, color=colors1[i])

        ax2.plot(-edges[i,j,0], avg_density[i,j]['water'], label=f'{cation}' , color=colors2[i])
        ax2.fill_between(-edges[i,j,0], avg_density[i,j]['water']-std_density[i,j]['water'], avg_density[i,j]['water']+std_density[i,j]['water'], alpha=0.3, color=colors2[i])

    ax1.set_xlabel(r'z ($\mathrm{\AA}$)', fontsize=11)
    # ax1.set_ylabel('$\Phi$ (V)', fontsize=12, color=colors1[1])
    ax2.set_ylabel('Water Density (g/cm³)', fontsize=11, color=colors2[1])
    ax1.grid()

    ax1.set_xlim(-5, 15)
    ax1.set_xticks(np.arange(0, 16, 5))
    ax1.set_xticklabels([str(x) for x in np.arange(0, 16, 5)], fontsize=10)

    ax1.set_ylim(-0.4, 2.0)
    ax1.set_yticks(np.arange(-0.4, 2.1, 0.4))
    ax1.set_yticklabels([str(round(y,1)) for y in np.arange(-0.4, 2.1, 0.4)], fontsize=10, color=colors1[1])
    ax1.spines['left'].set_color(colors1[1])


    ax2.set_ylim(0, 1.8)
    ax2.set_yticks(np.arange(0, 1.81, 0.4))
    ax2.set_yticklabels([str(round(y,1)) for y in np.arange(0, 1.81, 0.4)], fontsize=10, color=colors2[1])
    ax2.spines['right'].set_color(colors2[1])
    plt.tight_layout()
    fig.savefig(f'{figure_folder}potential-waterdensity-2V.tiff', dpi=300)

    fig, ax1 = plt.subplots(1,1, figsize=(3.5,3.25))
    ax2 = ax1.twinx()
    j = 0
    for i, cation in enumerate(cations):
        ax1.plot(-edges[i,j,0], avg_epot[i,j], label=f'{cation}', color=colors1[i])
        ax1.fill_between(-edges[i,j,0], avg_epot[i,j]-std_epot[i,j], avg_epot[i,j]+std_epot[i,j], alpha=0.3, color=colors1[i])

        ax2.plot(-edges[i,j,0], avg_density[i,j]['water'], label=f'{cation}' , color=colors2[i])
        ax2.fill_between(-edges[i,j,0], avg_density[i,j]['water']-std_density[i,j]['water'], avg_density[i,j]['water']+std_density[i,j]['water'], alpha=0.3, color=colors2[i])

    ax1.set_xlabel(r'z ($\mathrm{\AA}$)', fontsize=12)
    ax1.set_ylabel('$\Phi$ (V)', fontsize=12, color=colors1[1])
    # ax2.set_ylabel('Water Density (g/cm³)', fontsize=12, color=colors2[1])
    ax1.grid()

    ax1.set_xlim(-5, 15)
    ax1.set_xticks(np.arange(0, 16, 5))
    ax1.set_xticklabels([str(x) for x in np.arange(0, 16, 5)], fontsize=10)

    ax1.set_ylim(-1.0, 0.5)
    ax1.set_yticks(np.arange(-1.0, 0.6, 0.2))
    ax1.set_yticklabels([str(round(y,1)) for y in np.arange(-1.0, 0.6, 0.2)], fontsize=10, color=colors1[1])
    ax1.spines['left'].set_color(colors1[1])


    ax2.set_ylim(0, 1.8)
    ax2.set_yticks(np.arange(0, 1.81, 0.4))
    ax2.set_yticklabels([str(round(y,1)) for y in np.arange(0, 1.81, 0.4)], fontsize=10, color=colors2[1])
    ax2.spines['right'].set_color(colors2[1])
    plt.tight_layout()
    fig.savefig(f'{figure_folder}potential-waterdensity-0V.tiff', dpi=300)

    return None

def num_nitrate_in_stern(avg_no3_counts, std_no3_counts, figure_folder):
    colors = ["#A27514", "#55bbbb", "#016e76", "#e6c352"]
    markers = ['o', 's', '^', 'D']
    fig, ax = plt.subplots(1,1, figsize=(3.25, 3.25), sharex=True, sharey=True)

    cation_xlabels = [r'Cs$^+$', r'K$^+$', r'Na$^+$', r'Li$^+$']
    xlist = [3, 2, 1, 0]

    for j, pot in enumerate(potentials):
        ax.scatter(xlist, avg_no3_counts[:, j], color=colors[j], marker=markers[j])
        ax.errorbar(xlist, avg_no3_counts[:, j], yerr=std_no3_counts[:, j], fmt='o', color=colors[j], capsize=5)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xticks(xlist)
        ax.set_xticklabels(cation_xlabels, fontsize=12)
        ax.set_ylabel("Number NO$_3^-$ in Stern Layer", fontsize=12)
        ax.grid()
    plt.tight_layout()
    fig.savefig(f'{figure_folder}num-nitrate-stern-layer.tiff', dpi=300)

def residence_time(figure_folder):

    adsorption_decay_avg = np.load(data_folder + "residence_time_analysis/adsorption_decay_avg.npy", allow_pickle=True)
    adsorption_decay_std = np.load(data_folder + "residence_time_analysis/adsorption_decay_std.npy", allow_pickle=True)

    colors = ["#A27514", "#55bbbb", "#016e76", "#e6c352"]
    markers = ['o', 's', '^', 'D']
    fig, ax = plt.subplots(1,3, figsize=(7, 3.25), sharex=True, sharey=True)

    cation_xlabels = [r'Cs$^+$', r'K$^+$', r'Na$^+$', r'Li$^+$']
    xlist = [3, 2, 1, 0]

    for j, pot in enumerate(potentials):
            ax[j].scatter(xlist, adsorption_decay_avg[:, j, 1, 0], label=f"{int(pot)/10} V", color=colors[j], marker=markers[j])
            ax[j].errorbar(xlist, adsorption_decay_avg[:, j, 1, 0], yerr=adsorption_decay_std[:, j, 1, 0], fmt='o', color=colors[j], capsize=5)
            ax[j].set_xlim(-0.5, 3.5)
            ax[j].set_xticks(xlist)
            ax[j].set_xticklabels(cation_xlabels)
            ax[0].set_ylabel("Residence Time (ps)")
            ax[j].grid()

    plt.tight_layout()
    fig.savefig(f'{figure_folder}residence-time.tiff', dpi=300)
    return None

def inplane_diffusion(figure_folder): 
    avg_cation_diff = np.load(data_folder + "2d_diffusion/avg_cation.npy", allow_pickle=True)
    avg_nitrate_diff = np.load(data_folder + "2d_diffusion/avg_nitrate.npy", allow_pickle=True)
    std_cation_diff = np.load(data_folder + "2d_diffusion/std_cation.npy", allow_pickle=True)
    std_nitrate_diff = np.load(data_folder + "2d_diffusion/std_nitrate.npy", allow_pickle=True)

    # z-direction msds
    avg_cation_zdiff = np.load(data_folder + "2d_diffusion/avg_cation_z_msd.npy", allow_pickle=True)
    avg_nitrate_zdiff = np.load(data_folder + "2d_diffusion/avg_nitrate_z_msd.npy", allow_pickle=True)
    std_cation_zdiff = np.load(data_folder + "2d_diffusion/std_cation_z_msd.npy", allow_pickle=True)
    std_nitrate_zdiff = np.load(data_folder + "2d_diffusion/std_nitrate_z_msd.npy", allow_pickle=True)

    # 2d diffusion
    # plot self-diffusion coefficients
    colors = ["#A27514", "#55bbbb", "#016e76", "#e6c352"]
    fig, ax = plt.subplots(1,3, figsize=(7,3.25))

    x_ticks = np.arange(len(cations))
    x_labels = [r'Li$^{+}$', r'Na$ ^{+}$', r'K$^{+}$', r'Cs$^{+}$']


    for l, pot in enumerate(potentials):
        ax[l].plot(x_ticks, avg_cation_diff[::-1,l]*1e5, '--', color=colors[0], label='Cation')
        ax[l].errorbar(x_ticks, avg_cation_diff[::-1,l]*1e5, yerr=std_cation_diff[::-1,l]*1e5, fmt='o', color=colors[0], capsize=5)
        ax[l].plot(x_ticks, avg_nitrate_diff[::-1,l]*1e5, '--', color=colors[1], label='Nitrate')
        ax[l].errorbar(x_ticks, avg_nitrate_diff[::-1,l]*1e5, yerr=std_nitrate_diff[::-1,l]*1e5, fmt='^', color=colors[1], capsize=5)

        ax[l].set_xticks(x_ticks)
        ax[l].set_xticklabels(x_labels, fontsize=12)
        ax[l].set_xlim(-0.5, 3.5)
        ax[0].set_ylabel(r'Self-Diffusion Coeff. ($10^{-5}$ cm$^2$/s)', fontsize=12)
        ax[l].set_ylim(1.0, 2.2)
        # ax[l].set_title(f'Potential: {int(pot)/10} V')
        ax[l].grid()
    # ax[0].legend()
    plt.tight_layout()
    fig.savefig(f'{figure_folder}inplane-diffusion-coefficients.tiff', dpi=300)

    # z-direction msds
    fig, ax = plt.subplots(1,3, figsize=(7,3.25))
    for l, pot in enumerate(potentials):
        ax[l].plot(x_ticks, avg_cation_zdiff[::-1,l], '--', color=colors[0], label='Cation')
        ax[l].errorbar(x_ticks, avg_cation_zdiff[::-1,l], yerr=std_cation_zdiff[::-1,l], fmt='o', color=colors[0], capsize=5)
        ax[l].plot(x_ticks, avg_nitrate_zdiff[::-1,l], '--', color=colors[1], label='Nitrate')
        ax[l].errorbar(x_ticks, avg_nitrate_zdiff[::-1,l], yerr=std_nitrate_zdiff[::-1,l], fmt='^', color=colors[1], capsize=5)

        ax[l].set_xticks(x_ticks)
        ax[l].set_xticklabels(x_labels, fontsize=12)
        ax[l].set_xlim(-0.5, 3.5)
        ax[0].set_ylabel(r'MSD$_z$ (t=10ps) ($\mathrm{\AA}^2$)', fontsize=12)
        # ax[l].set_ylim(1.0, 2.2)
        # ax[l].set_title(f'Potential: {int(pot)/10} V')
        ax[l].grid()
    # ax[0].legend()
    plt.tight_layout()
    fig.savefig(f'{figure_folder}z-direction-msd.tiff', dpi=300)

    return None

def supporting_electrolyte(colors, figure_folder):
    # load density data
    reps = ['0','1','2','3','4','5','6','7','8','9']
    density = np.empty((len(cations),len(reps)), dtype=object)
    edges = np.empty((len(cations),len(reps)), dtype=object)
    cell_width = np.empty((len(cations),len(reps)), dtype=object)

    # compute units
    # determine density profile in g/cm^3
    atom_plot = ['water', 'nitrate', 'cation', 'perchlorate']
    atom_name = ['O', 'N_NO', 'C', 'Cl_ClO']
    Mw = {'water': 18.015, 'nitrate': 62.0049, 'Cs': 132.905, 'K': 39.0983, 'Na': 22.9897, 'Li': 6.94, 'perchlorate': 99.45}
    Na = 6.02214076e23  # Avogadro's number

    for i, cation in enumerate(cations):
        for k, rep in enumerate(reps):
            if cation=='Li' and rep=='1':
                continue
            print(f'Loading {cation} {rep}')
            folder = f'{data_folder}{cation}/constant-potential/supporting-elyte/rep{rep}/'
            density_raw = np.load(f'{folder}density.npy', allow_pickle=True).item()
            edges[i,k] = np.load(f'{folder}edges.npy', allow_pickle=True)

            area = np.load(f'{folder}area.npy', allow_pickle=True)
            n_atoms = np.load(f'{folder}n-atoms.npy', allow_pickle=True).item()
            density[i,k] = {}
            for l, atom in enumerate(atom_plot):
                if atom == 'cation' or atom == 'perchlorate' or atom == 'nitrate':
                    density[i,k][atom] = (density_raw[atom_name[l]] / (area * 1e-3)) * n_atoms[atom_name[l]] # number density
                else:
                    density[i,k][atom] = (density_raw[atom_name[l]] / (area * 1e-24)) * n_atoms[atom_name[l]] * Mw[atom] / Na 

            # find the electrode edge
            idx_electrode = np.where(density_raw['Au'] > 0.1)
            elec_edge_1 = edges[i,k][idx_electrode[0][9]]
            elec_edge_2 = edges[i,k][idx_electrode[0][10]]
            cell_width[i,k] = elec_edge_2 - elec_edge_1
            edges[i,k] = edges[i,k] - elec_edge_2

    # average density profiles
    avg_density = np.empty((len(cations)), dtype=object)
    std_density = np.empty((len(cations)), dtype=object)

    avg_pmf = np.empty((len(cations)), dtype=object)
    std_pmf = np.empty((len(cations)), dtype=object)

    atoms = ['cation', 'nitrate', 'perchlorate', 'water']

    def pmf(density):
        bulk = density[len(density)//2]
        density = density / bulk
        return - np.log(density)

    for i, cation in enumerate(cations):
        if cation=='Li':
            continue

        avg_density[i] = {}
        std_density[i] = {}
        avg_pmf[i] = {}
        std_pmf[i] = {}

        for atom in atoms:
            avg_density[i][atom] = np.mean([density[i,k][atom] for k in range(len(reps))], axis=0)
            std_density[i][atom] = np.std([density[i,k][atom] for k in range(len(reps))], axis=0)
            avg_pmf[i][atom] = np.mean([pmf(density[i,k][atom]) for k in range(len(reps))], axis=0)
            std_pmf[i][atom] = np.std([pmf(density[i,k][atom]) for k in range(len(reps))], axis=0)

    # average lithium separately
    i = 3
    avg_density[i] = {}
    std_density[i] = {}
    avg_pmf[i] = {}
    std_pmf[i] = {}

    for atom in atoms:
        avg_density[i][atom] = np.mean([density[i,k][atom] for k in range(len(reps)) if k != 1], axis=0)
        std_density[i][atom] = np.std([density[i,k][atom] for k in range(len(reps)) if k != 1], axis=0)
        avg_pmf[i][atom] = np.mean([pmf(density[i,k][atom]) for k in range(len(reps)) if k != 1], axis=0)
        std_pmf[i][atom] = np.std([pmf(density[i,k][atom]) for k in range(len(reps)) if k != 1], axis=0)

    # density profile
    # plot charge density profiles
    fig, ax = plt.subplots(1,3, figsize=(7,3), sharey=True)
    species_labels = ['cation', 'nitrate', 'perchlorate']


    for i, cation in enumerate(cations):
        for j, species in enumerate(species_labels):
            ax[j].plot(edges[i,0], avg_density[i][species], label=f'{cation}', color=colors[i])
            ax[j].fill_between(edges[i,0], avg_density[i][species]-std_density[i][species], avg_density[i][species]+std_density[i][species], alpha=0.3, color=colors[i])

            ax[j].set_xlim(1, -26)
            xticks = np.arange(-25, 1, 5)
            xlabels = [str(-x) for x in xticks]
            ax[j].set_xticks(xticks)
            ax[j].set_xticklabels(xlabels, fontsize=10)
            # ax[j].set_title(f'{species.capitalize()}', fontsize=12)
            # ax[j].set_ylim(0, 0.3)
            # ax[j].set_yticks(np.arange(0, 0.1, 0.01))
            # ax[j].set_yticklabels(np.round(np.arange(0, 0.1, 0.01), 3), fontsize=10)

            ax[j].set_xlabel(r'z ($\mathrm{\AA}$)', fontsize=12)
            ax[0].set_ylabel(r'Species Density (1/$\mathrm{nm^3}$)', fontsize=12)
            # ax[0].legend(fontsize=12, ncols=1, loc='lower right')
            if i == 0:
                ax[j].grid()
    plt.tight_layout()

    fig.savefig(f'{figure_folder}supp-elyte-density-profiles-surface.tiff', dpi=300)

    # water - potential comparison
    # load electric field and electrostatic potential data
    efield = np.empty((len(cations),len(reps)), dtype=object)
    epot = np.empty((len(cations),len(reps)), dtype=object)
    cutoffs = np.empty((len(cations),len(reps)), dtype=object)

    for i, cation in enumerate(cations):
        for k, rep in enumerate(reps):
            print(f'Loading {cation} {rep}')
            folder = f'{data_folder}{cation}/constant-potential/supporting-elyte/rep{rep}/'
            efield[i,k] = np.load(f'{folder}electric-field.npy', allow_pickle=True)
            epot[i,k] = np.load(f'{folder}electrostatic-potential.npy', allow_pickle=True)
            cutoffs[i,k] = np.load(f'{folder}cutoffs.npy', allow_pickle=True)

    # average profiles
    avg_efield = np.empty((len(cations)), dtype=object)
    std_efield = np.empty((len(cations)), dtype=object)
    avg_epot = np.empty((len(cations)), dtype=object)
    std_epot = np.empty((len(cations)), dtype=object)
    avg_cutoffs = np.empty((len(cations)), dtype=object)
    std_cutoffs = np.empty((len(cations)), dtype=object)

    for i, cation in enumerate(cations):
        avg_epot[i] = np.mean([epot[i,k] for k in range(len(reps))], axis=0)
        std_epot[i] = np.std([epot[i,k] for k in range(len(reps))], axis=0)
        avg_efield[i] = np.mean([efield[i,k] for k in range(len(reps))], axis=0)
        std_efield[i] = np.std([efield[i,k] for k in range(len(reps))], axis=0)
        avg_cutoffs[i] = np.mean([cutoffs[i,k] for k in range(len(reps))], axis=0)
        std_cutoffs[i] = np.std([cutoffs[i,k] for k in range(len(reps))], axis=0)
        
    fig, ax1 = plt.subplots(1,1, figsize=(3.25,3.25))
    ax2 = ax1.twinx()

    colors1 = ["#0c2944", "#0254A1", "#3d8ee0", "#9acaf7"]
    colors2 = [ "#5e1316",  "#B72D31",  "#D64454",  "#E6A7B8"]

    for i, cation in enumerate(cations):
        ax1.plot(-edges[i,0], avg_epot[i], label=f'{cation}', color=colors1[i])
        ax1.fill_between(-edges[i,0], avg_epot[i]-std_epot[i], avg_epot[i]+std_epot[i], alpha=0.3, color=colors1[i])

        ax2.plot(-edges[i,0], avg_density[i]['water'], label=f'{cation}' , color=colors2[i])
        ax2.fill_between(-edges[i,0], avg_density[i]['water']-std_density[i]['water'], avg_density[i]['water']+std_density[i]['water'], alpha=0.3, color=colors2[i])

    ax1.set_xlabel(r'z ($\mathrm{\AA}$)', fontsize=12)
    ax1.set_ylabel('$\Phi$ (V)', fontsize=12, color=colors1[1])
    ax2.set_ylabel('Oxygen Density (g/cm³)', fontsize=12, color=colors2[1])
    ax1.grid()

    ax1.set_xlim(-5, 15)
    ax1.set_xticks(np.arange(0, 16, 5))
    ax1.set_xticklabels([str(x) for x in np.arange(0, 16, 5)], fontsize=10)

    ax1.set_ylim(-0.5, 1.5)
    ax1.set_yticks(np.arange(-0.5, 1.6, 0.4))
    ax1.set_yticklabels([str(round(y,1)) for y in np.arange(-0.5, 1.6, 0.4)], fontsize=10, color=colors1[1])
    ax1.spines['left'].set_color(colors1[1])


    ax2.set_ylim(0, 1.8)
    ax2.set_yticks(np.arange(0, 1.81, 0.4))
    ax2.set_yticklabels([str(round(y,1)) for y in np.arange(0, 1.81, 0.4)], fontsize=10, color=colors2[1])
    ax2.spines['right'].set_color(colors2[1])
    plt.tight_layout()
    fig.savefig(f'{figure_folder}supp-elyte-potential-waterdensity.tiff', dpi=300)
    return None

def load_density(param, data_folder=data_folder):
    # load density data
    density = np.empty((len(cations),len(potentials),len(reps)), dtype=object)
    pmf_array = np.empty((len(cations),len(potentials),len(reps)), dtype=object)
    edges = np.empty((len(cations),len(potentials),len(reps)), dtype=object)
    cell_width = np.empty((len(cations),len(potentials)), dtype=object)

    # compute units
    # determine density profile in g/cm^3
    atom_plot = ['water', 'nitrate', 'cation', 'H']
    atom_name = ['O', 'N_NO', 'C', 'H']
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
                folder = f'{data_folder}{cation}/constant-potential/{param}/{potential}/rep{rep}/'
                density_raw = np.load(f'{folder}density.npy', allow_pickle=True).item()
                edges[i,j,k] = np.load(f'{folder}edges.npy', allow_pickle=True)

                area = np.load(f'{folder}area.npy', allow_pickle=True)
                n_atoms = np.load(f'{folder}n-atoms.npy', allow_pickle=True).item()
                density[i,j,k] = {}
                pmf_array[i,j,k] = {}

                for l, atom in enumerate(atom_plot):
                    pmf_array[i,j,k][atom] = pmf(density_raw[atom_name[l]])
                    if atom == 'cation' or atom == 'H' or atom == 'nitrate':
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

    atoms = ['cation', 'nitrate', 'water', 'H']

    def pmf(density):
        pmf_raw = -np.log(density)
        bulk = pmf_raw[len(pmf_raw)//2]
        pmf_corrected = pmf_raw - bulk
        return pmf_corrected

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
                    
    return edges, avg_density, std_density, avg_pmf, std_pmf

def load_electric_potential(data_folder=data_folder):
    # load electric field and electrostatic potential data
    efield = np.empty((len(cations),len(potentials),len(reps)), dtype=object)
    epot = np.empty((len(cations),len(potentials),len(reps)), dtype=object)
    cutoffs = np.empty((len(cations),len(potentials),len(reps)), dtype=object)

    for i, cation in enumerate(cations):
        for j, potential in enumerate(potentials):
            for k, rep in enumerate(reps):
                folder = f'{data_folder}{cation}/constant-potential/wca/{potential}/rep{rep}/'
                efield[i,j,k] = np.load(f'{folder}electric-field.npy', allow_pickle=True)
                epot[i,j,k] = np.load(f'{folder}electrostatic-potential.npy', allow_pickle=True)
                cutoffs[i,j,k] = np.load(f'{folder}cutoffs.npy', allow_pickle=True)

    # average profiles
    avg_efield = np.empty((len(cations),len(potentials)), dtype=object)
    std_efield = np.empty((len(cations),len(potentials)), dtype=object)
    avg_epot = np.empty((len(cations),len(potentials)), dtype=object)
    std_epot = np.empty((len(cations),len(potentials)), dtype=object)
    avg_cutoffs = np.empty((len(cations),len(potentials)), dtype=object)

    for i, cation in enumerate(cations):
        for j, potential in enumerate(potentials):
            avg_epot[i,j] = np.mean([epot[i,j,k] for k in range(len(reps))], axis=0)
            std_epot[i,j] = np.std([epot[i,j,k] for k in range(len(reps))], axis=0)
            avg_efield[i,j] = np.mean([efield[i,j,k] for k in range(len(reps))], axis=0)
            std_efield[i,j] = np.std([efield[i,j,k] for k in range(len(reps))], axis=0)
            avg_cutoffs[i,j] = np.mean([cutoffs[i,j,k] for k in range(len(reps))], axis=0)
    return avg_efield, std_efield, avg_epot, std_epot, avg_cutoffs


def load_ionpairing(data_folder=data_folder):
    # import rdf and pmf data
    potentials = ['00','10','20']
    cations = ['Cs', 'K', 'Na', 'Li']
    layers = ['stern', 'diffuse', 'bulk']
    reps = ['0', '1', '2', '3', '4', '5', '6', '7']

    rdf = np.empty((len(cations),len(potentials),len(reps), len(layers)), dtype=object)
    rdf_b = np.empty((len(cations), (len(potentials)*len(reps))), dtype=object)
    rdf_avg = np.empty((len(cations),len(potentials), len(layers)+1), dtype=object)
    rdf_std = np.empty((len(cations),len(potentials), len(layers)+1), dtype=object)

    pmf_d = np.empty((len(cations),len(potentials),len(reps), len(layers)), dtype=object)
    pmf_avg = np.empty((len(cations),len(potentials), len(layers)+1), dtype=object)
    pmf_std = np.empty((len(cations),len(potentials), len(layers)+1), dtype=object)

    num_no3 = np.empty((len(cations),len(potentials),len(reps)), dtype=object)
    num_no3_avg = np.empty((len(cations),len(potentials)), dtype=object)
    num_no3_std = np.empty((len(cations),len(potentials)), dtype=object)

    for i, cation in enumerate(cations):
        for j, potential in enumerate(potentials):
            for l, layer in enumerate(layers):
                for k, rep in enumerate(reps):
                    folder = f'{data_folder}{cation}/constant-potential/wca/{potential}/rep{rep}/layer-analysis/{layer}/'
                    bins = np.load(f'{folder}bins.npy', allow_pickle=True)
                    rdf[i,j,k,l] = np.load(f'{folder}rdf.npy', allow_pickle=True)
                    pmf_d[i,j,k,l] = np.load(f'{folder}pmf.npy', allow_pickle=True)

                    if l == 2:
                        rdf_b[i, j*len(reps)+k] = rdf[i,j,k,l]

                    if l == 0:
                        new_folder = f'{data_folder}{cation}/constant-potential/wca/{potential}/rep{rep}/layer-analysis/'
                        num_no3[i,j,k] = np.load(f'{new_folder}no3_counts.npy', allow_pickle=True)[0]

                rdf_avg[i,j,l] = np.mean([rdf[i,j,k,l] for k in range(len(reps))], axis=0)
                rdf_std[i,j,l] = np.std([rdf[i,j,k,l] for k in range(len(reps))], axis=0)

                pmf_avg[i,j,l] = np.mean([pmf_d[i,j,k,l] for k in range(len(reps))], axis=0)
                pmf_std[i,j,l] = np.std([pmf_d[i,j,k,l] for k in range(len(reps))], axis=0)

            # average number of nitrate in stern layer
            num_no3_avg[i,j] = np.mean([num_no3[i,j,k] for k in range(len(reps))], axis=0)
            num_no3_std[i,j] = np.std([num_no3[i,j,k] for k in range(len(reps))], axis=0)

    rdf_b_avg = np.empty((len(cations)), dtype=object)
    rdf_b_std = np.empty((len(cations)), dtype=object)

    for i, cation in enumerate(cations):
        rdf_b_avg[i] = np.mean([rdf_b[i,j] for j in range(rdf_b.shape[1])], axis=0)
        rdf_b_std[i] = np.std([rdf_b[i,j] for j in range(rdf_b.shape[1])], axis=0)
    return bins, rdf_avg, rdf_std, pmf_avg, pmf_std, rdf_b_avg, rdf_b_std, num_no3_avg, num_no3_std

def read_and_plot_metal_comparison():
    cations = ['Cs', 'K', 'Na', 'Li']
    potentials = ['00','10', '20']
    reps_lj = ['0', '1', '2']
    reps_wca = ['0','1','2','3','4', '5', '6', '7', '8', '9']

    def load_data(param, reps):
        # load density data for lj_raw
        density = np.empty((len(cations),len(potentials),len(reps)), dtype=object)
        density_wu = np.empty((len(cations),len(potentials),len(reps)), dtype=object)
        edges = np.empty((len(cations),len(potentials),len(reps)), dtype=object)

        pmf_array = np.empty((len(cations),len(potentials),len(reps)), dtype=object)

        # determine density profile in g/cm^3
        atom_plot = ['water', 'nitrate', 'cation']
        atom_name = ['O', 'N_NO', 'C']
        Mw = {'O': 18.015, 'N_NO': 62.0049}
        Na = 6.02214076e23  # Avogadro's number

        def pmf(density):
            pmf_raw = -np.log(density)
            bulk = pmf_raw[len(pmf_raw)//2]
            pmf_corrected = pmf_raw - bulk
            return pmf_corrected

        for i, cation in enumerate(cations):
            for j, potential in enumerate(potentials):
                for k, rep in enumerate(reps):
                    folder = f'{data_folder}{cation}/constant-potential/{param}/{potential}/rep{rep}/'
                    density[i,j,k] = np.load(f'{folder}density.npy', allow_pickle=True).item()
                    edges[i,j,k] = np.load(f'{folder}edges.npy', allow_pickle=True)
                    area = np.load(f'{folder}area.npy', allow_pickle=True)
                    n_atoms = np.load(f'{folder}n-atoms.npy', allow_pickle=True).item()
                    density_wu[i,j,k] = {}
                    pmf_array[i,j,k] = {}
                    for a, atom in enumerate(atom_plot):
                        pmf_array[i,j,k][atom] = pmf(density[i,j,k][atom_name[a]])
                        # convert to g/cm^3
                        if atom == 'cation' or atom == 'nitrate':
                            density_wu[i,j,k][atom] = (density[i,j,k][atom_name[a]] / (area * 1e-3)) * n_atoms[atom_name[a]] # * Mw[cation] / Na
                        else:
                            density_wu[i,j,k][atom] = (density[i,j,k][atom_name[a]] / (area * 1e-24)) * n_atoms[atom_name[a]] * Mw[atom_name[a]] / Na


                    # find the electrode edge
                    idx_electrode = np.where(density[i,j,k]['Au'] > 0.1)
                    elec_edge_1 = edges[i,j,k][idx_electrode[0][9]]
                    elec_edge_2 = edges[i,j,k][idx_electrode[0][10]]

                    edges[i,j,k] = edges[i,j,k] - elec_edge_2

        # average density profiles
        avg_density= np.empty((len(cations),len(potentials)), dtype=object)
        std_density= np.empty((len(cations),len(potentials)), dtype=object)
        avg_pmf = np.empty((len(cations),len(potentials)), dtype=object)
        std_pmf = np.empty((len(cations),len(potentials)), dtype=object)

        atom_plot = ['water', 'nitrate', 'cation']

        for i, cation in enumerate(cations):
            for j, potential in enumerate(potentials):
                avg_density[i,j] = {}
                std_density[i,j] = {}
                avg_pmf[i,j] = {}
                std_pmf[i,j] = {}
                for atom in atom_plot:
                    avg_density[i,j][atom] = np.mean([density_wu[i,j,k][atom] for k in range(len(reps))], axis=0)
                    std_density[i,j][atom] = np.std([density_wu[i,j,k][atom] for k in range(len(reps))], axis=0)
                    avg_pmf[i,j][atom] = np.mean([pmf_array[i,j,k][atom] for k in range(len(reps))], axis=0)
                    std_pmf[i,j][atom] = np.std([pmf_array[i,j,k][atom] for k in range(len(reps))], axis=0)
        return edges[:,:,0], avg_density, std_density, avg_pmf, std_pmf

    # load all data
    params = ['wca', 'lj_a', 'lj']
    avg_density= np.empty((len(params),len(cations),len(potentials)), dtype=object)
    std_density= np.empty((len(params),len(cations),len(potentials)), dtype=object)
    avg_pmf = np.empty((len(params),len(cations),len(potentials)), dtype=object)
    std_pmf = np.empty((len(params),len(cations),len(potentials)), dtype=object)
    edges = np.empty((len(params),len(cations),len(potentials)), dtype=object)

    for p, param in enumerate(params):
        if param == 'wca':
            reps = reps_wca
        else:
            reps = reps_lj
        edges[p], avg_density[p], std_density[p], avg_pmf[p], std_pmf[p] = load_data(param, reps)

    def plot_property(avg_property, std_property, prop_label, params, ylims, ylabel, colors, figname):
        fig, ax = plt.subplots(3,3, figsize=(7,5), sharex=True)
        pot_labels = ['0.0 V', '1.0 V', '2.0 V']

        for i, cation in enumerate(cations):
            for j, pot in enumerate(pot_labels):
                for p, param in enumerate(params):
                    ax[p, j].plot(-edges[p,i,j], avg_property[p][i,j][prop_label], color=colors[i])
                    ax[p, j].fill_between(-edges[p,i,j], avg_property[p][i,j][prop_label]-std_property[p][i,j][prop_label], avg_property[p][i,j][prop_label]+std_property[p][i,j][prop_label], alpha=0.3, color=colors[i])
                    

                    if i == 0:
                        # print(f'p: {p}, j: {j}, ylims: {ylims[p]}')
                        ax[p, j].set_xlim(-5, 15)
                        ax[p,j].set_ylim(ylims[p][0], ylims[p][1])
                        if j == 0:
                            ax[p,j].set_ylabel(ylabel, fontsize=11)
                        if p == len(params)-1:
                            ax[p,j].set_xlabel(r'z ($\mathrm{\AA}$)', fontsize=11)
                        ax[p,j].grid()

        plt.tight_layout()
        fig.savefig(f'{figure_folder}{figname}.tiff', dpi=300)
        return None
    
    colors = ["#1c4f7e", "#E05656",  "#4db4e8", "#8d0a0e"]

    # cation density
    ylims_density = [(0, 2.0), (0, 2.0), (0, 9.0)]
    plot_property(avg_density, 
                  std_density, 
                  'cation',
                  params, 
                  ylims_density, 
                  r'Density (1/$\mathrm{\AA^3}$)', 
                  colors, 
                  'cation-density-comparison')
    
    # water density
    plot_property(avg_density, 
                  std_density, 
                  'water',
                  params, 
                  ylims_density, 
                  r'Density (1/$\mathrm{\AA^3}$)', 
                  colors, 
                  'water-density-comparison')
    
    # nitrate density
    ylims_density = [(0, 2.5), (0, 2.5), (0, 15)]
    plot_property(avg_density, 
                  std_density, 
                  'nitrate',
                  params, 
                  ylims_density, 
                  r'Density (1/$\mathrm{\AA^3}$)', 
                  colors, 
                  'nitrate-density-comparison')
    
    # nitrate pmf
    ylims_pmf = [(-2, 6), (-2, 6), (-4, 8)]
    plot_property(avg_pmf, 
                  std_pmf, 
                  'nitrate',
                  params, 
                  ylims_pmf, 
                  r'Free Energy (k$_{\mathrm{B}}$T)', 
                  colors, 
                  'nitrate-pmf-comparison')
    



def main():
    load_density_param = 'wca'
    edges, avg_density, std_density, avg_pmf, std_pmf = load_density(load_density_param)
    avg_efield, std_efield,  avg_epot, std_epot, avg_cutoffs = load_electric_potential()
    bins, rdf_avg, rdf_std, pmf_avg, pmf_std, rdf_b_avg, rdf_b_std, num_no3_avg, num_no3_std = load_ionpairing()

    colors = ["#1c4f7e", "#E05656",  "#4db4e8", "#8d0a0e"]
    colors2 =  ["#A27514", "#55bbbb", "#016e76", "#e6c352"]

    # generate figures
    nitrate_density(edges, avg_density, std_density, colors, figure_folder)
    nitrate_free_energy(edges, avg_pmf, std_pmf, colors, figure_folder)
    electric_field(edges, avg_epot, std_epot, colors, figure_folder)
    # full_density(edges, avg_density, std_density, colors2, cations, figure_folder)
    rdf_layers(bins, rdf_avg, rdf_std, colors, figure_folder)
    rdf_bulk(colors, figure_folder)
    pmf_layers(bins, pmf_avg, pmf_std, colors, figure_folder)
    cutoffs(edges, avg_epot, std_epot, avg_cutoffs, figure_folder, colors)
    hydrogen_density_profiles(edges, avg_density, std_density, figure_folder, colors)
    one_pot_water_density(edges, avg_epot, std_epot, avg_density, std_density, figure_folder)
    all_pot_water(edges, avg_epot, std_epot, avg_density, std_density, figure_folder)
    num_nitrate_in_stern(num_no3_avg, num_no3_std, figure_folder)
    residence_time(figure_folder)
    inplane_diffusion(figure_folder)
    supporting_electrolyte(colors, figure_folder)

    # metal parameters comparison
    read_and_plot_metal_comparison()

if __name__ == "__main__":
    main()