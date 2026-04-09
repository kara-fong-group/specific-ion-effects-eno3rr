# --------------------------------
# script to run the electrostatic toy model and save the results
# Madeline Murphy - 2025
# --------------------------------

import numpy as np
import matplotlib.pyplot as plt

# define constants
e = 1.602e-19 #C
k = 1 / (4 * np.pi * 8.854e-12* 80) #Nm^2/C^2 for water at room temperature
kT = 298.15 * 1.3806e-23


def electric_field_map(figname):
    # Create grid for field magnitude
    nx, ny = 200, 200
    k = 8.99e9  # Coulomb's constant in N m²/C²
    x = np.linspace(-5e-10, 5e-10, nx)
    y = np.linspace(-5e-10, 5e-10, ny)
    X, Y = np.meshgrid(x, y)

    bulk_free = [(-1, (0,0))]  # single negative charge at origin
    bulk_paired = [(-1, (0,0)), (+1, (1.e-10,1.e-10))]  # negative charge at origin, positive image charge at (1.5ang,1.5ang)

    image_free = [(-1, (1.0e-10,0)), (+1, (-1.0e-10,0)) ]
    image_paired = [(-1, (1.0e-10,0)), (+1, (-1.0e-10,0)), (+1, (2.0e-10,1.0e-10)), (-1, (-2.0e-10,1.0e-10))]  

    charges = [bulk_free, bulk_paired, image_free, image_paired]

    fig, axes = plt.subplots(2,2, figsize=(5, 4.5), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, charge in enumerate(charges):
        ax = axes[i]
        Ex, Ey = np.zeros_like(X), np.zeros_like(Y)
        for q, (xq, yq) in charge:
            dx = X - xq
            dy = Y - yq
            r2 = dx**2 + dy**2
            r2[r2 == 0] = 1e-12  # avoid div by zero
            Ex += q * k * dx / r2
            Ey += q * k * dy / r2

        E_mag = np.sqrt(Ex**2 + Ey**2)

        heat = ax.imshow(E_mag, extent=[-5e-10, 5e-10, -5e-10, 5e-10], origin="lower",
                        cmap="bone", norm=plt.cm.colors.LogNorm(vmin=5e17, vmax=5e20))
        last_heat = heat  # for colorbar later

        # Plot charges
        for q, (xq, yq) in charge:
            if q > 0:
                ax.scatter(xq, yq, c="red", s=10, marker="o", edgecolors="k")
            else:
                ax.scatter(xq, yq, c="blue", s=10, marker="o", edgecolors="k")

        ax.set_aspect(1.0)
        ax.set_xlim(-5e-10, 5e-10)
        ax.set_ylim(-5e-10, 5e-10)
        ticks = np.linspace(-5e-10, 5e-10, 3)
        ax.set_xticks(ticks)
        tick_labels = [f'-5', '0', f'5']
        ax.set_xticklabels(tick_labels, fontsize=12)

        ticks = np.linspace(-5e-10, 5e-10, 3)
        ax.set_yticks(ticks)
        tick_labels = [f'-5', '0', f'5']
        ax.set_yticklabels(tick_labels, fontsize=12)

    axes[0].set_ylabel(r"y, $\mathrm{\AA}$", fontsize=14)
    axes[3].set_xlabel(r"z, $\mathrm{\AA}$", fontsize=14)

    axes[2].set_xlabel(r"z, $\mathrm{\AA}$", fontsize=14)
    axes[2].set_ylabel(r"y, $\mathrm{\AA}$", fontsize=14)
    plt.tight_layout()

    fig.savefig(figname, dpi=300)

    # add a colorbar separately
    fig, ax = plt.subplots(1,1, figsize=(6.0,3.5))
    cbar = fig.colorbar(last_heat, ax=ax, orientation="vertical", aspect=10, pad=0.04)
    cbar.set_label("Electric field strength |E| (N/C)", fontsize=14)
    plt.tight_layout()
    fig.savefig("../../figures/ms/efield_magnitude_colorbar.tiff", dpi=300)
    return None

def energy_vs_distance_plot(figname):
    def u_energy(a, d, theta, q):
        energy = k * q**2 * (- 2/d - 1/(2*a) - 1/(2*a + d*np.cos(theta)) + 2/(4*a**2 + 4*a*d*np.cos(theta) + d**2)**(1/2))
        return energy / kT

    def integrate_energy(a, d, q, theta_bounds=(0, np.pi)):
        theta = np.linspace(theta_bounds[0], theta_bounds[1], 1000)
        
        energy_values = u_energy(a, d, theta, q) * np.sin(theta)  # multiply by sin(theta) for integration over spherical coordinates
        integral = (1/2) * np.trapz(energy_values, theta)
        return integral
    
    # distance of the pair from the plate, for different separations
    q = e

    d_ref = 8.0  # reference distance in Angstroms
    d_ref_m = d_ref * 10**(-10)  # reference distance in meters
    d_pair = 3.5  # pair distance in Angstroms
    d_pair_m = d_pair * 10**(-10)  # pair distance in meters

    a_array = np.linspace(4, 30, 100)  # distances in Angstroms
    u_array = np.zeros((len(a_array)))

    fig, ax = plt.subplots(1,1, figsize=[3,3.25])
    colors = plt.get_cmap('tab10').colors  # use a colormap for colors


    for j, a in enumerate(a_array):
        a_m = a*10**(-10)
        theta_bounds = (0, np.pi)  # cut range for theta
        u_ref = integrate_energy(a_m, d_ref_m, q, theta_bounds)
        u = integrate_energy(a_m, d_pair_m, q, theta_bounds) 

        # print(f"a={a}, reference energy: {np.round(u_ref, 2)} kT, pair energy: {np.round(u, 2)} kT")
        u -= u_ref  # subtract reference energy
        u_array[j] = u

    ax.plot(a_array, u_array[:], color='k', linewidth=2.5)
    # ax.axvline(0, color='k', linewidth=2)

    ax.grid()
    ax.set_xlabel(r'z, $\mathrm{\AA}$', fontsize=12)
    ax.set_ylabel(r'$\Delta E_{\mathrm{pair}}$, $\mathrm{k_B T}$', fontsize=12)
    ax.set_ylim(-2.4, -1.6)
    ax.set_xlim((0, 25)) 

    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    return None

def main():
    electric_field_map("../../figures/ms/efield_magnitude_varied_separation.tiff")
    energy_vs_distance_plot("../../figures/ms/energy_vs_distance.tiff")
    return None

if __name__ == "__main__":
    main()
