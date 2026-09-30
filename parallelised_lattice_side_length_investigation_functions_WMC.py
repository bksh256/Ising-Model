# Calculations 
import numpy as np

# Constants
from scipy.constants import mu_0

# File reading and config
import os

# Make the random 2D lattice 
def make_random_2D_lattice(lattice_side_length, generator):
    
    """Produces a randomly generated 2D lattice of +1 or -1 spins using a specific random seed and numpy's random number generator."""
    
    # PCG64DXSM is the most random number generator available in numpy.
    # True randomness is achieved by passing no seed - Seed is therefore sourced from the operating system
    
    random_2D_lattice = generator.choice([1, -1], size=(lattice_side_length, lattice_side_length))

    return random_2D_lattice

# Function to calculate per spin lattice energy
def calculate_lattice_energy_per_spin(lattice, lattice_side_length, exchange_constant_J, magnetic_field_B, u_b):

    """Calculates the total energy of the entire lattice using the same formula as the energy used to calculate a spin flip, just
    applied to the entire lattice than a specific set of spin coordinate"""

    # Uses np.roll to get periodic summation - Vectorised method of calculating the energy which is much faster than nested for loops
    periodic_boundary_sum = np.roll(lattice, 1, axis=0) + np.roll(lattice, -1, axis=0) + np.roll(lattice, 1, axis=1) + np.roll(lattice, -1, axis=1)

    # Nearest neighbour term in the hamiltonian
    nearest_neighbour_term = (-exchange_constant_J/2)*np.sum(lattice*periodic_boundary_sum)

    # Magnetic term in the hamiltonian
    magnetic_term = u_b*np.sum(magnetic_field_B*lattice)

    # Total lattice energy is given by the subtraction of the magnetic term from the hamiltonian term
    total_lattice_energy = nearest_neighbour_term - magnetic_term

    # Calculating the energy per spin
    energy_per_spin = total_lattice_energy / (lattice_side_length**2)

    return energy_per_spin

# Function to compute a wolff monte carlo step
def WMC_step(lattice_side_length, random_2D_lattice, magnetic_field_B, temperature, k_b, u_b, exchange_constant_J, generator):

    """Performs a Wolff monte carlo step as seen on page 5 of G. T. Barkema and M. E. J. Newman 1997.

    Grows a cluster within that lattice, and then decides whether to flip an entire cluster or not rather than a singluar spin.
    
    Copy-pasted text from page 5:
    
    1) Choose a seed spin at random from the lattice.
    
    2) Look in turn at each of the neighbours of that spin. If they are pointing
    in the same direction as the seed spin, add them to the cluster with
    probability Padd = 1− e−2βJ. 
    
    3) For each spin that was added in the last step, examine each of its neigh-
    bours to find the ones which are pointing in the same direction and add
    each of them to the cluster with the same probability Padd. (Notice that
    as the cluster becomes larger we may find that some of the neighbours
    are already members of the cluster, in which case obviously you don’t
    have to consider adding them again. Also, some of the spins may have
    been considered for addition before, as neighbours of other spins in the
    cluster, but rejected. In this case, they get another chance to be added to
    the cluster on this step.) This step is repeated as many times as necessary
    until there are no spins left in the cluster whose neighbours have not been
    considered for inclusion in the cluster.
    
    4) Flip all the spins in the cluster."""
    
    ## Cluster-growth probability from equation 5 in the 1997 paper ##

    # Probability a spin is added to the cluster as it grows in the wolff algorithm
    beta = 1/(k_b*temperature)
    p_add = 1 - np.exp(-2*beta*exchange_constant_J)

    ## Choosing a random spin, re-used code from the MMC_step ##

    # Set of random coordinates
    x_coord = generator.integers(0, lattice_side_length)
    y_coord = generator.integers(0, lattice_side_length)

    # Selecting the spin
    cluster_origin_spin = random_2D_lattice[x_coord, y_coord]

    ## Making the cluster ##

    # Cluster cannot contain duplicates, so a set is used (the set datatype CANNOT contain duplicates).
    cluster = set()

    # Adding the initial two spins to the cluster
    cluster.add((x_coord, y_coord))

    # Making a list of spins that are to be compared in the cluster to see if new spins can be added to it.
    # Spins are checked later to see if they can be added based 
    cluster_neighbour_comparison_spins_list = [(x_coord, y_coord)]

    # Nearest neighbour boundary conditions
    nearest_neighbours_conditions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    ## Cluster growth loop ##

    # When this list empties itself at the end of the loop, the cluster is fully grown
    while cluster_neighbour_comparison_spins_list:

        # Removing the last element off the comparison list, so that its neighbours can be checked.
        initial_x_coord, initial_y_coord = cluster_neighbour_comparison_spins_list.pop()

        # Growing the cluster (adding spins to it based on the positions and spin orientation)
        for x_coord_add, y_coord_add in nearest_neighbours_conditions:

            # Coordinates to add to the cluster
            n_x_coord_to_add = (initial_x_coord + x_coord_add) % lattice_side_length
            n_y_coord_to_add = (initial_y_coord + y_coord_add) % lattice_side_length

            # Growing the cluster based on neighbours if the spins are:
            # 1: Neighbours are not already in the cluster
            # 2: Only has the same z axis orientation as the original  
            if (n_x_coord_to_add, n_y_coord_to_add) not in cluster:

                if random_2D_lattice[n_x_coord_to_add, n_y_coord_to_add] == cluster_origin_spin:

                    # Add to cluster with the previously calculated probability p_add
                    if generator.random() < p_add:

                        # Adding non-duplicate elements to the cluster
                        cluster.add((n_x_coord_to_add, n_y_coord_to_add))

                        # Adding to neighbour comparison list
                        cluster_neighbour_comparison_spins_list.append((n_x_coord_to_add, n_y_coord_to_add))

    # Flip the entire cluster
    for x_coord_to_flip, y_coord_to_flip in cluster:
        
        random_2D_lattice[x_coord_to_flip, y_coord_to_flip] *= -1

    return random_2D_lattice

# Worker function for running an investigation simulation at a pre-defined temperature
def run_WMC_lattice_side_length_investigation_worker(args):

    """Runs a single wolff algorithm simulation at a pre-defined temperature, and takes its arguements as one object so this function 
        can be run in parallel across N cores across a temperature sweep, so multiple temperatures are investigated at the same time 
        by running this function multiple times using multiprocessing. """

    # Worker random generator
    generator = np.random.Generator(np.random.PCG64DXSM())

    # Extracting arguements from args object
    # Simulation parameters as a tuple so it will work with mutliple of these functions running in parallel
    (temperature, lattice_side_length, magnetic_field_B, exchange_constant_J, k_b, u_b, nsteps) = args
    
    # Initialising a random 2D lattice
    initial_lattice = make_random_2D_lattice(lattice_side_length, generator)
    
    # Creating the lattice used for the ising model simulation itself, leaving the initial lattice separate and unchanged.
    lattice = initial_lattice.copy()
    
    # Burn-in limit for calculations of variances - This is to account for equilibration
    burnin_step_proportion = 0.6
    burnin_equilibration_steps_threshold = int(burnin_step_proportion*nsteps)

    # Lists to store magnetisations, magnetic susceptibilities and heat capacity
    equilibrated_lattice_energy_per_spin_list = []
        
    ### MAIN SIMULATION LOOP ###
    for i in range(nsteps):

        # Reassigning the lattice with every step, making the ising model run. 
        lattice = WMC_step(lattice_side_length, lattice, magnetic_field_B, temperature, k_b, u_b, exchange_constant_J, generator)
            
        # Only begin the welford loop after equilibration - Past the burnin threshold.
        if i > burnin_equilibration_steps_threshold:

            # Calling functions on the lattice every timestep to calculate the quantities of interest
            lattice_energy_per_spin = calculate_lattice_energy_per_spin(lattice, lattice_side_length, exchange_constant_J, magnetic_field_B, u_b)

            # Appending function outputs to lists after the burn-in
            equilibrated_lattice_energy_per_spin_list.append(lattice_energy_per_spin)

    # Turning lists into arrays
    equilibrated_energy_array = np.array(equilibrated_lattice_energy_per_spin_list)

    ## Computing thermodynamic quantities after the burn in ##

    # Energy burned-in variance
    energy_burned_in_variance = np.var(equilibrated_energy_array)

    # Heat capacity
    heat_capacity_burned_in = ((lattice_side_length**2)/(k_b*(temperature**2)))*energy_burned_in_variance

    return heat_capacity_burned_in