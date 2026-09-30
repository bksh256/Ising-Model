-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Introduction
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

This code is a python3 jupter notebook that simulates the 2D, 2-spin Ising Model using both the Wolff and Metropolis algorithms and provides numerical validations of the Ising Model's critical exponents and critical temperature, as well as plots and graphical lattice diagrams made using matplotlib to show how the lattice's properties vary with the initial conditions the system is simulated under. Please see the LaTeX report pdf for a detailed explanation of the code and the scientific results.

I wrote this code as my MPhys Computational Physics final programming project at the University of Exeter in March 2026. This repository contains all the python files, parameter files, final report and feedback excel spreadsheet received after coursework submission.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Operating Systems and CPU architecture dependencies and requirements
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Any version of Windows, MacOS or Linux capable of running python3 will work. This code has been tested on both x86 Windows, x86 Linux and ARM Apple Silicon CPU architectures. I haven't yet been able to test it on ARM Windows or ARM Linux, but if the python install is natively compiled for your CPU architecture and operating system there is a very good chance it will work just fine.

WARNING: It is not recommended to run this code through a python install that is running through an emulation or real-time translation layer such as Parallels, Apple Rosetta 2 or Wine since this will cause performance slowdowns or even numerical errors. The multiprocessing implementation used for validation of the critical temperature, critical exponents and critical amplitude is extremely demanding and will use all of the logical threads and CPU resources present on the CPU for running parallelised simulations. Please run through a python install that is natively compiled for your CPU architecture and operating system.

The sole exception to this rule is running it under WSL2 in Windows since this uses a genuine Linux image designed for the host CPU architecture (x86 and ARM supported). 

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Python module dependencies (Minimum versions)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

python >= 3.10.0 (recommended)

For python 3.10 and above:

NumPy >= 1.21.3
SciPy >= 1.7.3
matplotlib >= 3.5.0
tqdm >= 4.36.0
Jupyter Notebook >= 6.4.5

Anaconda or miniconda are highly recommended, since these tools allow you to install jupyter notebook easily, and they are free.

WARNING: The .py multiprocessing worker files and all the .txt parameter files downloaded from GitHub must be present in the same directory / folder on your system for the code to work correctly. The code is designed to throw errors and stop running if it detects any of the parameter files are missing.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Running the code
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

To run the code, please install Jupyter Notebook inside your python or conda environment. 

Method 1: Use anaconda / miniconda, open the .ipynb file using Jupyter Notebook (Ising_Model_Metropolis_and_Wolff_Monte_Carlo.ipynb) and then run from the GUI: Kernel > Restart and run all cells.

Method 2: Alternatively if you don't want to use anaconda / miniconda use these commands to start jupyter and open the notebook :

cd /path/to/Ising_Model_Metropolis_and_Wolff_Monte_Carlo.ipynb
jupyter notebook Ising_Model_Metropolis_and_Wolff_Monte_Carlo.ipynb

Then run from the GUI: Kernel > Restart and run all cells.

Method 3: If you just want to run the notebook from the terminal without the GUI:

cd /path/to/Ising_Model_Metropolis_and_Wolff_Monte_Carlo.ipynb
jupyter execute Ising_Model_Metropolis_and_Wolff_Monte_Carlo.ipynb

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Critical Temperature Validation
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The code is designed to validate the Ising Model by using both of these numerical Monte-Carlo methods to calculate Onsager's landmark 1944 solution for the critical temperature (2.269K) using four different methods:

Method 1: Finding the midpoint between the two temperatures where the magnetisation drop is the highest.

Method 2: The temperature at which the magnetic susceptibility of the entire 2D lattice peaks.

Method 3: The temperature at which the heat capacity per spin of the 2D lattice peaks.

Method 4: Average of the previous 3 methods to get a statistically averaged result for the critical temperature of the simulated systems from multiple methods.

This is done by running hundreds of Ising Model simulations with different temperatures and analysing the results. This is where multiprocessing comes in handy: Each logical thread on the CPU can spawn a single Ising Model simulation, and this is repeated using a for loop to spawn one simulation on each thread, each with a different temperature until the temperature range is exhausted and the critical temperature can be calculated. The lower the temperature step in the parameter file and the higher the amount of Ising model simulations run, the more precise the value for the critical temperature will be given the temperature range steps over Onsager's known value. This procedure, using multiprocessing to perform different runs at different temperatures simultaneously across all available logical threads

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Critical Exponent Validation
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The three critical exponents, Alpha, Beta and Gamma, are either calculated or verified (especially in the case of the Alpha exponent) using the same variable temperature multiprocessing runs as the critical temperature.

This is done using these methods:

Alpha: Onsager's 1944 result for the critical exponent Alpha is 0. Attempting to calculate this directly would not only be very computationally expensive because the parallel simulation resolution would need to be very high (low temperature step of 0.001K or below, large lattice sizes and 500 or more parallelised runs in total being batch processed across all threads), but this would also leave behind residuals. So instead the code verifies the behaviour that confirms that Alpha is 0. This is done by plotting the natural logarithm of reduced temperature against the heat capacity. If the gradient is the negative of the Onsager Critical Amplitude (see next section in README), then Alpha is 0 and the code is behaving as expected. Please see section 3.1 part 13 of the report for a detailed explanation of the mathematical reasoning behind how Alpha is validated.

Beta: Onsager's 1944 result for the critical exponent Beta is 1/8, and Beta is only valid below the critical temperature. The code attempts to calculate Beta by extracting all the temperatures below the critical temperature (tcrit) to within 1% of tcrit and then calculating the linear regression gradient of the natural logarithm of the absolute reduced temperatures, and the natural logarithm of the magnetisations all below the critical temperature. The linear regress fit is plotted as a gradient comparison plot: The calculated gradient from the linear regress fit is plotted alongside a line with a gradient of precisely 1/8 to show the deviation between the theoretical and numerically calculated results.

Gamma: Onsager's 1944 result for the critical exponent Gamma is -7/8. Gamma is valid both above and below the critical temperature, but not at the critical temperature itself because the phase change occurs there. The exact same method used to calculate Beta is applied to Gamma, but repeated for 1% safety thresholds both above and below the critical temperature. Gamma is calculated using the same linear regression fit method but rather than using the magnetisation only below tcrit, the magnetic susceptibility is used instead and it is done above and below the critical temperature with 1% safety thresholds. Separate gradients are computed for both regimes and then averaged. The gradient calculation cannot go over tcrit itself because this would cause the natural logarithm calculation to converge into a singularity due to the reduced temperature being 0.

These methods are done for both the Metropolis-Hastings and Wolff Algorithms, so the results from the different algorithms can be compared.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calculating the Onsager Critical Amplitude
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The Onsager Critical Amplitude is a coefficient relating the heat capacity C and the logarithm of the lattice size. Due to the laws of logarithms (please see section 2.9 in the report for the mathematical reasoning) if the heat capacity is related to the lattice side length (so sqrt(num_of_spins)) the gradient of a graph showing the logarithm of lattice size against heat capacity will be exactly half of that of the Onsager Critical Amplitude. In order to validate Onsager's Critical Amplitude, the code uses the same multiprocessing method as before but rather than spawning simulations with a different temperature, the lattice size is changed. The logarithm of the lattice size is then plotted against the heat capacity, and the value of half of the Onsager Critical Amplitude is then extracted from a linear regression fit. 

This method are done for both the Metropolis-Hastings and Wolff Algorithms, so the results from the different algorithms can be compared.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Visualising the effect of changing the Initial Conditions
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The next section of the code is designed for visualising the effects of changing the following initial conditions four times (so where applicable, two small and two large values for both positive and negative parameters), using both of the Metropolis Hastings and Wolff Algorithms. 

1. Lattice size.
2. Temperature.
3. Exchange Interaction Constant J.
4. Magnetic field.

In order to show convergence, the graphs for the per-spin energy, per-spin magnetisation, per-spin lattice internal magnetic field, per-spin magnetic susceptibility and per-spin heat capacity are plotted with respect to the timestep. In the case of the quantities that rely on variance, the running variance has to be computed because this is the only way to show how variance, or a quantity dependent on it varies with time. This is done in the code using Welford's Online Algorithm. The quantities that depend on variance that are being visualised are the heat capacity and magnetic susceptibility. Please see section 3.1, parts 11 and 12 of the report to see a detailed explanation of how the data visualisation code works.

NOTE: Whilst this procedure is repeated for both the Metropolis-Hastings and the Wolff Algorithms, only the Metropolis-Hastings algorithm was designed to account for a magnetic induction field, something the Wolff Algorithm lacks despite the Wolff Algorithm being superior for quickly equilibrating a lattice and escaping from localised optima, which the Metropolis Algorithm struggles with.
