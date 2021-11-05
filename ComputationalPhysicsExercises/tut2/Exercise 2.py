import numpy as np
import random
import math
import mpmath
import matplotlib.pyplot as plt
from numpy.lib.function_base import append
from scipy.special import ellipk

def single_sweep(initial_config,N,J,T,h):
    initial_config_energy = energy(initial_config, N, J,h) # we take the energy of initial configuration
    # creation of empty arrays to store the observables 
    magnetization = np.array([])
    Energy=np.array([])
    avg_Energy=np.array([])

    #Metropolis-Hastings algorithm 
    for sweep in range(N**2):
        row,column=random.randint(0,N-1),random.randint(0,N-1) #pick a random site
        new_config=np.copy(initial_config)                     #create a new configuration so we can manipulate it
        new_config[row][column] = -(new_config[row][column])   #we sweep
        new_config_energy=energy(new_config,N,J,h)             #we calculate the energy of the new config 
        delta_s=(new_config_energy/T) - (initial_config_energy/T) #we calculate DeltaS

        if  delta_s < 0: #if DeltaS is less than 0 
            magnetization=np.append(magnetization,np.sum(new_config))              #we save all observables needed
            Energy=np.append(Energy,math.exp(-new_config_energy/T))
            avg_Energy=np.append(avg_Energy,avg_energy_per_site(new_config,N,J,h))
            initial_config=np.copy(new_config)
        else:    #else
            y=random.uniform(0,1)          # we impliment the accept reject method 
            if y<= math.exp(-delta_s):
                magnetization = np.append(magnetization, np.sum(new_config))
                Energy = np.append(Energy, math.exp(-new_config_energy/T))
                avg_Energy=np.append(avg_Energy,avg_energy_per_site(new_config,N,J,h))
                initial_config = np.copy(new_config)

    return magnetization,Energy,avg_Energy    # we return all the observables we need


def energy(config,N,J,h):       #Here we calculate the energy of each configuration 
    sum=0                                   
    for i in range(N):
        for j in range(N):
            sum += config[i][j] * (config[(i + 1) % N][j] + config[(i - 1)][j] + config[i][(j + 1) % N] + config[i][(j - 1)])
    return (-J/2)*sum - (h*np.sum(config))

def average_magnetization(Energy,magnetization,N):
    avg_m=0                                         # Here we sum over all energies and magnetizations to take an average
    for i in range(Energy.size):
        avg_m += magnetization[i]*Energy[i]
    return (avg_m / np.sum(Energy))/(N**2)


def E_exact_J(J): # exact calculation for average energy per site
    c=mpmath.coth(2*J)
    t2=mpmath.tanh(2*J)**2
    s2=mpmath.sech(2*J)**2
    
    Kappa=ellipk(float(4*s2*t2))
    epsilon=-J*c*(1+((2/np.pi)*(2*t2-1)*Kappa))
    return epsilon

def m_exact_J(J): # exact calculation for absolute magnetization
    Jc=0.440686793509772
    if J>Jc:
        return (1-(1/(np.sinh(2*J)**4)))**(1/8)
    elif J<=Jc:
        return 0

def avg_energy_per_site(s,N,J,h):
    H=h                        # Like that we can set h to a constant
    sum_neigh=0
    sum_xy=0
    #we calculate the energy for the whole system
    for i in range(N-1):
        for j in range(N-1):
            sum_neigh+=s[j][i]*s[j+1][i]+s[j][i]*s[j][i+1]
            sum_xy+=s[j][i]
    
    energy_sys=-J*sum_neigh-H*sum_xy
    return energy_sys/(N*N) #and then we take the average of the energy so we obtain the average energy per site

def main():
    grid_size = 15  # we define the N of the grid and then the grid is N*N

    #We fix our variables
    J=0.25
    T=1
    iterations=100
    h_range = np.arange(-1, 1.1, 0.1)
    fixxed_N_m_list=[]

    #we Initialize a configuration with a uniform distribution of -1 and 1
    initial_config = np.empty([grid_size, grid_size])
    for i in range(grid_size):
        for j in range(grid_size):
            initial_config[i][j] = np.random.choice([-1, 1])

    #we srart with the calculation with the fixed J and the variable h
    for h in h_range:
        m_list = []
        for k in range(iterations):
            #we impliment the sweep for a single spin
            magnetization, Energy ,unwanted_var= single_sweep(initial_config,grid_size,J,T,h) 
            #we take the average magnetisation of each configuration 
            avg_m=average_magnetization(Energy,magnetization,grid_size)
            m_list.append(avg_m)
        #here is the average of the average magnetization for each iteration
        fixxed_N_m_list.append(sum(m_list)/iterations)

    # we make the plots
    plt.plot(h_range,fixxed_N_m_list,'b')
    plt.xlabel("h")
    plt.ylabel("<m>")
    plt.title("Plot for fixed grid size (J=0.25,grid size= {}*{},T=1, iterations = {} )".format(grid_size,grid_size,iterations))
    plt.show()

    #Here the variables are redecided in order to make the new calculations
    J_ranrge=np.linspace(0.25,1,40)
    h=0

    #we create some lists to save our measurements
    fixxed_N_E_list=[]
    exact_list=[]

    abs_magn_list=[]
    exact_list_abs=[]

    #and we reinitialize an other configuration
    initial_config = np.empty([grid_size, grid_size])
    for i in range(grid_size):
        for j in range(grid_size):
            initial_config[i][j] = np.random.choice([-1, 1])
    
    # here we start the process with a fixed h=0 and a variable J
    for Jp in J_ranrge:
        #we create lists for the values of each iteration
        E_list=[]
        E_list_exact=[]
        abs_mag=[]
        abs_mag_exact=[]
        for k in range(iterations):
            magnetization, Energy,avg_Energy=single_sweep(initial_config,grid_size,Jp,T,h)

            #we take the average energy per site for every configuration after each sweep
            E_list.append(sum(avg_Energy))
            E_list_exact.append(E_exact_J(Jp))

            #we take the average magnetization for every configuration after each sweep
            abs_mag.append(abs(average_magnetization(Energy,magnetization,grid_size)))
            abs_mag_exact.append(m_exact_J(Jp))

        #Here we take the average after all iterations for every value of J
        fixxed_N_E_list.append(sum(E_list)/iterations)
        exact_list.append(sum(E_list_exact)/iterations)

        abs_magn_list.append(sum(abs_mag)/iterations)
        exact_list_abs.append(sum(abs_mag_exact)/iterations)

    #we make the plots
    plt.plot(J_ranrge,fixxed_N_E_list,"ro")
    plt.plot(J_ranrge,exact_list,"k--")
    plt.xlabel("J")
    plt.ylabel(r"<$\epsilon$>")
    plt.title("Plot for fixed grid size (h=0,grid size= {}*{},T=1, iterations = {} )".format(grid_size,grid_size,iterations))
    plt.show()

    inv_J_range=[1/i for i in J_ranrge] #we create an array with the inverse of J

    plt.plot(inv_J_range,abs_magn_list,"ro")
    plt.plot(inv_J_range,exact_list_abs,"k--")
    plt.xlabel("J")
    plt.ylabel(r"<|m|>")
    plt.title("Plot for fixed grid size (h=0,grid size= {}*{},T=1, iterations = {} )".format(grid_size,grid_size,iterations))
    plt.show()


main()  

