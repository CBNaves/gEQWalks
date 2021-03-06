import numpy as np
import gEQWalks
from scipy import sparse

def position_statistics(state,lattice,coin): 
    
    '''Function that returns an array of probabilities associated with a given 
    time step in all directions, the mean positions, the mean square
    of the positions and the variance respectively. The first parameter must be
    the density function, the second the lattice in which he walks. The last 
    parameter is the dimension of the coin, i.e. 2 if a fermion, 3 if a boson 
    coin. 
    '''
    size = lattice.size
    dimension = lattice.dimension
    h_size = int(size//2)

    positions = np.array(lattice.pos_eig_val)
    # Array that stores the mean position in the lattice in every direction. 
    mean_pos = np.zeros((1,dimension))
    # Array that stores the mean squared position in the lattice ''. 
    mean_sq_pos = np.zeros((1,dimension))

    sigma_squared = []

    pos_prob_dist = np.zeros((dimension,size))

    for pos in positions:

        pos_index = gEQWalks.pos_index_function(pos,size,dimension)
        pos_state = state[pos_index] 
        pos_prob = np.real(np.dot(np.conj(pos_state.T),pos_state)[0])

        for i in range(0,dimension):
            pos_prob_dist[i][pos[i]+h_size] = pos_prob_dist[i][pos[i]+h_size] + pos_prob 
            mean_pos[0][i] = mean_pos[0][i] + pos[i]*pos_prob
            mean_sq_pos[0][i] = mean_sq_pos[0][i] + (pos[i]**2)*pos_prob

    for i in range(0,dimension):
        sigma_squared.append([mean_sq_pos[0][i] - (mean_pos[0][i])**2])

    return pos_prob_dist,mean_pos,mean_sq_pos,sigma_squared
    

def entanglement_entropy(state,lattice):
    
    ''' Function that returns the entanglement entropy of the coins degrees.
        The first parameter must be the density matrix and the second the
        lattice.
    '''
    
    dimension = lattice.dimension
    size = lattice.size
   
    coin_density = np.zeros((2**dimension,2**dimension),dtype='csingle')

    for coin_state in state:

        coin_density = coin_density + np.dot(coin_state,np.conj(coin_state.T)) 

    coin_density = (1/(2**(dimension-1)))*np.array(coin_density)
    # Reshaping the coin density in a suitable manner.
    coin_reshape_list = [2 for x in range(0,2**dimension)]
    coin_density = coin_density.reshape(coin_reshape_list)

    remaining_density = coin_density 
    entanglement_entropy_list = []

    for i in range(0,dimension):
        try:
            coin_reshape_list = (2,2**(dimension-i-1),2,2**(dimension-i-1))
            coin_density = remaining_density.reshape(coin_reshape_list)
            remaining_density = np.trace(coin_density,axis1=0,axis2=2)
            coin_density = np.trace(coin_density,axis1=1,axis2=3)
        
        # If the reshape returns an error that means that we dont need to
        # do it anymore.     
        except:
            coin_density = remaining_density

        eigen_val,eigen_vec = np.linalg.eig(coin_density)
        lp = np.real(eigen_val[0])  # First eigen value
        lm = np.real(eigen_val[1])  # Second eigen value

        if lp == 0: lgp = 1 # To avoid log(0)
        else: lgp = np.log(lp)
        if lm == 0: lgm = 1
        else: lgm = np.log(lm)

        # S_E = -\lamb_+log_2(\lamb+) - \lamb-log_2(\lamb-)
        entropy = (-lm*lgm- lp*lgp)/np.log(2)
        entanglement_entropy_list.append(entropy)

    return entanglement_entropy_list

def trace_distance(rho,sigma,lattice):
    
    dimension = lattice.dimension
    size = lattice.size

    coin_rho = np.zeros((2**dimension,2**dimension))
    coin_sigma = np.zeros((2**dimension,2**dimension))

    for coin_rho_state in rho:
        coin_rho = coin_rho + np.dot(coin_rho_state,np.conj(coin_rho_state.T))
    
    for coin_sigma_state in sigma:
        coin_sigma = coin_sigma + np.dot(coin_sigma_state,np.conj(coin_sigma_state.T))

    dif_density = coin_rho - coin_sigma
    dif_eigen_val, dif_eigen_vec = np.linalg.eig(dif_density)
    dif_eigen_val = np.real(dif_eigen_val)

    trace_dist = 0 
    for i in dif_eigen_val:
        trace_dist = trace_dist + (1/2)*np.linalg.norm(i)

    return trace_dist    
