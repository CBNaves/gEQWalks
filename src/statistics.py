import numpy as np
import gEQWalks
from scipy import sparse

def position_statistics(state,lattice): 
    """ Function that returns an array of position probability distribuitions 
    for all directions, the mean positions, the mean square of the positions 
    and the variance respectively. 

    state = walker's state
    lattice = lattice in which the walker walks in.
    """
    size = lattice.size
    dimension = lattice.dimension
    h_size = int(size//2)
    positions = lattice.useful_eig_val

    mean_pos = np.zeros((1,dimension))
    mean_sq_pos = np.zeros((1,dimension))
    pos_prob_dist = np.zeros(size**dimension) 
    sigma_squared = []

    for pos in positions:
        pos_index = gEQWalks.pos_index_function(pos,size,dimension)
        pos_state = state[pos_index] 
        pos_prob_dist[pos_index] = np.real(np.dot(np.conj(pos_state.T),
                                                  pos_state)[0])
        for i in range(0,dimension): 
            mean_pos[0][i] = mean_pos[0][i] + pos[i]*pos_prob_dist[pos_index]
            mean_sq_pos[0][i] = mean_sq_pos[0][i] + (pos[i]**2)*pos_prob_dist[pos_index]

    for i in range(0,dimension):
        sigma_squared.append([mean_sq_pos[0][i] - (mean_pos[0][i])**2])

    pos_prob_dist = pos_prob_dist.reshape(int(size)*np.ones(dimension,dtype = int))

    return pos_prob_dist, mean_pos, mean_sq_pos, sigma_squared
    

def entanglement_entropy(state,lattice):
    """ Returns the entanglement entropy of the coin degree.

    state = walker's state, method of the Walker class;
    lattice = lattice in which the walker walks in. 
    """
    dimension = lattice.dimension
    size = lattice.size
    positions = lattice.useful_eig_val

    coin_density = np.zeros((2**dimension,2**dimension),dtype='csingle')

    for pos in positions:
        pos_index = gEQWalks.pos_index_function(pos,size,dimension)
        coin_state = state[pos_index]
        coin_density = coin_density + np.dot(coin_state,np.conj(coin_state.T)) 

    # Reshaping the coin density in a suitable manner for partial tracing.
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
        lp = np.real(eigen_val[0])
        lm = np.real(eigen_val[1])

        if lp == 0: lgp = 1
        else: lgp = np.log(lp)
        if lm == 0: lgm = 1
        else: lgm = np.log(lm)

        entropy = (-lm*lgm- lp*lgp)/np.log(2)
        entanglement_entropy_list.append(entropy)

    return entanglement_entropy_list

def trace_distance(rho,sigma,lattice):
    """ Returns the trace distance between two coin states.

    rho, sigma = coin states;
    lattice = lattice in which the coins walks in.
    """   
    dimension = lattice.dimension
    size = lattice.size

    coin_rho = np.zeros((2**dimension,2**dimension))
    coin_sigma = np.zeros((2**dimension,2**dimension))

    for coin_rho_state in rho:
        coin_rho = coin_rho + np.dot(coin_rho_state,np.conj(coin_rho_state.T))
    
    for coin_sigma_state in sigma:
        coin_sigma = (coin_sigma + 
                     np.dot(coin_sigma_state,np.conj(coin_sigma_state.T)))

    dif_density = coin_rho - coin_sigma
    dif_eigen_val, dif_eigen_vec = np.linalg.eig(dif_density)
    dif_eigen_val = np.real(dif_eigen_val)

    trace_dist = 0 
    for i in dif_eigen_val:
        trace_dist = trace_dist + (1/2)*np.linalg.norm(i)

    return trace_dist  
  
