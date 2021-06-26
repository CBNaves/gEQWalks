import numpy as np
from itertools import product
from scipy import sparse

def DisplacementsGenerator(prob_dist_parameters,prob_dist_function,size):
    """ Returns: 
    max_time_step = the time step in which the walker doesn't exceeds the 
    lattice size;
    displacements_vector = displacements sizes that will be used in the walk 
    step, accordingly with the probability distribution used.

    prob_dist_parameters: parameters used by the probability distribution 
    function;

    prob_dist_function: probability of the step sizes function;
    size = size of the lattice.

    """

    displacements_vector = []
    # Parameter to check if the sum of displacements pass the lattice size.
    sum_dx = 0 
    # The maximum time used is equal to half of the linear size.
    tmax = size//2 

    for i in range(1,tmax+1):
        time_vector = np.arange(1,i+1)
        probability_distribution = prob_dist_function(*prob_dist_parameters,
                                                     time_vector)
        dx = np.random.choice(time_vector, p = probability_distribution)
        displacements_vector.append(dx)
        sum_dx = sum_dx + dx

        if sum_dx == size//2:
            max_time_step = i
            break
        elif sum_dx > size//2:
            max_time_step = i-1
            break

    return max_time_step, displacements_vector

def qExponential(q,x):
    """ Returns a discrete Q-exponential probability distribution, defined by 
    the q parameter.
    """
    if q == 1:
        probability_distribution = (2-q)*np.exp(-x)

    elif q < 1 and q >=0:
        probability_distribution = []
        for i in x:
            if (1-q)*i <= 1:
                probability_distribution.append((2-q) * (1-(1-q)*i)**(1/(1-q)))
            else:
                probability_distribution.append(0)

    elif q > 1 and q <= 10**(3):
        probability_distribution = (2-q)*(1-(1-q)*x)**(1/(1-q))

    elif q > 10**(3):
        probability_distribution = np.ones((np.size(x)))
    
    normalization = sum(probability_distribution)
    
    return probability_distribution/normalization

def gaussian_dist(pos,sigma_sq=0.0):
    """ Returns a gaussian distribution array.

    pos: position array for the gaussian distribution calculation;

    sigma_sq:  float that specifies the square of the standard deviation of the 
    distribution. The non-specified value is set equal to zero, i.e. localized
    at the origin.  
    """
    if sigma_sq == 0.0 : 
        return 1
    else: 
        return np.exp(-pos**(2)/(4*sigma_sq))/(2*np.pi*sigma_sq)**(1/4)
    
def pos_index_function(pos,size,dimension):
    """ Returns the index of the matrix element of the walker
    state corresponding to the position specified by the parameter pos.

    pos = (x,y,z,..): position tuple/array of which the index will be 
    calculated;

    size: integer that specifies the lattice linear size;

    dimension: dimension of the lattice.

    """
    pos_index = 0
    h_size = int(size//2)

    for i in range(0,dimension):
        pos_index = pos_index + (size)**(dimension-(i+1)) * (pos[i]+h_size)

    return int(pos_index)


class Lattice:
    """ Class that defines a square lattice which the walkers jumps in. 

    Instance variables: size, pos_eig_val, useful_eig_val.
    """
    
    def __init__(self,dimension,size):
        """ dimension: geometric dimension of the lattice, must be an integer;

        size: integer that specifies the linear size of the lattice, i.e.
        the lattice size along one dimension;

        pos_eig_val: array of ordinate pairs corresponding to the lattice sites;

        useful_eig_val: array of ordinate pairs that can be accessible by the
        walker at a given time. It's initialized as 1, but it's changed in the
        walker initialization.
        """
        self.dimension = dimension
        self.size = size

        pos_array = np.arange(-size//2+1,size//2+1)
        reshape_array = [-1]
        mesh_tuple = []
        transpose_array = []

        for i in range(0,dimension):
            transpose_array.append(dimension-i-1)
            reshape_array.append(size)
            mesh_tuple.append(pos_array)

        mesh_tuple = tuple(mesh_tuple)
        reshape_array.append(dimension)

        self.pos_eig_val = np.meshgrid(*mesh_tuple)
        self.pos_eig_val = np.array(self.pos_eig_val).T.reshape(reshape_array)
        self.pos_eig_val = self.pos_eig_val[0]
        if dimension > 2:
            self.pos_eig_val = np.transpose(self.pos_eig_val,
                                            axes=transpose_array)
        self.useful_eig_val = 1


class FermionCoin:
    """ The fermion coin class defines the coin operator for a fermion spin, 
    that will act as the coin toss defining the direction which the walker 
    will go, accordingly with the possible fermion spin states (up and  
    down).

    Instance variables: coin_parameters, coin.

    Methods: toss(state), entangling_toss2D(state)
    """
    def __init__(self,coin_parameters,lattice):
        """coin_parameters: the set of parameters that specifies the entries
        of the operator. The number of elements must be equal to the 
        dimension of the problem, accordingly with the separable coin operator 
        definition;

        coin: coin operator matrix;
        """
        dimension = lattice.dimension
        size = lattice.size

        self.coin_parameters = coin_parameters
        self.coin = 1
        #  Takes the tensor product of the coins operators in the total spin 
        #  basis order C1 \otimes C2 ...
        #  This only works for separable coins.
        for parameter in self.coin_parameters: 
            theta = parameter
            coin = np.array([[np.cos(theta),1j*np.sin(theta)],[1j*np.sin(theta),np.cos(theta)]])
            self.coin = np.kron(self.coin,coin)

    def toss(self, state):
        """Returns the coin operator applied in the walker state."""
        return np.dot(self.coin,state) # C \ket{coin_state}.
    
    def entangling_toss2D(self,state):
        """Returns a state with a coin operator that entangles the coins 
        degrees in the 2D problem.
        """
        entangling_op = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        entangling_toss = np.dot(self.coin,entangling_op)
        return np.dot(entangling_toss,state)
    

class FermionSpin:
    """ Class that defines a l = 1/2 fermion spin state.
        
        Instance variables: up, down.
    """
    
    def __init__(self):
        """ up: spin up state;
            down: spind down state;
        """
        self.up = np.array([[1],[0]])
        self.down = np.array([[0],[1]])  


class BosonSpin:
    """ Class that defines a l = 1 boson spin state. The main difference 
    between the fermion spin is that the boson spin introduces the possibility
    to remain in the same position, associated with the zero spin state.

    Instance variables: up, zero, down.
    """
    
    def __init__(self):
        self.up = np.array([[1],[0],[0]]) 
        self.zero = np.array([[0],[1],[0]]) 
        self.down = np.array([[0],[0],[1]]) 
           
class Walker:
    """ The walker class defines the walker that will go under the walk and the
    walk itself, i.e. the unitary evolution that he goes under.

    Methods: walk.

    Instance variables: q, memory_dependence, state, tmax, spin_bin, max_pos.
    """
    
    def __init__(self,in_pos_var, coin_instate, lattice, memory_dependence, q):
        """ Parameters:
            in_pos_var (float): initial position variance;

            coin_instate (array): initial coin state;

            lattice: lattice object in which the walker walks in;

            memory_dependence [p_11,p_12,...,p1(d-1),p21,...]: for a random 
        step size multidimensional evolution it gives the probability of 
        selecting the displacement of the other directions, e.g. p_12 would be 
        the probability of selecting the displacement of the direction 2 for 
        1.
            q (float): parameter that specifies the q-Exponential 
        distribution.    
        """
        dimension = lattice.dimension
        size = lattice.size
        pos_basis = lattice.pos_eig_val

        self.q = q
        self.memory_dependence = memory_dependence
        # Makes a column matrix for the walker state, in the fashion
        # [|coin_state(r)>,|coin_state(r')>,..] so that we have an list
        # of coin states for every position of the lattice.
        self.state = np.zeros((size**(dimension),2**dimension,1),dtype='csingle')

        # Localized initial position
        if in_pos_var.any() == 0.0:
            self.max_pos = np.zeros((dimension),dtype = int)
            lattice.useful_eig_val = np.zeros((1,dimension),dtype = int)
            origin = np.zeros((1,dimension))
            origin_index = pos_index_function(origin[0],size,dimension)
            self.state[origin_index] = coin_instate

        # Gaussian position initial state
        else:
            self.max_pos = size//2*np.ones((1,dimension),dtype=int)

            lattice.useful_eig_val = np.copy(lattice.pos_eig_val)
            normalization = 0
            
            if dimension == 1:
                linearized_pos_basis = pos_basis
            elif dimension == 2:
                linearized_pos_basis = pos_basis.reshape(
                                       size**dimension,dimension)
            for pos in linearized_pos_basis:
                pos_amp = 1

                for j in range(0,dimension):
                    pos_amp = pos_amp*gaussian_dist(pos[j],in_pos_var[j])

                normalization = normalization + pos_amp*np.conj(pos_amp)
                pos_index = pos_index_function(pos,size,dimension)
                self.state[pos_index] = pos_amp*spin_instate  
  
            self.state = (1/np.sqrt(normalization))*self.state

        self.tmax = size//2

        self.spin_bins = [] # List that saves the spin binaries.
        for j in range(0,2**dimension):
            spin_str = bin(j)
            spin_str = spin_str[2:]
            while len(spin_str) < dimension:
                    spin_str = '0' + spin_str
            self.spin_bins.append(spin_str)

        displacements_vector = []
        for i in range(0,dimension):
            if q[i] == 0.5:
                displacements = np.ones(size//2)
                max_index = size//2
            else:
                prob_dist_parameters = [q[i]]
                prob_dist_function = qExponential
                max_index, displacements = DisplacementsGenerator(prob_dist_parameters, prob_dist_function,size)

            displacements_vector.append(displacements)

            self.tmax = min(self.tmax,max_index)
            
            # Conditional to make the displacements vector in all the direc-
            # tions the same size, appending zeros.
            if i != 0:
                past_len = np.size(displacements_vector[i-1])
                act_len = np.size(displacements_vector[i])

                if past_len < act_len:
                    for j in range(0,act_len-past_len):
                        displacements_vector[i-1].append(0)
                elif past_len > act_len:
                    for j in range(0,past_len-act_len):
                        displacements_vector[i].append(0)
        
        self.displacements_vector = np.array(displacements_vector)

    def walk(self, coin, lattice, entang, t):
        """ Method that makes the walker walk in one time step. The first 
            parameter must be the coin(s) that will be used, the second the 
            shift operator and the last the lattice.
        """
        dimension = lattice.dimension
        size = lattice.size
        h_size = int(size//2)
        pos_basis = lattice.pos_eig_val
        state = np.copy(self.state)

        # interdependence of the displacements.
        displacements = []
        dim_index_array = np.arange(0,int(dimension),dtype=int)
        for i in range(0,dimension):
            j = np.random.choice(dim_index_array,p = self.memory_dependence[i])
            displacements.append(self.displacements_vector[j,t])

        for i in range(0,dimension):
            if self.max_pos[i] != size//2:
                self.max_pos[i] = self.max_pos[i] +  displacements[i]

        # Updating the accessible positions of the walker
        # for the 1D and 2D problem.
        if dimension == 1: 
            x_i = int(-self.max_pos[0]+size//2)
            x_f = int(self.max_pos[0]+size//2+1)
            lattice.useful_eig_val = np.copy(pos_basis[x_i:x_f])

        elif dimension == 2:
            x_i = int(-self.max_pos[0]+size//2)
            x_f = int(self.max_pos[0]+size//2+1)
            y_i = int(-self.max_pos[1]+size//2)
            y_f = int(self.max_pos[1]+size//2+1) 
            lattice.useful_eig_val = np.copy(pos_basis[x_i:x_f,y_i:y_f])
            lattice.useful_eig_val = lattice.useful_eig_val.reshape((x_f-x_i)*(y_f-y_i),2)         

        for pos in lattice.useful_eig_val:
            pos_index = pos_index_function(pos,size,dimension)
            for i in range(0,2**dimension):
                spin_bin = self.spin_bins[i]
                displaced_pos = np.copy(pos)

                for k in range(0,dimension):
                    int_spin_str = int(spin_bin[k])
                    displacement = (-1)**(int_spin_str+1)*displacements[k]

                    if (pos[k] + displacement <= h_size and 
                        pos[k] + displacement >= -h_size):
                        displaced_pos[k] = displaced_pos[k] + displacement 
                    else:
                        displaced_pos[k] = -pos[k]

                displaced_index = pos_index_function(displaced_pos,size,dimension)
                if entang == False:
                    self.state[pos_index][i] = (
                        coin.toss(state[displaced_index])[i])
                else:
                    self.state[pos_index][i] = ( 
                        coin.entangling_toss2D(state[displaced_index])[i])


