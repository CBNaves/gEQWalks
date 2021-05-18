import numpy as np
from itertools import product
from scipy import sparse
import gc 

def DisplacementsGenerator(q,size):

    ''' Function that generates the displacements that will be used in every
        time step, accordingly with the parameters of the probability       
        distribuition used.
    '''

    displacements_vector = []
    # Parameter to check if the sum of displacements pass the lattice size.
    sum_dx = 0 
    # The maximum time used is equal to half of the linear size.
    tmax = size//2 

    for i in range(1,tmax+1):

        time_vector = np.arange(1,i+1)
        probability_distribuition = qExponential(q,time_vector)
        dx = np.random.choice(time_vector, p = probability_distribuition)
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

    'Q-exponential probability distribuition, defined by the q parameter.'

    if q == 1:
        probability_distribuition = (2-q)*np.exp(-x)

    elif q < 1 and q >=0:

        probability_distribuition = []
        for i in x:
            if (1-q)*i <= 1:
                probability_distribuition.append((2-q)*(1 - (1-q)*i)**(1/(1-q)))
            else:
                probability_distribuition.append(0)

    elif q > 1 and q <= 10**(3):
        probability_distribuition = (2-q)*(1 - (1-q)*x)**(1/(1-q))

    elif q > 10**(3):
        probability_distribuition = np.ones((np.size(x)))
    
    normalization = sum(probability_distribuition)    
    return probability_distribuition/normalization

def gaussian_dist(pos,in_pos_var):
    sigma_sq = in_pos_var
    return np.exp(-pos**(2)/(4*sigma_sq))/(2*np.pi*sigma_sq)**(1/4)
    

def pos_index_function(pos,size,dimension):

    ''' Function that returns the index of the matrix element of the walker
        state corresponding to the position specified by the parameter pos.

        The index is determined accordingly with the rule used in the           
        matrix_scan function.
    '''

    pos_index = 0
    h_size = int(size//2)
    for i in range(0,dimension):
        pos_index = pos_index + (size)**(dimension-(i+1))*(pos[i]+h_size)

    return int(pos_index)

class Lattice:
    
    ''' Class that defines a square lattice which the walkers jumps in. 
        The main caracteristics is the dimension and his size, a number with
        the number of sites in each direction (assumed equal and odd for all 
        directions). 
    '''
    
    def __init__(self,dimension,size):
        
        self.dimension = dimension
        self.size = size
        # Defines a list where every element is a list of position basis state 
        # with its eigenvalues, e.g. (x,y,z).
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
            self.pos_eig_val = np.transpose(self.pos_eig_val,axes=transpose_array)

class FermionCoin:

    ''' The fermion coin class defines the coin operator for a fermion spin, 
        that will act as the coin toss defining the direction which the walker 
        will go, accordingly with the possible fermion spin states (up and  
        down). 
    '''
    
    def __init__(self,coin_parameters,lattice):

        ''' The parameter to pass must be a list with the theta angles that 
            defines every coin operator, associated with all directions. 
        '''

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
        return np.dot(self.coin,state) # C \ket{coin_state}.
    
    def entangling_toss2D(self,state):
    
        ''' entangling_toss2D is a function that implements a coin toss with
            a coin operator that entangles the position states.
        '''
        entangling_op = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        entangling_toss = np.dot(self.coin,entangling_op)
        return np.dot(entangling_toss,state)
    
class FermionSpin:
    
    ''' Class that defines a l = 1/2 fermion spin state.
    '''
    
    def __init__(self):
        
        self.up = np.array([[1],[0]])    # \ket{up}
        self.down = np.array([[0],[1]])  # \ket{down}

class BosonSpin:
    
    ''' Class that defines a l = 1 boson spin state. The main difference betwe-
    en the fermion spin is that the boson spin introduces the possibility to 
    remain in the same position, associated with the zero spin state.
    '''
    
    def __init__(self):
        
        self.up = np.array([[1],[0],[0]])   # \ket{up}
        self.zero = np.array([[0],[1],[0]]) # \ket{0}
        self.down = np.array([[0],[0],[1]]) # \ket{down}
           
class Walker:
    
    ''' The walker class defines the walker that will go under the walk and the
    walk itself, i.e. the unitary evolution that he goes under.
    '''
    
    def __init__(self, in_pos_var, spin_instate, lattice, memory_dependence, q):
        
        ''' The position initial state is always on the center of the lattice.
        The first parameter is the spin initial state and he has to be given 
        accordingly with the dimension of the lattice. The second parameter 
        must be the lattice that the walker walks in.
        '''
        dimension = lattice.dimension
        size = lattice.size
        self.q = q
        self.memory_dependence = memory_dependence

        # Makes a column matrix for the walker state, in the fashion
        # [|coin_state(r)>,|coin_state(r')>,..] so that we have an list
        # of coin states in every position of the lattice.
        self.state = 1
        self.max_region = []
        self.max_displacement = []
        reshape_array = []
        for i in range(0,dimension):

            reshape_array.append(size)
            self.max_displacement.append(0)
            linear_state = np.zeros((size,2,1),dtype='csingle')

            # Change to pass a general initial dist. function !!
            # Localized initial position
            if in_pos_var[i] == 0:
                
                linear_state[size//2] = spin_instate[i]
                self.max_region.append(0)           
            # Gaussian position initial state
            else:

                normalization = 0

                for pos in range(-(size//2),(size//2)+1):
                    pos_amp = gaussian_dist(pos,in_pos_var[i])
                    normalization = normalization + pos_amp*np.conj(pos_amp)
                    linear_state[pos+(size//2)] = pos_amp*spin_instate[i]  
  
                linear_state = (1/np.sqrt(normalization))*linear_state
                self.max_region.append(size//2)

            self.state = np.kron(self.state,linear_state)
        reshape_array.append(2**dimension)
        reshape_array.append(1) 
        self.state = self.state.reshape(reshape_array) 

        self.spin_bins = [] # List that saves the spin binaries.
        for j in range(0,2**dimension):
            spin_str = bin(j)
            spin_str = spin_str[2:]
            while len(spin_str) < dimension:
                    spin_str = '0' + spin_str
            self.spin_bins.append(spin_str)

        self.tmax = size//2
        displacements_vector = []
        for i in range(0,dimension):

            max_time_step, displacements = DisplacementsGenerator(q[i],size)
            displacements_vector.append(displacements)
            self.tmax = min(self.tmax,max_time_step)
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
        
        ''' Method that makes the walker walk in one time step. The first 
            parameter must be the coin(s) that will be used, the second the 
            shift operator and the last the lattice.
        '''

        dimension = lattice.dimension
        size = lattice.size
        pos_basis = lattice.pos_eig_val
        h_size = int(size//2)
        state = np.copy(self.state)
        displacements = []

        dim_index_array = np.arange(0,int(dimension),dtype=int)
        for i in range(0,dimension):

            j = np.random.choice(dim_index_array,p = self.memory_dependence[i])
            displacements.append(self.displacements_vector[j,t])

        for i in range(0,dimension):

            self.max_displacement[i] = self.max_displacement[i] +  displacements[i]
            self.max_region[i] = max(self.max_region[i],self.max_displacement[i])


        for pos in pos_basis:

            pos_index = pos_index_function(pos,size,dimension)

            for i in range(0,2**dimension):

                spin_bin = self.spin_bins[i]

                displaced_pos = np.copy(pos)

                for k in range(0,dimension):

                    int_spin_str = int(spin_bin[k])
                    displacement = (-1)**(int_spin_str+1)*displacements[k]

                    if pos[k] + displacement <= h_size and pos[k] + displacement >= -h_size:
                        displaced_pos[k] = displaced_pos[k] + displacement 
                    else:
                        displaced_pos[k] = -pos[k]

                displaced_index = pos_index_function(displaced_pos,size,dimension)
                if entang == False:
                    self.state[pos_index][i] = coin.toss(state[displaced_index])[i]
                else:
                    self.state[pos_index][i] = coin.entangling_toss2D(state[displaced_index])[i]
