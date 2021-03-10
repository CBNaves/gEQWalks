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
            max_index = i
            break
        elif sum_dx > size//2:
            max_index = i-1
            break

    return max_index, displacements_vector

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
        probability_distribuition = (2-q)*(1 - (1-q)*i)**(1/(1-q))

    elif q > 10**(3):
        probability_distribuition = np.ones((np.size(x)))
    
    normalization = sum(probability_distribuition)    
    return probability_distribuition/normalization

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
        self.pos_eig_val = matrix_scan([],size,dimension,dimension,[])

def matrix_scan(a,size,dimension,dimension_f,eig_pos):    
  
    ''' Matrix_scan is a recursive function that passes through every position 
    in the n-dimensional square lattice, to define the position basis. The re-
    -cursivity is used to generalize the scan to any dimension. We take a posi-
    -tion in the N-1 directions and scans to every possible position on the 
    last, then we chage the position on the N-2 direction and rebegin this pro-
    -cess (e.g. in a square 2D lattice we scan in the vertical lines).
            
    The first argument is a list to save the position in every direction, the 
    second the dimension of the lattice, the third a fixed copy, and the last a
    parameter to save the position ket on the returning of the recursivity.
    '''
            
    # If the dimension is not zero, we take a position in the lattice in a di-
    # -rection and pass to the next.

    if dimension !=0:
        for i in range(-(size//2),(size//2) +1):
            a.append(i)
            if dimension != 1: 
                pos = matrix_scan(a,size,dimension-1,dimension_f,eig_pos)
            # When the dimension parameter is equal to one, this means that we 
            # already have the n-tuple of positions and we can define the state
            # vector to that position.
                    
            else:
                # A copy has to be made to not modify the n-tuple in pos when 
                # we pop.                  
                b = a.copy()  
                eig_pos.append(b)
       
            a.pop(dimension_f-dimension)
              
    return eig_pos

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
    
    def __init__(self, spin_init_state, lattice, q):
        
        ''' The position initial state is always on the center of the lattice.
        The first parameter is the spin initial state and he has to be given 
        accordingly with the dimension of the lattice. The second parameter 
        must be the lattice that the walker walks in.
        '''
        
        dimension = lattice.dimension
        size = lattice.size

        # Makes a column matrix for the walker state, in the fashion
        # [|coin_state(r)>,|coin_state(r')>,..] so that we have an list
        # of coin states in every position of the lattice.
        self.state = np.zeros((size**(dimension),2**dimension,1),dtype='csingle')
        origin = np.zeros((1,dimension))
        origin_index = pos_index_function(origin[0],size,dimension)
        self.state[origin_index] = spin_init_state

        self.q = q
        self.tmax = size//2
        self.spin_bins = [] # List that saves the spin binaries.
        self.max_pos = [] 

        for j in range(0,2**dimension):
            spin_str = bin(j)
            spin_str = spin_str[2:]
            while len(spin_str) < dimension:
                    spin_str = '0' + spin_str
            self.spin_bins.append(spin_str)

        displacements_vector = []

        for i in range(0,dimension):

            self.max_pos.append(0)

            max_index, displacements = DisplacementsGenerator(q[i],size)

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
        
        ''' Method that makes the walker walk in one time step. The first 
            parameter must be the coin(s) that will be used, the second the 
            shift operator and the last the lattice.
        '''
        dimension = lattice.dimension
        size = lattice.size
        h_size = int(size//2)
        pos_basis = np.array(lattice.pos_eig_val)
        state = np.copy(self.state)

        displacements = self.displacements_vector[:,t]
        
#        max_region = 0
#        for i in range(0,dimension):
#            self.max_pos[i] = self.max_pos[i] +  displacements[i]
#            max_region = max(max_region,self.max_pos[i])

#        max_region = int(max_region)

#        min_ind = ((2*h_size)**(dimension))/2 - (2*h_size)**(dimension-1)*max_region
#        max_ind = ((2*h_size)**(dimension))/2 + (2*h_size)**(dimension-1)*max_region

#        min_ind = int(min_ind)
#        max_ind = int(max_ind)

        for pos in pos_basis[:,:]:

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
