import numpy as np
from itertools import product
from scipy import sparse
import gc 

def DisplacementsGenerator(q,size):

    displacements_vector = []
    sum_dx = 0 
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

    if q == 1:
        probability_distribuition = (2-q)*np.exp(-x)
    elif q != 1 and q <= 10**(3):
        probability_distribuition = (2-q)*(1 - (1-q)*x)**(1/(1-q))
    elif q > 10**(3):
        probability_distribuition = np.ones((np.size(x)))
    
    normalization = sum(probability_distribuition)    
    return probability_distribuition/normalization

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
        self.pos_eig_val, self.pos_eig_vect = matrix_scan([],size,dimension,dimension,[],[])

def matrix_scan(a,size,dimension,dimension_f,eig_pos,eig_vect):    
  
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
                pos = matrix_scan(a,size,dimension-1,dimension_f,eig_pos,eig_vect)
            # When the dimension parameter is equal to one, this means that we 
            # already have the n-tuple of positions and we can define the state
            # vector to that position.
                    
            else:
                pos_state = 1 # Initial scalar for tensor product                     
                for j in range(0,dimension_f):
                    pos_state = sparse.kron(pos_state, position_ket(a[j],size))

                # A copy has to be made to not modify the n-tuple in pos when 
                # we pop.                  
                b = a.copy()  
                eig_pos.append(b)
                eig_vect.append(pos_state)
       
            a.pop(dimension_f-dimension)
              
    return eig_pos, eig_vect

        
def position_ket(position,size):
    
    ''' The position ket defines a column matrix associated with a position 
        state in a direction. The matrix element associated with the origin
        will be the central element, e.g. in a three sites lattice (0,1,0),
        where (1,0,0) is the -1 site and (0,0,1) 1. So we ordenate the states  
        from -N/2 to N/2, N+1 beeing the number of sites.
    '''

    pos_ket = sparse.lil_matrix((size,1),dtype=np.single)
    pos_ket[position + (size//2),0] = 1
   
    return pos_ket

        
class FermionCoin:

    ''' The fermion coin class defines the coin operator, that will act as the 
    coin toss defining the direction which the walker will go, accordingly 
    with the possible fermion spin states (up and down). 
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

        # Change the coin matrix to a sparse matrix.
        self.coin = sparse.lil_matrix(self.coin,dtype=np.csingle)
        pos_identity = sparse.identity(size**dimension,dtype=np.single)
        self.coin = sparse.kron(pos_identity,self.coin,format='lil')
            
    def toss(self, state):
        
        # (I \otimes C)\rho(I \otimes C\dagger)
        return np.dot(self.coin,state)
    
    def entangling_toss2D(self,density,lattice):
    
        ''' entangling_toss2D is a function that implements a coin toss with
            a coin operator that entangles the position states.
        '''
        size = lattice.size
        dimension = lattice.dimension

        entangling_op = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        entangling_op = sparse.lil_matrix(entangling_op,dtype=np.csingle)
        entangling_toss = np.dot(self.coin,entangling_op)
        pos_identity = sparse.identity(size**dimension,dtype=np.single)
        entangling_toss = sparse.kron(pos_identity,entangling_toss,format='lil')
        density = np.dot(entangling_toss,density)
        return np.dot(density,np.conj(entangling_toss.T))
    
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
        # Makes a column matrix for the pos. state, in the fashion |x>|y>.. in 
        # the center of the lattice.
        pos_state = position_ket(0,size)
        for i in range (0,dimension-1):
            pos_state = sparse.kron(pos_state,position_ket(0,size),format='lil')

        spin_state = sparse.lil_matrix(spin_init_state,dtype=np.csingle)
        # |psi> = |pos>|spin>.
        self.state = sparse.kron(pos_state,spin_state,format='lil')
        # \rho = |psi><psi|.
        self.density = np.dot(self.state,np.conj((self.state).T))   
        self.q = q
        self.tmax = size//2
        self.spin_bins = []
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

            if q[i] == 0.5: 
                displacements = np.ones(size)
                max_index = size//2 
            else: 
                max_index, displacements = DisplacementsGenerator(q[i],size)

            displacements_vector.append(displacements)

            self.tmax = min(self.tmax,max_index)
            
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

    def walk(self, coin, lattice, fermion, t):
        
        ''' Method that makes the walker walk in one time step. The first 
        parameter must be the coin(s) that will be used, the second the shift 
        operator and the last the lattice.
        '''
        
        dimension = lattice.dimension
        h_size = int(lattice.size//2)
        pos_basis = np.array(lattice.pos_eig_val)
        state = np.copy(self.state)
        #        if entangling:
#            self.state = coin.entangling_toss2D(self.state,lattice)
#        else:
        state = coin.toss(state)  
        displacements = self.displacements_vector[:,t]
        
        max_region = 0
        for i in range(0,dimension):
            self.max_pos[i] = self.max_pos[i] +  displacements[i]
            max_region = max(max_region,self.max_pos[i])

        max_region = int(max_region)

        for pos in pos_basis[h_size-max_region:h_size+max_region+1,:]:

            col_index = ((pos[0]+h_size)*(2*h_size))**(dimension-1) - 1
#            for spin_bin in self.spin_bins:
            for i in range(0,2**dimension):

                spin_bin = self.spin_bins[i] 
                pos_index = col_index
                displaced_pos_index = col_index

                for k in range(0,dimension):

                    int_spin_str = int(spin_bin[k]) 
                    displacement = (-1)**(int_spin_str+1)*displacements[k]
                    pos_index = pos_index + 2*pos[k]

                    if pos[k] + displacement <= h_size and pos[k] + displacement >= -h_size:
                        displaced_pos_index = displaced_pos_index + 2*(pos[k]+ displacement)
                    else:
                        displaced_pos_index = displaced_pos_index - 2*pos[k]

                self.state[2**(dimension)*h_size + pos_index +i] = state[2**(dimension)*h_size + displaced_pos_index + i]

        self.density = sparse.kron(self.state,np.conj(self.state.T))
 
