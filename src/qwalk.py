import numpy as np
from itertools import product
from scipy import sparse

class Lattice:
    
    ''' Class that defines a square lattice which the walkers jumps in. 
        The main caracteristics is the dimension and his size, a number with
        the number of sites in each direction (assumed equal and odd for all 
        directions). 
    '''
    
    def __init__(self,dimension,size):
        
        self.dimension = dimension
        self.size = size
        # Defines a list where every element is a list of position basis state with its 
        # eigenvalues, e.g. (x,y,z).
        self.pos_basis = matrix_scan([],size,dimension,dimension,[])
        
        
def position_ket(position,size):
    
    ''' The position ket defines a column matrix associated with a position sta-
        -te in a direction. The matrix element associated with the origin will 
        be the central element, e.g. in a three sites lattice (0,1,0), where 
        (1,0,0) is the -1 site and (0,0,1) 1. So we ordenate the states from 
        -N/2 to N/2, N+1 beeing the number of sites.
    '''

    pos_ket = sparse.csr_matrix((size,1))
    pos_ket[position + (size//2),0] = 1
   
    return pos_ket

def matrix_scan(a,size,dimension,dimension_f,pos):    
  
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
                pos = matrix_scan(a,size,dimension-1,dimension_f,pos)
                    
            # When the dimension parameter is equal to one, this means that we already have the n-tuple of positions 
            # and we can define the state vector to that position.
                    
            else:
                pos_state = 1 # Initial scalar for tensor product                     
                for j in range(0,dimension_f):
                    pos_state = sparse.kron(pos_state, position_ket(a[j],size))
                                
                b = a.copy()  # A copy has to be made to not modify the n-tuple in pos when we pop.   
                pos.append([b,pos_state])                                     
       
            a.pop(dimension_f-dimension)
              
    return pos
        
class FermionCoin:

    ''' The fermion coin class defines the coin operator, that will act as the 
        coin toss defining the direction which the walker will go, accordingly 
        with the possible fermion spin states (up and down). 
    '''
    
    def __init__(self,coin_parameters):

        ''' The parameter to pass must be a list with the theta angles that de-
        fines every coin operator, associated with all directions. 
        '''
        
        self.coin_parameters = coin_parameters
        self.coin = 1
        #  Takes the tensor product of the coins operators in the total spin ba-
        #  -sis order C1 \otimes C2 ...
        #  This only works for separable coins.
        for parameter in self.coin_parameters: 
            theta = parameter
            self.coin = np.kron( self.coin, np.array([[np.cos(theta),1j*np.sin(theta)],
                             [1j*np.sin(theta),np.cos(theta)]]))

        self.coin = sparse.csr_matrix(self.coin)
            
        
    def toss(self,walker,lattice):
        
        # First we have to take the tensor product of the identity on the 
        # position space with the coin op.
        dimension = lattice.dimension
        size = lattice.size
        pos_identity = sparse.identity(size**dimension)
        coin_toss = sparse.kron(pos_identity,self.coin,format='csr')
        return np.dot(coin_toss,walker)  # (I \otime Cs)|state>
    
    def entangling_toss2D(self,walker,lattice):
        
        entangling_op = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        entanglin_op = sparse.csr_matrix(entanglin_op)
        entangling_toss = np.dot(self.coin,entangling_op)
        pos_identity = sparse.identity(size**dimension)
        entangling_toss = sparse.kron(pos_identity,entangling_toss,format='csr')
        return np.dot(entangling_toss,walker)
    
class FermionSpin:
    
    ''' Class that defines a l = 1/2 fermion spin state.
    '''
    
    def __init__(self):
        
        self.up = np.array([[1],[0]])    # \ket{up}
        self.down = np.array([[0],[1]])  # \ket{down}

class BosonSpin:
    
    ''' Class that defines a l = 1 boson spin state. The main difference between the fermion spin is that 
    the boson spin introduces the possibility to remain in the same position, associated with the zero spin 
    state.'''
    
    def __init__(self):
        
        self.up = np.array([[1],[0],[0]])   # \ket{up}
        self.zero = np.array([[0],[1],[0]]) # \ket{0}
        self.down = np.array([[0],[0],[1]]) # \ket{down}

class FermionShiftOperator:
    
    ''' Class that defines the shift operator to a walker with half spin coins.
    To make the shift operator a  lattice must be defined, i.e. his dimension 
    and number of sites, and a fermion must be given. We associate with the 
    spin up a displacement l+1, while spin down with a displacement l-1.
    '''
    
    def __init__(self,lattice,fermion):

        ''' The first parameter must be a lattice object, while the second must
        be a fermion.
        '''                        
      
        f = [fermion.up,fermion.down]
        dimension = lattice.dimension
        size = lattice.size
        pos_basis = lattice.pos_basis
       
        ''' Here we define the shift operator accordingly to the dimension of the
        problem. The dimension of the shift operator is the dimension of an o-
        -perator on the space Hp \otimes Hc, where Hp is the position Hilbert
        Space and Hc the coin Hilbert space, considering all coins.
        '''
        # Dimension of the shift operator.
        shift_dimension = size**dimension*(2**dimension)
        # Sparse matrix of the shift operator. 
        self.shift = sparse.csr_matrix((shift_dimension,shift_dimension),dtype=float)
        
        
        ''' In the below loop we take all possible configurations of spin sta-
        -tes, 2^(dimension). We associate to a binary number (string) every con-
        -figuration, with 0 to a spin up and 1 to a spin down. So the up,up,up 
        state will be associated with 000, while up,down,up is associated with 
        010. 
        '''        
            
        for i in range(0,2**dimension):
            
            b = bin(i)[2:]  # Binary number associated with the configuration.
            # Here we attach zeros to the string to match the number of spins.
            if len(b) < dimension:  
                for k in range(0,(dimension-len(b))):
                    b = '0' + b
            # Below we define the spin state associated with a configuration, 
            # e.g. \ket{up,up,up}.        
            spin = f[int(b[0])] 
            for j in b[1:]:
                spin = np.kron(spin,f[int(j)])
        
            # Zeros array to sum with the pos. part of the shift operator
            pos_shift = sparse.csr_matrix((size**(dimension),size**(dimension)),dtype=float)
            
            # Loop that catchs every possible position state and creates the 
            # correspondingly shift accordingly with the spins state in every 
            # direction.

            for pos in pos_basis:
                old_pos = pos[1]
                new_pos = np.array([1])
                for x in range(0,dimension):   
                    old_p = pos[0][x]
                    new_p = old_p + (-1)**(int(b[x]))   # Spin up : l --> l+1, spin down: l --> l-1.
                            
                    if (new_p) <= (size//2) and new_p >= -(size//2):
                                 
                        new_pos = sparse.kron(new_pos,position_ket(new_p,size),format='csr')
                                
                    else:
                        new_pos = np.kron(new_pos,position_ket(-1*old_p,size))                 
                
                pos_shift = pos_shift + np.dot(new_pos,(old_pos.T)) # Sum of |l><l +/-1|.
            
            spin_op = np.dot(spin,spin.T)   # \ket{spin} --> \ket{\spin}\bra{\spin} (operator).
            
            self.shift = self.shift + np.kron(pos_shift,spin_op)    # Sum of (sum l)|l><l +- 1|\otimes|spin><spin|.
        
    
class Walker:
    
    ''' The walker class defines the walker that will go under the walk and the
    walk itself, ie. the unitary evolution that he goes under.
    '''
    
    def __init__(self,spin_init_state,lattice):
        
        ''' The position initial state is always on the center of the lattice.
        The first parameter is the spin initial state and he has to be given 
        accordingly with the dimension of the lattice. The second parameter must
        be the lattice that the walker walks in.
        '''
        
        # Makes a column matrix for the pos. state, in the fashion |x>|y>.. in 
        # the center of the lattice.
        self.pos_state = position_ket(0,lattice.size)
        for i in range (0,lattice.dimension-1):
            self.pos_state = sparse.kron(self.pos_state,position_ket(0,lattice.size),format='csr)

        self.spin_state = sparse.csr_matrix(spin_init_state)
        # |psi> = |pos>|spin>
        self.state = sparse.kron(self.pos_state,self.spin_state,format='csr') 
        self.density = np.dot(self.state,np.conj((self.state).T))   # \rho = |psi><psi|
            
            
    def walk(self,coin,shift_operator,lattice):
        
        ''' Method that makes the walker walk in one time step. The first 
        parameter must be the coin(s) that will be used, the second the shift 
        operator and the last the lattice.
        '''
        
        self.state = coin.toss(self.state,lattice)  # Coin toss.
        #self.state = coin.entangling_toss2D(self.state,lattice) # Entanglig coin toss
        self.state = np.dot(shift_operator.shift,self.state)    # Shift.
        # Update of the density matrix.
        self.density = np.dot(self.state,np.conj((self.state).T)) 
        
        
class ElephantFermionShiftOperator:
    
    ''' Class that defines the fermion shift operator to the elephant quantum walk.
    '''
    
    def __init__(self,elephant_shift,elephant_memory_combinations,lattice,fermion,time):
        
        ''' The first parameter must be the previous elephant shift, the second
        the combinations of the delta parameters in every direction, the third 
        the lattice, the forth the fermion and the last the time in which the 
        shift operator will be defined. 
        '''
        
        f = [fermion.up,fermion.down]
        dimension = lattice.dimension
        size = lattice.size
        pos_basis = lattice.pos_basis
       
        # Here we define the shift operator accordingly to the dimension of the problem. The dimension of
        # the shift operator is the dimension of an operator on the space Hp \otimes Hc, where Hp is the 
        # position Hilbert Space and Hc the coin Hilbert space, considering all coins.
        
        self.shift = np.zeros(((size**dimension)*(2**dimension),(size**dimension)*(2**dimension)))      
        
        # Updates the coeficients of the past elephant_shift operator, 
        # assuming the uniform distribuition of the memory.
        if time > 1: 
            elephant_shift = (((time-1)/time)**(dimension))*elephant_shift
        
        # In the below loop we take all possible configurations of fermion spin states, 2^(dimension). 
        # We associate to a binary number (string) every configuration, with 0 to a spin up and 1 to a spin 
        # down. So the up,up,up state will be associated with 000, while up,down,up is associated with 010.  
        
        for i in range(0,2**dimension):
    
            pos_shift = np.zeros((size**(dimension),size**(dimension)))
            
            # Loop that takes every delta_x,delta_y,delta_z,... combinations.
            for j in elephant_memory_combinations: 
                
                b = bin(i)[2:] # Binary number associated with the configuration.
                if len(b) < dimension: # Here we attach zeros to the string to match the number of spins.
                    for k in range(0,(dimension-len(b))):
                        b = '0' + b
                        
                # Below we define the spin state associated with a configuration, e.g. \ket{up,up,up}.
                spin = f[int(b[0])] 
                for k in b[1:]:
                    spin = np.kron(spin,f[int(k)])

                for pos in pos_basis:
                    old_pos = pos[1]    # Old position ket.
                    new_pos = np.array([1]) # Scalar for the initial tensor product.
                    for x in range(0,dimension):   
                        old_p = pos[0][x]   # Old position.
                        new_p = old_p + (-1)**(int(b[x]))*j[x]  # l_j -> l_j + (-1)^(spin)*delta^j_t 
                        
                        # Conditional on the borders, imposing ciclic conditions, maintaining the unitarity.
                        
                        if (new_p) <= (size//2) and new_p >= -(size//2): 
                                 
                            new_pos = np.kron(new_pos,position_ket(new_p,size))
                                
                        else:
                            new_pos = np.kron(new_pos,position_ket(-1*old_p,size))
                        
                    # Summing the position shift operator part.          
                    pos_shift = pos_shift + np.dot(new_pos,(old_pos.T))
                    
                spin_op = np.dot(spin,spin.T)   # \ket{spin} --> \ket{\spin}\bra{\spin} (operator)
            
                if time!= 0 : self.shift = self.shift + (1/time**(dimension))*np.kron(pos_shift,spin_op)
                else:         self.shift = self.shift + np.kron(pos_shift,spin_op)
                    
        self.shift = elephant_shift + self.shift # Sum the old shift operator, with the old memory combs.
                    
                
        
class ElephantWalker:
    
    ''' Class that defines the elephant walker.
    '''
    
    def __init__(self,spin_init_state,lattice):
        
        ''' The first parameter must be the spin initial state, accordingly with the dimension of the problem.
        The second the lattice in which the elephant walks.
        '''
        
        # Makes the initial position state in the center of the lattice.
        self.pos_state = sparse.csr_matrix(position_ket(0,lattice.size))
        for i in range(0,lattice.dimension-1):
            self.pos_state = sparse.kron(self.pos_state,position_ket(0,lattice.size),format='csr)
        
        self.spin_state = spin_init_state   # Initial spin state of the walker.
        self.state = np.kron(self.pos_state,self.spin_state)    # |state> = |pos> \otimes |spin>.
        self.density = np.dot(self.state,np.conj(self.state.T)) # \rho = |state><state|.
        
        # We save the elephant shift operator with the elephant in order to not redefine him entirely 
        # every time step.
        self.elephant_shift = np.zeros((lattice.size**(lattice.dimension)*2**(lattice.dimension),lattice.size**(lattice.dimension)*2**(lattice.dimension)))
        self.memory = []    # Defines a list that saves the displacements of every time step.
    
    def walk(self,coin,lattice,fermion,time):
        
        ''' Method that makes the elephant walker walk in one time step. The first parameter must be the coin(s)
        that will be used, the second the lattice and the third and last a fermion and the time to define 
        the shift operator. '''
        
        dimension = lattice.dimension
        
        self.state = coin.toss(self.state,lattice) # coin toss (I \otimes C)|state>.
        
        memory_combinations = [] # list of the new memory comb. to define the new shift operator.
        
        
        if time == 0: # In the first time step the elephant has to no memory.
            deltas = []
            for i in range(0,dimension): # Loop that selects the displacements, GENERALIZE FOR q and 1-q!!!
                
                # If the random number in [0,1] is <= 0.5 delta = +1.
                if np.random.random()<=0.5: deltas.append(+1)
                # Else delta = -1    
                else: deltas.append(-1)
                    
            self.memory.append(deltas) # Save the displacements in the elephant memory.
            memory_combinations.append(self.memory[0]) # In this case we dont have to make any combinations.
            e_s = elephant_fermion_shift_operator(self.elephant_shift,memory_combinations,lattice,fermion,0)
            self.elephant_shift = e_s.shift # Saves the elephant shift operator in the elephant.
            
        else:
            # Here we have to consider if the dimension is greater than one because if it isnt we dont have
            # to make any combination of the deltas, as we have just one dimension.
            if dimension > 1 :
                combinations = [] # list of the n-tuples of coordinates, i.e. (x1,x2,...,xn),(y1,y2,...,yn)...
                
                # Loop that takes every n-tuple of deltas in the memory and makes combinations n to n with the 
                # new n-tuple, where the n is the dimension of the problem and the combinations do not repeat deltas 
                # in the same direction.
    
                for i in self.memory[:-1]:
                    coordinates = np.array([i,self.memory[-1]]) 
                    
                    for j in range(0,dimension):
                        combinations.append(coordinates[:,j])
                    
                # This list comprehension makes every combination of n elements of the n-tuples in combinations list
                [memory_combinations.append(p) for p in product(*combinations)]
                    
                memory_combinations.pop(0) # Remove the first element, that always will be repeated
                
            else:
                memory_combinations = [self.memory[-1]]
                
            
            # Below we define the elephant fermion shift operator accordingly with the time step.
            
            if time!=1: # In the t=1 time instant the elephant will be shifted with the same operator in t=0.
                e_s = elephant_fermion_shift_operator(self.elephant_shift,memory_combinations,lattice,fermion,time)
                self.elephant_shift = e_s.shift
            
            deltas = [] # List that saves the actual random displacements +/1 for every direction.
        
            for i in range(0,dimension): # Loop that selects the displacements, GENERALIZE FOR q and 1-q!!!
                if np.random.random()<=0.5: deltas.append(+1) 
                else: deltas.append(-1) 
        
            self.memory.append(deltas) # Saving the news displacements in the memory.
            
        self.state = np.dot(self.elephant_shift,self.state) # S_E |state>
        self.density = np.dot(self.state,np.conj(self.state.T)) # updates the density operator        
        
