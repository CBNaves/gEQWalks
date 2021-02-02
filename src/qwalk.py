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
        # Defines a list where every element is a list of position basis state 
        # with its eigenvalues, e.g. (x,y,z).
        self.pos_basis = matrix_scan([],size,dimension,dimension,[])
        
        
def position_ket(position,size):
    
    ''' The position ket defines a column matrix associated with a position 
        state in a direction. The matrix element associated with the origin
        will be the central element, e.g. in a three sites lattice (0,1,0),
        where (1,0,0) is the -1 site and (0,0,1) 1. So we ordenate the states  
        from -N/2 to N/2, N+1 beeing the number of sites.
    '''

    pos_ket = sparse.lil_matrix((size,1),dtype=complex)
    pos_ket[position + (size//2),0] = 1
   
    return pos_ket

def memory_ket(memory_loc,size):

    mem_ket = sparse.lil_matrix((size,1),dtype=complex)
    mem_ket[memory_loc,0] = 1
    
    return mem_ket

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
                pos.append([b,pos_state])                                     
       
            a.pop(dimension_f-dimension)
              
    return pos
        
class FermionCoin:

    ''' The fermion coin class defines the coin operator, that will act as the 
    coin toss defining the direction which the walker will go, accordingly 
    with the possible fermion spin states (up and down). 
    '''
    
    def __init__(self,coin_parameters):

        ''' The parameter to pass must be a list with the theta angles that 
        defines every coin operator, associated with all directions. 
        '''
        
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
        self.coin = sparse.lil_matrix(self.coin,dtype=complex)
            
    def toss(self, density, lattice):
        
        # First we have to take the tensor product of the identity on the 
        # position space with the coin op.
        dimension = lattice.dimension
        size = lattice.size
        pos_identity = sparse.identity((size**dimension),dtype=complex,format='lil')
        mem_identity = sparse.identity((size//2),dtype=complex,format='lil')
        coin_toss = sparse.kron(pos_identity,self.coin,format='lil')
        coin_toss = sparse.kron(coin_toss,mem_identity,format='lil')

        # (I \otimes C)\rho(I \otimes C\dagger)
        return np.dot(np.dot(coin_toss,density),np.conj(coin_toss.T))
    
    def entangling_toss2D(self,density,lattice):
    
        ''' entangling_toss2D is a function that implements a coin toss with
            a coin operator that entangles the position states.
        '''
        size = lattice.size
        dimension = lattice.dimension

        entangling_op = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        entangling_op = sparse.lil_matrix(entangling_op)
        entangling_toss = np.dot(self.coin,entangling_op)
        pos_identity = sparse.identity(size**dimension)
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

class FermionShiftOperator:
    
    # Não faz muito sentido ser uma classe!!

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
       
        '''Here we define the shift operator accordingly to the dimension of the
        problem. The dimension of the shift operator is the dimension of an o-
        -perator on the space Hp \otimes Hc, where Hp is the position Hilbert
        Space and Hc the coin Hilbert space, considering all coins.
        '''
        # Dimension of the shift operator.
        shift_dimension = size**dimension*(2**dimension)
        # Sparse matrix of the shift operator. 
        self.shift = sparse.lil_matrix((shift_dimension,shift_dimension),dtype=complex)
        
        
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
            pos_shift = sparse.lil_matrix((size**(dimension),size**(dimension)),dtype=complex)
            
            # Loop that catchs every possible position state and creates the 
            # correspondingly shift accordingly with the spins state in every 
            # direction.

            for pos in pos_basis:
                old_pos = pos[1]
                new_pos = np.array([1])
                for x in range(0,dimension):   
                    old_p = pos[0][x]
                    # Spin up : l --> l+1, spin down: l --> l-1.
                    new_p = old_p + (-1)**(int(b[x]))  
                    
                    # Cyclic conditions for the unitarity of the operator.      
                    if (new_p) <= (size//2) and new_p >= -(size//2):
                        npk = position_ket(new_p,size)         
                        new_pos = sparse.kron(new_pos, npk, format='lil')
                                
                    else:
                        npk = position_ket(-1*old_p,size)
                        new_pos = sparse.kron(new_pos, npk, format='lil')                
                # Sum of |l><l +/-1|.
                pos_shift = pos_shift + np.dot(new_pos,(old_pos.T))
            
            # \ket{spin} --> \ket{\spin}\bra{\spin} (operator).
            spin_op = np.dot(spin,np.conj(spin.T))
  
            # Sum of (sum l)|l><l +- 1|\otimes|spin><spin|.
            self.shift = self.shift + sparse.kron(pos_shift,spin_op)    
        
    
class Walker:
    
    ''' The walker class defines the walker that will go under the walk and the
    walk itself, i.e. the unitary evolution that he goes under.
    '''
    
    def __init__(self,spin_init_state,lattice):
        
        ''' The position initial state is always on the center of the lattice.
        The first parameter is the spin initial state and he has to be given 
        accordingly with the dimension of the lattice. The second parameter 
        must be the lattice that the walker walks in.
        '''
        
        # Makes a column matrix for the pos. state, in the fashion |x>|y>.. in 
        # the center of the lattice.
        pos_state = position_ket(0,lattice.size)
        for i in range (0,lattice.dimension-1):
            pos_state = sparse.kron(pos_state,position_ket(0,lattice.size),format='lil')

        spin_state = sparse.lil_matrix(spin_init_state)
        # |psi> = |pos>|spin>.
        state = sparse.kron(pos_state,spin_state,format='lil')
        # \rho = |psi><psi|.
        self.density = np.dot(state,np.conj((state).T))   
            
            
    def walk(self,coin,shift_operator,lattice,entangling):
        
        ''' Method that makes the walker walk in one time step. The first 
        parameter must be the coin(s) that will be used, the second the shift 
        operator and the last the lattice.
        '''
        
        if entangling:
            self.density = coin.entangling_toss2D(self.density,lattice)
        else:
            self.density = coin.toss(self.density,lattice)  
        
        # E(\rho) = S\rhoS\dagger
        self.density = np.dot(shift_operator.shift,self.density)    
        self.density = np.dot(self.density,np.conj((shift_operator.shift).T)) 
        
        
def ElephantFermionShiftOperator(lattice,fermion,time,j):
    
    ''' Function that returns the fermion shift operator to the elephant quantum
        walk. 
    '''
        
    ''' The first parameter must be the lattice, the second the fermion, the 
        third the time in which the shift operator will be defined and the 
        last the deltas defining the shift kraus operator. 
    '''

    dimension = lattice.dimension
    size = lattice.size
    pos_basis = lattice.pos_basis

    f = [fermion.up,fermion.down]        
    shift_dimension = (size**dimension)*(2**dimension)        

    # Defines the type and size of the fermion shift operator.
    shift = sparse.lil_matrix((shift_dimension,shift_dimension),dtype=complex)

    mem_state = memory_ket(time,size//2)
    mem_op = sparse.kron(mem_state,np.conj(mem_state.T))      
            
    ''' In the below loop we take all possible configurations of fermion 
    spin states, 2^(dimension). We associate to a binary number (string) 
    every configuration, with 0 to a spin up and 1 to a spin down. So the 
    up,up,up state will be associated with 000, while up, down, up is asso-
    ciated with 010.
    '''  
        
    for i in range(0,2**dimension):
    
        # Position part of the shift operator.
        pos_shift = sparse.lil_matrix((size**(dimension),size**(dimension)),dtype=complex)
        # Binary number associated with the configuration.
        b = bin(i)[2:]
 
        # Here we attach zeros to the string to match the number of 
        # spins.
        if len(b) < dimension: 
            for k in range(0,(dimension-len(b))):
                b = '0' + b

        # def. spin state, e.g. |up>\otimes|\down>... 
        spin = f[int(b[0])] 
        for k in b[1:]:
            spin = np.kron(spin,f[int(k)])

        # Loop that go through every position basis element.
        for pos in pos_basis:

            old_pos = pos[1]    # Old position ket.
            new_pos = 1 # Scalar for the initial tensor product.

            for x in range(0,dimension):
                old_p = pos[0][x]   # Old position.
                # l_j -> l_j + (-1)**(spin)*delta^j_t
                new_p = old_p + (-1)**(int(b[x]))*j[x] 
                        
                # Conditional on the borders, imposing ciclic condi-
                # tions, maintaining the unitarity.
                if (new_p) <= (size//2) and new_p >= -(size//2): 
                    npk = position_ket(new_p,size)      
                    new_pos = sparse.kron(new_pos,npk)
                                
                else:
                    npk = position_ket(new_p - (new_p/np.linalg.norm(new_p))*size,size)
                    new_pos = sparse.kron(new_pos,npk)
                        
            # Summing the position shift operator part.          
            pos_shift = pos_shift + np.dot(new_pos,(old_pos.T))

        # \ket{spin} --> \ket{\spin}\bra{\spin} (operator)    
        spin_op = np.dot(spin,np.conj(spin.T))  

        new_shift = sparse.kron(pos_shift,spin_op)
        shift = shift + new_shift

    shift = sparse.kron(shift,mem_op)

    return shift                         
        
class ElephantWalker:
    
    ''' Class that defines the elephant walker.
    '''
    
    def __init__(self,spin_init_state,lattice,q,p):
        
        ''' The first parameter must be the spin initial state, accordingly
            with the dimension of the problem. The second the lattice in 
            which the elephant walks. The third is the probability that +1 is   
            sorted in the first time step, and the last that the elephant will 
            use the same shift from a past instant, while 1-p is the prob. of 
            using the opposite.  
        '''

        size = lattice.size
        dimension = lattice.dimension
        
        # q is the probability of sorting +1 in the first step.
        self.q = q
        # p is the probability of using +\delta_t sorted.
        self.p = p
        # Makes the initial position state in the center of the lattice.
        pos_state = position_ket(0,lattice.size)

        for i in range(0,lattice.dimension-1):
            pos_state = sparse.kron(pos_state,position_ket(0,lattice.size),format='lil')
        # Initial spin state of the walker.
        spin_state = sparse.lil_matrix(spin_init_state,dtype=complex)
        mem_state = memory_ket(0,size//2) 
        # |state> = |pos> \otimes |spin> \otimes |mem> .
        state = sparse.kron(pos_state,spin_state)
        state = sparse.kron(state,mem_state)
        # \rho = |state><state|.
        self.density = np.dot(state,np.conj(state.T))
        
        # We save the elephant shift operator with the elephant in order to not
        # redefine him entirely every time step.
        shift_dimension = size**(dimension)*2**(dimension)*(size//2)
        self.elephant_shift = sparse.lil_matrix((shift_dimension,shift_dimension),dtype= complex)

        # Defines a list that saves the displacements of every time step.
        self.memory = []    
    
    def walk(self,coin,lattice,fermion,time):
        
        ''' Method that makes the elephant walker walk in one time step. The 
        first parameter must be the coin(s) that will be used, the second the 
        lattice and the third and last a fermion and the time to define the 
        shift operator. 
        '''
        
        dimension = lattice.dimension
        size = lattice.size
        shift_dim = (size**dimension)*(2**dimension)*(size//2)

        self.density = coin.toss(self.density,lattice)
 
        ''' List of the new memory comb. to define the new shift operator.
        We have to consider this if there are independent memories for all
        directions.
        '''  
        memory_combinations = [] 
        
        if time == 0: # In the first time step the elephant has no memory.
            deltas = []
            # Loop that selects the displacements.
            for i in range(0,dimension): 
                
                # If the random number in [0,1] is <= q delta = +1.
                if np.random.random()<=self.q[i]: deltas.append(+1)
                # Else delta = -1    
                else: deltas.append(-1)

            # Save the displacements in the elephant memory.        
            self.memory.append(deltas) 

            # In this case we dont have to make any combinations.
            memory_combinations.append(self.memory[0])

            a = lattice
            b = fermion
            e_s = ElephantFermionShiftOperator(a,b,time,memory_combinations[0])
            # S_E\rhoS_E^\dagger    
            self.density = np.dot(e_s,self.density)
            self.density = np.dot(self.density,np.conj(e_s.T))
            self.elephant_shift = e_s

        else:
            
            ''' Here we have to consider if the dimension is greater than one 
            because if it isnt we dont have to make any combination of the del-
            -tas, as we have just one dimension. 
            '''

            if dimension > 1 :
                # list of the n-tuples of coordinates
                combinations = [] 
                
                ''' Loop that takes every n-tuple of deltas in the memory and 
                makes combinations n to n with the new n-tuple, where the n is
                the dimension of the problem and the combinations do not repeat
                deltas in the same direction. 
                '''
    
                for i in self.memory[:-1]:
                    coordinates = np.array([i,self.memory[-1]]) 
                    
                    for j in range(0,dimension):
                        combinations.append(coordinates[:,j])
                    
                # This list comprehension makes every combination of n elements
                # of the n-tuples in combinations list.
                [memory_combinations.append(p) for p in product(*combinations)]

                # Remove the first element, that always will be repeated.    
                memory_combinations.pop(0) 
                
            else:
                # Using all the memory for testing 
                memory_combinations = self.memory[-2:]
                
            # Below we define the elephant fermion shift operator accordingly 
            # with the time step.

            a = lattice
            b = fermion
            
            new_shift = sparse.lil_matrix((shift_dim,shift_dim),dtype=complex)

            # Loop that runs through all the memory
            for j in memory_combinations:

                new_shift = new_shift + ElephantFermionShiftOperator(a,b,time,j)

            # The shift operator will be the sum of all shift operators
            # with a given past displacement.
            self.elephant_shift = self.elephant_shift + new_shift
  
            self.density = np.dot(self.elephant_shift,self.density)
            self.density = np.dot(self.density,np.conj(self.elephant_shift.T))
            
#            self.density = self.density/np.trace(self.density.todense())
                
        # List that saves the actual random displacements +/1 for 
        # every direction.
        deltas = []
                      
        for i in range(0,dimension):
            # SEE HOW TO PASS A GENERAL FUNCTION! 

            ''' Here we sort randomly past instant accordingly with a uniform
                distribuition. To make the intervals in which a given time 
                step is chosen equal, we sort in the interval [0,t) and
                pickup the ceil of the sorted number. 
            '''

            past_t = np.random.uniform(0,time+1)
            if past_t == 0 : past_t = 1 
            if np.random.random()<=self.p[i]:
                deltas.append(self.memory[int(np.ceil(past_t))-1][i])
            else:
                deltas.append(-1*self.memory[int(np.ceil(past_t))-1][i])
                
        # Saving the news displacements in the memory.
        self.memory.append(deltas)    
