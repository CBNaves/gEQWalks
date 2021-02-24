import numpy as np

def position_statistics(density_function,lattice,coin): 
    
    '''Function that returns an array of probabilities associated with a given 
    time step in all directions, the mean positions, the mean square
    of the positions and the variance respectively. The first parameter must be
    the density function, the second the lattice in which he walks. The last 
    parameter is the dimension of the coin, i.e. 2 if a fermion, 3 if a boson 
    coin. 
    '''
    dense_df = density_function.todense() # Dense matrix of the dense function.
    dense_df = np.array(dense_df)
    dimension = lattice.dimension
    size = lattice.size

    positions = [] # List to save every position n-tuple.
    for i in lattice.pos_basis:
        positions.append(i[0])
        
    positions = np.array(positions)
  
    # Array that stores the mean position in the lattice in every direction. 
    mean_pos = np.zeros((1,dimension))
    # Array that stores the mean squared position in the lattice ''. 
    mean_sq_pos = np.zeros((1,dimension)) 
    
    # To calculate the probabilities of beeing in one position we have first 
    # trace out the spins degree of freedom.
    a = size**(dimension)
    b = coin**(dimension)

    dense_df = dense_df.reshape(a,b,a,b)
    pos_density_function = np.trace(dense_df,axis1=1,axis2=3)
    
    ''' The probabilities of beeing in one lattice site is given by the diago-
    -nal elements of the pos. density function. We take the real part of the 
    diagonal elements because a non-zero, but negligible, imaginary part is 
    always present. 
    '''

    sites_probabilities = np.real(pos_density_function.diagonal()) 
    
    # Function that calculates the mean position and the mean of the square of 
    # the position.
    for i in range(0,size**(dimension)):
        
        # Mean positions, Square of the positions, 
        # mean of the square of the positions.
        mp = [sum(x) for x in zip(mean_pos,sites_probabilities[i]*positions[i])]
        mean_pos = np.array(mp)
        sq_pos = np.array([y**2 for y in positions[i]])
        msp = [sum(z) for z in zip(mean_sq_pos,sites_probabilities[i]*sq_pos)]
        mean_sq_pos = np.array(msp)
    
    # Calculating sigma_squared
    sq_of_the_mean = np.array([x**2 for x in mean_pos]) # <x>^(2).
    # <x^(2)> - <x>^(2).
    sigma_squared = np.array([sum(x) for x in zip(mean_sq_pos,-1*sq_of_the_mean)]) 
    
    # Array for reshaping the density pos. function.
    reshape_array = (size*np.ones((1,2*dimension),int))[0] 
    # List to save the one dimensional probabilities p(x),p(y),...
    one_d_probabilities = []
    
    # Loop that traces out all the other directions degree of freedom, 
    # for all directions.
    for j in range(0,dimension):

        # Reshaping for tracing.
        one_d_trace_pos = pos_density_function.reshape(reshape_array) 
        
        ''' The axes of the one_d_traced_pos is of the form 
        (size,size,size,size,size,size), if the dimension is equal 2. If we 
        want the prob. of the first direction, so we have to trace the 1,4 
        axes and the 2,5 axes.
        '''
        for k in range(0,dimension-1):
            
            # To trace out axes that comes before the axes that we wanna keep.
            if j+1+dimension-k == 2*(dimension-k):  
                one_d_trace_pos = np.trace(one_d_trace_pos,axis1=0,axis2=j+1)
                
            else:
                one_d_trace_pos = np.trace(one_d_trace_pos,axis1=j+1,axis2=j+1+dimension-k)
                
        one_d_probabilities.append(np.real(np.diagonal(one_d_trace_pos)))
        
    return one_d_probabilities,mean_pos,mean_sq_pos,sigma_squared
    
def negativity(density,lattice,coin,negativity_dimension):
    
    ''' Function that returns the negativity of a bipartity system, two direc-
    -tions or a pos. in a direction and the coin. The first parameter is the 
    density function and the second the lattice. The third parameter indicates 
    if the negativity will be calculated with the coin or not with True or 
    False, respectively. The last parameter gives the dimension numbers that 
    will be considered in the calculation of the negativity.
    '''
    
    dimension = lattice.dimension
    size = lattice.size
    dense_df = density.todense()
    dense_df = np.array(dense_df)
    if coin == False:
        a = size**dimension
        b =  2**(dimension)
        reshaped_density = dense_df.reshape((a,b,a,b))
        # Tracing the coin degrees.
        reshaped_density = np.trace(reshaped_density,axis1=1,axis2=3) 
    
        pos_reshape_list = [] # List to save the reshape array.
        for i in range(0,2*dimension):
            pos_reshape_list.append(size)
        
        reshaped_density = reshaped_density.reshape(pos_reshape_list)
        
        # List to save the dimensions axis of the reshaped density func. that 
        # will be traced
        traced_axis_dimensions = [] 
        
        for j in range(0,dimension):
            if j+1 in negativity_dimension: pass
            else: traced_axis_dimension.append(j)
            
        for k in traced_axis_dimensions: # Loop that traces out 
            reshaped_density = np.trace(reshaped_density,axis1=k,axis2=(k+dimension))
        negativity_density = reshaped_density
        
        # Partial transpose of the density function.
        pt_density = np.transpose(negativity_density,axes=(0,3,2,1))
        pt_density = pt_density.reshape(size**(2),size**(2))
        
        eigen_val,eigen_vec = np.linalg.eig(pt_density)
        pt_norm = 0
        for l in range(0,size**(2)):
            norm = np.linalg.norm(np.real(eigen_val[l]))
            pt_norm = pt_norm + norm
        
    else:
    
        pos_reshape_list = []
        for i in range(0,4*dimension):
            if (i < dimension or i >= 2*dimension) and i < 3*dimension : 
                pos_reshape_list.append(size)
            else:
                pos_reshape_list.append(2)
                
        reshaped_density = dense_df.reshape(pos_reshape_list)
    
        traced_axis_dimensions = []
        for j in range(0,2*dimension):
            if j+1 in negativity_dimension: pass
            else: traced_axis_dimensions.append(j)
            
        for k in traced_axis_dimensions: # ERRADO!!!
            reshaped_density = np.trace(reshaped_density,axis1=k,axis2=(k+dimension))
        negativity_density = reshaped_density
    
        pt_density = np.transpose(negativity_density,axes=(0,3,2,1))
        pt_density = pt_density.reshape(2*size,2*size)
    
        eigen_val,eigen_vec = np.linalg.eig(pt_density)
        pt_norm = 0
        for l in range(0,2*size):
            pt_norm = pt_norm + np.linalg.norm(np.real(eigen_val[l]))
 
    negativity = (pt_norm - 1)/2
    
    return negativity

def entanglement_entropy(density,lattice):
    
    ''' Function that returns the entanglement entropy of the coins degrees.
        The first parameter must be the density matrix and the second the
        lattice.
    '''
    
    dimension = lattice.dimension
    size = lattice.size
    dense_df = density.todense() # Dense density function.
    dense_df = np.array(dense_df)
   
    a = size**dimension
    b =  2**(dimension)
    reshaped_density = dense_df.reshape((a,b,a,b)) 
    
    # Performing the trace of the positions degrees.
    coins_density = np.trace(reshaped_density,axis1=0,axis2=2)

    # Reshaping the coin density in a suitable manner.
    coin_reshape_list = [2 for x in range(0,2**dimension)]
    coins_density = coins_density.reshape(coin_reshape_list)

    remaining_density = coins_density 
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
    dense_rho = rho.todense() # Dense density function.
    dense_rho = np.array(dense_rho)
    dense_sigma = sigma.todense() 
    dense_sigma = np.array(dense_sigma)
   
    a = size**dimension
    b =  2**(dimension)
    reshaped_rho = dense_rho.reshape((a,b,a,b))
    reshaped_sigma = dense_sigma.reshape((a,b,a,b))

    coin_rho = np.trace(reshaped_rho,axis1=0,axis2=2)
    coin_sigma = np.trace(reshaped_sigma,axis1=0,axis2=2)

    dif_density = coin_rho - coin_sigma
    dif_eigen_val, dif_eigen_vec = np.linalg.eig(dif_density)
    dif_eigen_val = np.real(dif_eigen_val)

    trace_dist = 0 
    for i in dif_eigen_val:
        trace_dist = trace_dist + (1/2)*np.linalg.norm(i)

    return trace_dist    
