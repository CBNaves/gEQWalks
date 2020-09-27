import numpy as np

def position_statistics(density_function,lattice,coin): 
    
    '''Function that returns an array of probabilities associated with a given time step in a given in all 
    directions, the mean positions, the mean square of the positions and the variance, respectively. The first 
    parameter must be the density function, the second the lattice in which he walks. The last parameter is the 
    dimension of the coin, i.e. 2 if a fermion, 3 if a boson coin. '''
    
    dimension = lattice.dimension
    size = lattice.size
    positions = [] # List to save every position n-tuple.
    for i in lattice.pos_basis:
        positions.append(i[0])
        
    positions = np.array(positions) 
    mean_pos = np.zeros((1,dimension)) # Array that stores the mean position in the lattice in every direction.
    mean_sq_pos = np.zeros((1,dimension)) # Array that stores the mean squared position in the lattice ''.
    
# To calculate the probabilities of beeing in one position we have first trace out the spins degree of freedom.
    pos_density_function = density_function.reshape(size**(dimension),coin**(dimension),size**(dimension),coin**(dimension))
    pos_density_function = np.trace(pos_density_function,axis1=1,axis2=3)
    
# The probabilities of beeing in one lattice site is given by the diagonal elements of the pos. density function.
# We take the real part of the diagonal elements because a non-zero, but negligible, imaginary part is always
# present.
    sites_probabilities = np.real(pos_density_function.diagonal()) 
    
    # Function that calculates the mean position and the mean of the square of the position.
    for i in range(0,size**(dimension)):
        
        # Mean positions, Square of the positions, Mean of the square of the positions.
        mean_pos = np.array([sum(x) for x in zip(mean_pos,sites_probabilities[i]*positions[i])])
        sq_pos = np.array([y**2 for y in positions[i]])
        mean_sq_pos = np.array([sum(z) for z in zip(mean_sq_pos,sites_probabilities[i]*sq_pos)])
    
    # Calculating sigma_squared
    sq_of_the_mean = np.array([x**2 for x in mean_pos]) # <x>^(2)
    sigma_squared = np.array([sum(x) for x in zip(mean_sq_pos,-1*sq_of_the_mean)]) # <x^(2)> - <x>^(2)
    
    # Now we create a list of probabilities vectors for every direction.
    reshape_array = (size*np.ones((1,2*dimension),int))[0] # Array for reshaping the density pos. function.
    one_d_probabilities = [] # list to save the one dimensional probabilities p(x),p(y),...
    
    # Loop that traces out all the other directions degree of freedom, for all directions.
    for j in range(0,dimension):
        
        one_d_trace_pos = pos_density_function.reshape(reshape_array) # Reshaping for tracing.
        
    # The axes of the one_d_traced_pos is of the form (size,size,size,size,size,size), if the dimension is 
    # equal 2. If we want the prob. of the first direction, so we have to trace the 1,4 axes and the 2,5 axes.
        for k in range(0,dimension-1):
            
            # To trace out axes that comes before the axes that we wanna keep.
            if j+1+dimension-k == 2*(dimension-k):  
                #one_d_trace_pos = (1/size)*np.trace(one_d_trace_pos,axis1=0,axis2=j+1)
                one_d_trace_pos = np.trace(one_d_trace_pos,axis1=0,axis2=j+1)
                
            else:
                #one_d_trace_pos = (1/size)*np.trace(one_d_trace_pos,axis1=j+1,axis2=j+1+dimension-k)
                one_d_trace_pos = np.trace(one_d_trace_pos,axis1=j+1,axis2=j+1+dimension-k)
                
        one_d_probabilities.append(np.real(np.diagonal(one_d_trace_pos)))
        
    return one_d_probabilities,mean_pos,mean_sq_pos,sigma_squared
    
def negativity(density,lattice,coin,negativity_dimension):
    
    ''' Function that returns the negativity of a bipartity system, two directions or a pos. in a direction and
    the coin. The first parameter is the density function and the second the lattice. The third parameter
    indicates if the negativity will be calculated with the coin or not with True or False, respectively.
    The last parameter gives the dimension numbers that will be considered in the calculation of the negativity.'''
    
    dimension = lattice.dimension
    size = lattice.size
    
    if coin == False:
        
        reshaped_density = density.reshape((size**(dimension),2**(dimension),size**(dimension),2**(dimension)))
        reshaped_density = np.trace(reshaped_density,axis1=1,axis2=3) # Tracing the coin degrees.
    
        pos_reshape_list = [] # List to save the reshape array.
        for i in range(0,2*dimension):
            pos_reshape_list.append(size)
        
        reshaped_density = reshaped_density.reshape(pos_reshape_list)
        
        # List to save the dimensions axis of the reshaped density func. that will be traced
        traced_axis_dimensions = [] 
        
        for j in range(0,dimension):
            if j+1 in negativity_dimension: pass
            else: traced_axis_dimension.append(j)
            
        for k in traced_axis_dimensions: # Loop that traces out 
            reshaped_density = np.trace(reshaped_density,axis1=k,axis2=(k+dimension))
        negativity_density = reshaped_density
        
        partial_transposed_density = np.transpose(negativity_density,axes=(0,3,2,1))
        partial_transposed_density = partial_transposed_density.reshape(size**(2),size**(2))
        
        eigen_val,eigen_vec = np.linalg.eig(partial_transposed_density)
        partial_transp_norm = 0
        for l in range(0,size**(2)):
            partial_transp_norm = partial_transp_norm + np.linalg.norm(np.real(eigen_val[l]))
        
    else:
    
        pos_reshape_list = []
        for i in range(0,4*dimension):
            if (i < dimension or i >= 2*dimension) and i < 3*dimension : pos_reshape_list.append(size)
            else: pos_reshape_list.append(2)
                
        reshaped_density = density.reshape(pos_reshape_list)
    
        traced_axis_dimensions = []
        for j in range(0,2*dimension):
            if j+1 in negativity_dimension: pass
            else: traced_axis_dimensions.append(j)
            
        for k in traced_axis_dimensions: # ERRADO!!!
            reshaped_density = np.trace(reshaped_density,axis1=k,axis2=(k+dimension))
        negativity_density = reshaped_density
    
        partial_transposed_density = np.transpose(negativity_density,axes=(0,3,2,1))
        partial_transposed_density = partial_transposed_density.reshape(2*size,2*size)
    
        eigen_val,eigen_vec = np.linalg.eig(partial_transposed_density)
        partial_transp_norm = 0
        for l in range(0,2*size):
            partial_transp_norm = partial_transp_norm + np.linalg.norm(np.real(eigen_val[l]))
        
    negativity = (partial_transp_norm - 1)/2
    
    return negativity   
