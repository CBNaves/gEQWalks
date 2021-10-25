import numpy as np
import matplotlib.pyplot as plt
import gEQWalks
import statistics
import displacements_distributions
import plots
import time
import os
import sys
from scipy import optimize
from datetime import datetime
from shutil import copy

            
def gEQWalk(dimension, size, thetas, in_pos_var, coin_type, 
           coin_instate_coeffs, displacement_functions,
           displacement_parameters, trace_entang):
    """ Function that simulates the generalized elephant quantum walk.
    The parameters are:
        dimension (int): dimension of the lattice;

        size (int): linear size of the lattice;
 
        thetas = [theta_1,theta_2,..](array,floats): is the thetas parameters 
        defining the coin operator for every coin;
 
        in_pos_var (float): initial position variance of the walker;

        coin_type (str): type of the coin, fermion or boson(not implemented);

        coin_instate_coeffs (array, csingle): coefficients of the basis states
        in the initial coin state.
    
        displacement_functions (array, functions): array of the probability
        distribution functions used to generated the displacements.

        displacement_parameters (array): array of parameters for each 
        probability function used to generate the displacements.

        trace_entang = [trace_dist,entang] (bool): list of boolean parameters
        that specifies if the trace distance between the coin state and an 
        orthogonal one will be calculated and a entangling coin operator (2D)
        will be used.

    Returns a tuple with the simulation directory and the maximum simulated 
    time.
    """

    trace_dist, entang = trace_entang
    # Main directory in which the simulation directory will be saved,
    # separated by qwalk type and dimension.
    main_dir = 'data/'+str(dimension)+'D_gEQWalks'
    # Date and time to name the simulation directory.
    date_time = datetime.now().strftime('%d%m%Y-%H:%M:%S')

    try:
        os.mkdir(main_dir+'/'+date_time)
        main_dir = main_dir+'/'+date_time
    except:
        os.mkdir(main_dir)
        main_dir = main_dir+'/'+date_time
        os.mkdir(main_dir)
    
    copy('gEQW.cfg', main_dir+'/parameters.txt')
    start_time = time.time()
    L = gEQWalks.Lattice(dimension,size)
    thetas = (np.pi/180)*thetas
    c = gEQWalks.FermionCoin(thetas,L)

    statistics_file = open(main_dir+'/statistics.txt','w+')
    coin_statistics_file = open(main_dir+'/coin_statistics.txt','w+')

    norm = (1/np.sqrt(np.dot(np.conj(coin_instate_coeffs),coin_instate_coeffs.T)))
    coin_instate_coeffs = norm*coin_instate_coeffs
    W = gEQWalks.Walker(in_pos_var, coin_instate_coeffs, L, 
                        displacement_functions, displacement_parameters)

    if trace_dist: 
        trace_dist_file = open(main_dir+'/trace_distance.txt','w+')
        # Gram-Schmidt process about the |00..> state
        ort_cstate_coeffs = -1*coin_instate_coeffs
        # Check if the initial state is the |00..> and do the process about
        # the |010..> state
        if not coin_instate_coeffs[1:].any(axis = 0):
            ort_cstate_coeffs = np.conj(coin_instate_coeffs[1])*ort_cstate_coeffs
            ort_cstate_coeffs[1] = 1 + ort_cstate_coeffs[1]
        else:
            ort_cstate_coeffs = np.conj(coin_instate_coeffs[0])*ort_cstate_coeffs
            ort_cstate_coeffs[0] = 1 + ort_cstate_coeffs[0]
        norm = (1/np.sqrt(np.dot(np.conj(ort_cstate_coeffs),ort_cstate_coeffs.T)))
        ort_cstate_coeffs = norm*ort_cstate_coeffs

        W_orthogonal = gEQWalks.Walker(in_pos_var, ort_cstate_coeffs, L, 
                                       displacement_functions, 
                                       displacement_parameters)
        # As we want the same evolution, the displacements in every time step
        # has to be the same for both walkers.
        W_orthogonal.displacements_vector = W.displacements_vector    

    print('Max. time: ', W.tmax,end = '\n')
    for t in range(0,W.tmax):
        ps,mp,msq,sq = statistics.position_statistics(W.state,L)
        entang_entrop, negativity = statistics.coin_statistics(W.state,L)

        statistics_file = open(main_dir+'/statistics.txt','a')
        statistics_file.write('%f\t' %t)
        statistics_file.writelines('%f\t' %i for i in mp[0])
        statistics_file.writelines('%f\t' %i[0] for i in sq)
        statistics_file.write('\n')

        coin_statistics_file = open(main_dir+'/coin_statistics.txt','a')
        coin_statistics_file.writelines('%f\t' %i for i in entang_entrop)
        coin_statistics_file.write('%f' %negativity)
        coin_statistics_file.write('\n')
        
        # For every time step a .npy file to save the probabilities is created.
        np.save(main_dir+'/pd_'+str(t),  ps)

        print('time: ',t,end = '\r')

        if trace_dist:
            trace_dist_file = open(main_dir+'/trace_distance.txt','a')
            td = statistics.trace_distance(W.state,W_orthogonal.state,L)
            trace_dist_file.write('%f\n' %td)
            W_orthogonal.walk(c,L,entang,t)
########## trace condition check ############################################
#        trace = 0
#        for state in W.state:
#            trace = trace + np.real(np.dot(np.conj(state.T),state))
#        print(trace[0][0],'\n')
#        if trace[0][0] < 0.999999999 : print('Error! Not TP! \n')
#############################################################################
        W.walk(c,L,entang,t)

    coin_statistics_file.close()
    statistics_file.close()
    if trace_dist:
        trace_dist_file.close()
    print("--- %s seconds ---" % (time.time() - start_time))

    return(main_dir,W.tmax)

if __name__ == '__main__':
    parameters = []
    for x in open('gEQW.cfg').read().splitlines():
        if x[0] != '#':
            parameters.append(x.split(' ')[2:])

    # The order in which the parameters are readed are specified by the cfg file.
    dimension = int(parameters[0][0])
    size = int(parameters[1][0])

    thetas = []
    coin_type = []
    in_pos_var = []

    for i in range(0,dimension):
        coin_type.append(parameters[2][i])
        thetas.append(float(parameters[3][i]))
        in_pos_var.append(float(parameters[4][i]))
    in_pos_var = np.array(in_pos_var)
    thetas = np.array(thetas)

    coin_instate_coeffs = [complex(i) for i in parameters[5]]
    coin_instate_coeffs = np.array(coin_instate_coeffs)

    try:
        os.mkdir('data')
    except:
        pass

    displacement_functions = []
    for function in parameters[6]:
        displacement_function = displacements_distributions.functions[function]
        displacement_functions.append(displacement_function)    
    
    displacement_parameters = [float(i) for i in parameters[7]]

    trace_dist = parameters[8][0]
    if trace_dist == 'False': 
        trace_dist = False
    elif trace_dist == 'True': 
        trace_dist = True

    entang = parameters[9][0]
    if entang == 'False': 
        entang = False
    elif entang == 'True': 
        entang = True

    trace_entang = [trace_dist,entang]

    main_dir,tmax = gEQWalk(dimension, size, thetas, in_pos_var, coin_type, 
                            coin_instate_coeffs, displacement_functions, 
                            displacement_parameters, trace_entang)

    displacement_functions = parameters[6]
    plots.plot(main_dir, dimension, size, thetas, in_pos_var,
               coin_instate_coeffs, displacement_functions, 
               displacement_parameters, trace_entang, tmax)
