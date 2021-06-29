import numpy as np
import matplotlib.pyplot as plt
import gEQWalks
import statistics
import plots
import time
import os
import sys
from scipy import optimize
from datetime import datetime
from shutil import copy

            
def gEQWalk(dimension, size, thetas, in_pos_var, coin_type, bloch_angle, 
            phase_angle, q, memory_dependence, trace_entang):
    """ Function that simulates the generalized elephant quantum walk.
    The parameters are:
        dimension (int): dimension of the lattice;

        size (int): linear size of the lattice;
 
        thetas = [theta_1,theta_2,..](array,floats): is the thetas parameters 
        defining the coin operator for every coin;
 
        in_pos_var (float): initial position variance of the walker;

        coin_type (str): type of the coin, fermion or boson(not implemented);

        bloch_angle (array,floats): array of the polar angles that defines the
        coins initial state in the bloch sphere;

        phase_angle (array,floats): array of the phase angles that defines the
        coins initial state in the bloch sphere;
    
        q = [q_1,q_2,...] (array,floats): array of floats that defines the 
        parameters of the q-Exponential distribution;

        memory_dependence = [p_11,p_12,...] (array): array with the 
        interdependence probabilities between the displacements in every 
        direction;

        trace_entangle = [trace_dist,entang] (bool): list of boolean parameters
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
    entanglement_file = open(main_dir+'/entanglement_entropy.txt','w+')

    coin_instate = 1
    if trace_dist: ort_coin_instate = 1
    for i in range(0,dimension):
        rad_ba = (np.pi/180)*bloch_angle[i]
        rad_pa = (np.pi/180)*phase_angle [i]
 
        if 'fermion' == coin_type[i]:
            f = gEQWalks.FermionSpin()
            up_state = np.cos(rad_ba)*f.up
            down_state = np.exp(1j*rad_pa)*np.sin(rad_ba)*f.down
            coin_instate = np.kron(coin_instate,up_state + down_state)

            # Here we pick a orthogonal coin state to the initial coin state.
            if trace_dist:
                o_up_state = -1*np.exp(-1j*rad_pa)*np.sin(rad_ba)*f.up
                o_down_state = np.cos(rad_ba)*f.down
                ort_coin_instate = np.kron(ort_coin_instate,
                                        o_up_state + o_down_state)

    W = gEQWalks.Walker(in_pos_var, coin_instate, L, memory_dependence, q)

    if trace_dist: 
        trace_dist_file = open(main_dir+'/trace_distance.txt','w+')
        W_orthogonal = gEQWalks.Walker(in_pos_var, ort_coin_instate, L, 
                                        memory_dependence, q)
        # As we want the same evolution, the displacements in every time step
        # has to be the same for both states.
        W_orthogonal.displacements_vector = W.displacements_vector    

    print('Max. time: ', W.tmax,end = '\n')
    for t in range(0,W.tmax):
        ps,mp,msq,sq = statistics.position_statistics(W.state,L)
        entang_entrop = statistics.entanglement_entropy(W.state,L)

        statistics_file = open(main_dir+'/statistics.txt','a')
        statistics_file.write('%f\t' %t)
        statistics_file.writelines('%f\t' %i for i in mp[0])
        statistics_file.writelines('%f\t' %i[0] for i in sq)
        statistics_file.write('\n')

        entanglement_file = open(main_dir+'/entanglement_entropy.txt','a')
        entanglement_file.writelines('%f\t' %i for i in entang_entrop)
        entanglement_file.write('\n')
        
        # For every time step a .npy file to save the probabilities is created.
        np.save(main_dir+'/pd_'+str(t),  ps)

        print('time: ',t,end = '\r')

        if trace_dist:
            trace_dist_file = open(main_dir+'/trace_distance.txt','a')
            td = statistics.trace_distance(W.state,W_orthogonal.state,L)
            trace_dist_file.write('%f\n' %td)
            trace_dist_file.close()
            W_orthogonal.walk(c,L,entang,t)

#        trace = 0
#        for state in W.state:
#            trace = trace + np.real(np.dot(np.conj(state.T),state))
#        if trace < 9e-10 : print('Error! Not TP! \n')
       
        W.walk(c,L,entang,t)

    entanglement_file.close()
    statistics_file.close()
    print("--- %s seconds ---" % (time.time() - start_time))

    return(main_dir,W.tmax)

if __name__ == '__main__':
    params = [x.split(' ')[2:] for x in open('gEQW.cfg').read().splitlines()]
    # The order in which the parameters are readed are specified by the cfg file.
    dimension = int(params[0][0])
    size = int(params[1][0])

    thetas = []
    coin_type = []
    bloch_angle = []
    phase_angle = []
    in_pos_var = []

    for i in range(0,dimension):
        coin_type.append(params[2][i])
        thetas.append(float(params[3][i]))
        in_pos_var.append(float(params[4][i]))

    in_pos_var = np.array(in_pos_var)

    for i in range (0,2*dimension,2):
        bloch_angle.append(float(params[5][i]))
        phase_angle.append(float(params[5][i+1]))

    thetas = np.array(thetas)

    try:
        os.mkdir('data')
    except:
        pass    
    
    q = []
    for i in range (0,dimension):
        q.append(float(params[6][i]))

    memory_dependence = params[7]
    for i in range(0,len(memory_dependence)):
        memory_dependence[i] = float(memory_dependence[i])
    memory_dependence = np.array(memory_dependence)
    memory_dependence = memory_dependence.reshape((dimension,dimension))

    trace_dist = params[8][0]
    if trace_dist == 'False': trace_dist = False
    elif trace_dist == 'True': trace_dist = True

    entang = params[9][0]
    if entang == 'False': entang = False
    elif entang == 'True': entang = True

    trace_entang = [trace_dist,entang]

    main_dir,tmax = gEQWalk(dimension, size, thetas, in_pos_var, coin_type, 
                        bloch_angle, phase_angle, q, memory_dependence, 
                        trace_entang)

#    plots.plot(main_dir, dimension, size, thetas, in_pos_var, bloch_angle, 
#              phase_angle, q, trace_entang, tmax)

