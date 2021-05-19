import numpy as np
import matplotlib.pyplot as plt
import gEQWalks
import statistics
import time
import os
import sys
from scipy import optimize
from datetime import datetime
from shutil import copy

def plot(main_dir, dimension, size, thetas, in_pos_var, bloch_angle, 
        phase_angle, q, trace_entang, tmax):

    ''' Function that makes the plots for the final position probabilities  
        distribuitions and the variances,entanglement entropy for every
        dimension. The first parameter is the directory string in which the
        plots will be saved and the last the parameters for the plots label 
        and titles.
    '''

    # trace_dist is a boolean parameter to choose if plot the trace distance
    # graphs.
    trace_dist, entang = trace_entang
    # String that specifies the q's used.
    str_q = '['
    for i in q:
        if i > 10**(3): str_q = str_q + '\infty ,'
        else: str_q = str_q + str(i)+' ,'
    
    # Removing the last comma.
    str_q = str_q[0:np.size(str_q)-2]
    str_q = str_q +']'
        
    title_str = r'$ \Theta (s) = '+str(thetas)
    title_str = title_str +', \Omega (s) = '+str(bloch_angle)
    title_str = title_str +', \phi (s) = '+str(phase_angle)
    title_str = title_str +', q = '+str_q+'$'
    
    prob_dist_file = open(main_dir+'/pd_'+str(tmax-1),'r')
    statistics_file = open(main_dir+'/statistics.txt','r')
    entanglement_file = open(main_dir+'/entanglement_entropy.txt','r')
    
    probabilities = [] 
    statistics = []
    entang_entrop = []    

    # Here we read the lines and separate the elements not including spaces.
    for x in prob_dist_file.readlines():
        d_prob = []
        for y in x.split('\t'):
            if y != '\n': d_prob.append(float(y))
        probabilities.append(d_prob)

    for x in statistics_file.readlines():
        t_stat = []
        for y in x.split('\t'):
            if y != '\n': t_stat.append(float(y))
        statistics.append(t_stat)

    for x in entanglement_file.readlines():
        ent_data = []
        for y in x.split('\t'):
            if y != '\n': ent_data.append(float(y))
        entang_entrop.append(ent_data)
 
    statistics = np.array(statistics)
    probabilities = np.array(probabilities)
    entang_entrop = np.array(entang_entrop)

    # List of the positions in the lattice for the plot.
    positions = []
    for x in range(-(size//2),(size//2)+1):
        positions.append(x)
    positions = np.array(positions)

    # Function to the log(variance) fiting. As we just want the t expoent, 
    # we fit in a straight line.
    def general_variance(x,a,b):
        return a*x+ b

    # In the statistics file, every line contains the time step in the first 
    # column, mean position and variance respectively for all dim.
    # in the fashion (mp1,v1,mp2,v2,...) separated by \t.
 
    time_steps = statistics[:,0]
    time_steps = np.array(time_steps)
    log_times = np.log(time_steps[1:])

    for i in range(0,dimension):

        label_dimension = 'x_{'+str(i+1)+'}'
        coin_dimension = 'c_{'+str(i+1)+'}'

        fig = plt.figure(figsize=(16,9),dpi=200) 
        plt.title(title_str+r'$, t ='+str(tmax-1)+'$',fontsize=16)
        k,= plt.plot(positions,probabilities[i,:],lw=2,label='Simulation')
        plt.grid(linestyle='--')
        plt.xlabel(r'$'+label_dimension+'$',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel(r'$Pr('+label_dimension+')$',fontsize=16)
        plt.legend(handles=[k],fontsize=14)
        save_str = main_dir+'/'+label_dimension+'_position_distribuition'        
        plt.savefig(save_str,bbox_inches='tight')
        plt.clf()
        
        variance = statistics[:,dimension+1+i]
        log_variance = [] 
        for j in variance[1:]:
            if j != '0' and j != 'inf':
                log_variance.append(np.log(j))

        log_variance = np.array(log_variance) 

        try:        
            fit_params, pcov = optimize.curve_fit(general_variance,log_times,log_variance)
        except:
            fit_params = [0,0]
            pcov = [[0,0],[0,0]]

        a = round(fit_params[0],5)
        b = round(fit_params[1],5)
        

        plt.title(title_str,fontsize=16)
        l, = plt.plot(log_times,log_variance,label = 'Simulation',lw=2)
        fit_label = str(a)+r'$log(t)$'+ '+' +str(b)
        m, = plt.plot(log_times,general_variance(log_times,*fit_params),label = fit_label,ls='--')
        plt.grid(linestyle='--')
        plt.xlabel(r'log(t)',fontsize=16)
        plt.ylabel(r'$log(\sigma_{'+label_dimension+'}^{2}$(t))',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(handles=[l,m],fontsize=14)
        plt.savefig(main_dir+'/'+label_dimension+'_variance',bbox_inches='tight')
        plt.clf()
        
        plt.title(r'$'+coin_dimension+'$'+', '+title_str,fontsize=16)
        n, = plt.plot(time_steps,entang_entrop[:,i],label = 'Entanglement Entropy',lw=2)
        plt.grid(linestyle='--')
        plt.xlabel(r't',fontsize=16)
        plt.ylabel(r'$S_{E}^{('+str(i+1)+')}$(t)',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(handles=[n],fontsize=14)
        plt.savefig(main_dir+'/'+coin_dimension+'_entropy',bbox_inches='tight')
        plt.clf()

    prob_dist_file.close()
    statistics_file.close()
    entanglement_file.close()

    if trace_dist:

        trace_dist_vector = []
        trace_dist_file = open(main_dir+'/trace_distance.txt','r')
     
        for x in trace_dist_file.readlines():
            for y in x.split('\n'):
                if y != '': trace_dist_vector.append(float(y))

        # Loop to calculate the trace distance derivative.
        # Here we use the middle point rule for mid points, and difference
        # for the terminals.
        trace_dist_derivate = []
        for i in time_steps[:]:

            j = int(i)
            if i == 0:
                derivate = (trace_dist_vector[j+1] - trace_dist_vector[j])
            elif i == time_steps[-1]:
                derivate = (trace_dist_vector[j] - trace_dist_vector[j-1])                
            else:
                derivate = (trace_dist_vector[j+1] - trace_dist_vector[j-1])/2

            trace_dist_derivate.append(derivate)

        trace_dist_file.close()

        fig = plt.figure(figsize=(16,9),dpi=200) 
        plt.title(title_str,fontsize=16)
        k,= plt.plot(time_steps,trace_dist_vector,lw=2,label='Trace Distance')
        plt.grid(linestyle='--')
        plt.xlabel(r'$t$',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel(r'$D(\rho,\rho^{\perp})$',fontsize=16)
        plt.legend(handles=[k],fontsize=14)
        save_str = main_dir+'/trace_distance'        
        plt.savefig(save_str,bbox_inches='tight')
        plt.clf()
        
        fig = plt.figure(figsize=(16,9),dpi=200) 
        plt.title(title_str,fontsize=16)
        k,= plt.plot(time_steps,trace_dist_derivate,lw=2,label='Trace Distance Derivative')
        plt.grid(linestyle='--')
        plt.xlabel(r'$t$',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel(r'$d/dt D(\rho,\rho^{\perp})$',fontsize=16)
        plt.legend(handles=[k],fontsize=14)
        save_str = main_dir+'/trace_distance_derivative'        
        plt.savefig(save_str,bbox_inches='tight')
        plt.clf()
            
def gEQWalk(dimension, size, thetas, in_pos_var, coin_type, bloch_angle, 
            phase_angle, q, memory_dependence, trace_entang):

    ''' Function that simulates the generalized elephant quantum walk. The      
        parameters are the dimension of the lattice, its size, coin_type is 
        the fermion, thetas the list that specifies the coins operators, bloch  
        and phase the angles that specifies the coin initial state, q is the q-
        exponential parameter and trace_dist the boolean parameter that 
        specifies if the trace distance between the coin state and a orthogo-   
        nal state will be 
        calculated.

        Returns the simulation directory.
    '''

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
    
    # Copying the parameters used in to the simulation directory.
    copy('gEQW.cfg', main_dir+'/parameters.txt')
    start_time = time.time() # Start time of the simulation.
    L = gEQWalks.Lattice(dimension,size) # Lattice.
    thetas = (np.pi/180)*thetas
    c = gEQWalks.FermionCoin(thetas,L)   # Coin operators.

    # Creating the file in which the statistics will be saved.
    statistics_file = open(main_dir+'/statistics.txt','w+')
    entanglement_file = open(main_dir+'/entanglement_entropy.txt','w+')

    coin_instate = 1
    if trace_dist: ort_coin_istate = 1
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
                ort_coin_state = np.kron(ort_coin_instate,
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
        
        # For every time step a file to save the probabilities is created.
        prob_dist_file = open(main_dir+'/pd_'+str(t),'w+')

        # We save the probabilities for a given dimension in on line, the next
        # in the next line, and so forth.
        for i in range(0,dimension):

            prob_dist_file.writelines('%f\t' %c for c in ps[i])
            prob_dist_file.write('\n')

        prob_dist_file.close()

        print('time: ',t,end = '\r')

        if trace_dist:

            trace_dist_file = open(main_dir+'/trace_distance.txt','a')
            td = statistics.trace_distance(W.state,W_orthogonal.state,L)
            trace_dist_file.write('%f\n' %td)
            trace_dist_file.close()
            W_orthogonal.walk(c,L,entang,t)

        trace = 0 # Checking if the trace of the density op. remains one.
#        for state in W.state:
#            trace = trace + np.real(np.dot(np.conj(state.T),state))
#        if trace < 9e-10 : print('Error! Not TP! \n')
       
        W.walk(c,L,entang,t) # A time step walk.

    entanglement_file.close()
    statistics_file.close()
    print("--- %s seconds ---" % (time.time() - start_time))

    return(main_dir,W.tmax)

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

plot(main_dir, dimension, size, thetas, in_pos_var, bloch_angle, phase_angle, 
    q, trace_entang, tmax)

