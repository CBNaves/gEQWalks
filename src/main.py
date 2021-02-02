import numpy as np
import matplotlib.pyplot as plt
import qwalk
import statistics
import time
import os
from scipy import optimize
from datetime import datetime
from shutil import copy

def plot(main_dir, parameters):

    ''' Function that makes the plots for the final position probabilities  
        distribuitions and the variances, for every dimension.
        The first parameter is the directory string in which the plots will 
        be saved and the last the parameters for the plots label and titles.
    '''

    # Conditional for the qwalk type, so that we include the extra parameters 
    # in the title in the case of the elephant.
    if 'E' in main_dir:
        dimension,size,thetas,bloch_angle,phase_angle,q,p = parameters
        title_str = r'$ \Theta (s) = '+str(thetas)
        title_str = title_str +', \Omega (s) = '+str(bloch_angle)
        title_str = title_str +', \phi (s) = '+str(phase_angle)
        title_str = title_str +', q = '+str(q)+', p = '+str(p)+'$'
    else:
        dimension,size,thetas,bloch_angle,phase_angle = parameters
        title_str = r'$ \Theta (s) = '+str(thetas)
        title_str = title_str +', \Omega (s) = '+str(bloch_angle)
        title_str = title_str +', \phi (s) = '+str(phase_angle)+'$'
    
    # Reading the probabilities dist. and the variances from the files.    
    prob_dist_file = open(main_dir+'/pd_'+str(size//2 - 1),'r')
    statistics_file = open(main_dir+'/statistics.txt','r')
    
    probabilities = [] 
    statistics = []
    
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

    statistics = np.array(statistics)
    probabilities = np.array(probabilities)

    # List of the positions in the lattice for the plot.
    positions = []
    for x in range(-(size//2),(size//2)+1):
        positions.append(x)
    positions = np.array(positions)

#    def gaussian(x,sig):
#        return np.sqrt(1/(2*np.pi*sig**2))*np.exp(-(x/sig)**2)

    # Function to the variance fiting. The polinomial form is specific.
    def general_variance(x,a,b,c,d):
        return a*x**3+b*x**2+c*x+d

    # In the statistics file, every line contains the time step in the first 
    # column, mean position and variance respectively for all dim.
    # in the fashion (mp1,v1,mp2,v2,...) separated by \t.
 
    time_steps = statistics[:,0]

    for i in range(0,dimension):

        label_dimension = 'x_{'+str(i+1)+'}'
        fig = plt.figure(figsize=(16,9),dpi=200) 
        plt.title(title_str+r'$, t ='+str(size//2 - 1)+'$',fontsize=16)
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
        
        fit_params, pcov = optimize.curve_fit(general_variance,time_steps,variance)
        a = round(fit_params[0],5)
        b = round(fit_params[1],5)
        c = round(fit_params[2],5)
        d = round(fit_params[3],5)

        plt.title(title_str,fontsize=16)
        l, = plt.plot(time_steps,variance,label = 'Simulation',lw=2)
        fit_label = str(a)+r'$t^{3}$'+ '+' +str(b)+r'$t^{2}$'
        fit_label = fit_label + '+' + str(c)+r'$t$' + '+' + str(d)
        m, = plt.plot(time_steps,general_variance(time_steps,*fit_params),label = fit_label,ls='--')
        plt.grid(linestyle='--')
        plt.xlabel(r't',fontsize=16)
        plt.ylabel(r'$\sigma_{'+label_dimension+'}^{2}$(t)',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(handles=[l,m],fontsize=14)
        plt.savefig(main_dir+'/'+label_dimension+'variance',bbox_inches='tight')
        plt.clf()

    prob_dist_file.close()
    statistics_file.close()     

def common_qwalk(dimension,size,f,thetas,coin_init_state):

    ''' Function that simulates the common quantum walk. The parameters are
        the dimension of the lattice, its size, f is the fermion, thetas the
        list that specifies the coins operators, and the last is the coin
        inital state.

        Returns the simulation directory.
    '''

    # Main directory in which the simulation directory will be saved,
    # separated by qwalk type and dimension.
    main_dir = 'data/'+str(dimension)+'D_qwalks'
    # date and time for the directory of the simulation runned.
    date_time = datetime.now().strftime('%d%m%Y-%H:%M:%S')

    try:
        os.mkdir(main_dir+'/'+date_time)
        main_dir = main_dir+'/'+date_time

    # If the main_dir defined firstly doesnt exists, create.
    except:
        os.mkdir(main_dir)
        main_dir = main_dir+'/'+date_time
        os.mkdir(main_dir)
    
    # Creating the file that will save the mean positions and variances.    
    statistics_file = open(main_dir+'/statistics.txt','w+')
    # A copy of the parameters used in the simulation is made in the 
    # simulation directory.
    copy('common.cfg', main_dir+'/parameters.txt')

    start_time = time.time() # Start time of the simulation.
    L = qwalk.Lattice(dimension,size) # Lattice.
    S = qwalk.FermionShiftOperator(L,f) # Shift Operator.
    c = qwalk.FermionCoin(thetas)   # Coin operators.
    W = qwalk.Walker(coin_init_state,L) # Walker.

    # Loop until the border of the lattice is reached.
    for t in range(0,size//2):

        ps,mp,msq,sq = statistics.position_statistics(W.density,L,2)

        statistics_file.write('%f\t' %t)
        statistics_file.writelines('%f\t' %c for c in mp[0])
        statistics_file.writelines('%f\t' %c for c in sq[0])
        statistics_file.write('\n')
        
        # For every time step a file to save the probabilities is created.
        prob_dist_file = open(main_dir+'/pd_'+str(t),'w+')
        
        # We save the probabilities for a given dimension in on line, the next
        # in the next line, and so forth.
        for i in range(0,dimension):
            prob_dist_file.writelines('%f\t' %c for c in ps[i])
            prob_dist_file.write('\n')

        W.walk(c,S,L,False) # A time step walk.
#        print(np.trace(W.density.todense()))
        print('time: ',t, end='\r')

    prob_dist_file.close()
    statistics_file.close()

    print("--- %s seconds ---" % (time.time() - start_time))
    return(main_dir)

def elephant_qwalk(dimension,size,f,thetas,coin_init_state,q,p):

    ''' Function that simulates the elephant quantum walk. The parameters are
        the dimension of the lattice, its size, f is the fermion, thetas the
        list that specifies the coins operators, coin_init_state is the inital
        state, q the probability of sorting +1 in t=0 and p the probability of
        using the same displacement sorted.

        Returns the simulation directory.
    '''
    
    # Main directory in which the simulation directory will be saved,
    # separated by qwalk type and dimension.
    main_dir = 'data/'+str(dimension)+'D_Eqwalks'
    # Date and time to name the simulation directory.
    date_time = datetime.now().strftime('%d%m%Y-%H:%M:%S')

    try:
        os.mkdir(main_dir+'/'+date_time)
        main_dir = main_dir+'/'+date_time

    # If the main directory doesnt exists, create.
    except:
        os.mkdir(main_dir)
        main_dir = main_dir+'/'+date_time
        os.mkdir(main_dir)
    
    # Creating the file in which the statistics will be saved.
    statistics_file = open(main_dir+'/statistics.txt','w+')
    # Copyng the parameters used in to the simulation directory.
    copy('elephant.cfg', main_dir+'/parameters.txt')

    start_time = time.time() # Start time of the simulation.
    L = qwalk.Lattice(dimension,size) # Lattice. 
    c = qwalk.FermionCoin(thetas)   # Coin operators.
    W = qwalk.ElephantWalker(coin_init_state,L,q,p) # Walker.
    
    for t in range(0,size//2):
 
        ps,mp,msq,sq = statistics.position_statistics(W.density,L,2)

        statistics_file.write('%f\t' %t)
        statistics_file.writelines('%f\t' %c for c in mp[0])
        statistics_file.writelines('%f\t' %c for c in sq[0])
        statistics_file.write('\n')
        
        # For every time step a file to save the probabilities is created.
        prob_dist_file = open(main_dir+'/pd_'+str(t),'w+')

        # We save the probabilities for a given dimension in on line, the next
        # in the next line, and so forth.
        for i in range(0,dimension):
            prob_dist_file.writelines('%f\t' %c for c in ps[i])
            prob_dist_file.write('\n')
        
        print(np.trace(W.density.todense()))
        W.walk(c,L,f,t) # A time step walk.        
#        print('time: ',t,end = '\r')
     
    prob_dist_file.close()
    statistics_file.close()

    print("--- %s seconds ---" % (time.time() - start_time))
    return(main_dir)

qwalk_type = input('Enter the quantum walk type (common, elephant): ')

params = [x.split(' ')[2:] for x in open(qwalk_type+'.cfg').read().splitlines()]
dimension = int(params[0][0])
size = int(params[1][0])

thetas = []
coin_init_state = 1

for i in range(0,dimension):
    coin_type = params[2][i]
    if 'fermion' == coin_type:
        f = qwalk.FermionSpin()
    thetas.append(float(params[3][i]))

for i in range (0,dimension,2):

    bloch_angle = float(params[4][i])
    phase_angle = float(params[4][i+1])
    rad_ba = (np.pi/180)*bloch_angle
    rad_pa = (np.pi/180)*phase_angle
    up_state = np.cos(rad_ba)*f.up
    down_state = np.exp(1j*rad_pa)*np.sin(rad_ba)*f.down 
    coin_init_state = np.kron(coin_init_state,up_state + down_state)

thetas = np.array(thetas)

try:
    os.mkdir('data')
except:
    pass    
    
if qwalk_type == 'common': 

    parameters = [dimension, size, thetas, bloch_angle, phase_angle]
    thetas = (np.pi/180)*thetas
    main_dir = common_qwalk(dimension,size,f,thetas,coin_init_state)
    plot(main_dir, parameters)

else:

    q = []
    p = []
    
    for i in range (0,dimension):
        q.append(float(params[5][i]))
        p.append(float(params[6][i]))

    parameters = [dimension, size, thetas, bloch_angle, phase_angle, q, p]
    thetas = (np.pi/180)*thetas
    main_dir = elephant_qwalk(dimension,size,f,thetas,coin_init_state,q,p)
    plot(main_dir, parameters)
