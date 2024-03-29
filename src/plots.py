import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import optimize

def plot(main_dir, dimension, size, thetas, in_pos_var, coin_instate_coeffs, 
         displacement_functions, displacement_parameters,trace_entang, tmax):
    """ Function that makes the plots for the final position probabilities  
    distribuitions and the variances,entanglement entropy for every
    dimension. The first parameter is the directory string in which the
    plots will be saved and the last the parameters for the plots label 
    and titles.
    """

    # trace_dist is a boolean parameter to choose if plot the trace distance
    # graphs.

    trace_dist, entang = trace_entang
    # String that specifies the q's used.

    coin_coeffs_str = '['
    for coin_coeff in coin_instate_coeffs:
        coin_coeffs_str += str(coin_coeff)+','
    coin_coeffs_str = coin_coeffs_str[:-1] + ']'

    title_str = r'$ \Theta (s) = '+str(thetas)
    title_str = title_str +', |\psi_c(0)\\rangle = '+coin_coeffs_str+'$'
    title_str = title_str +r', disp. functions $ ='+str(displacement_functions)+'$'
    title_str = title_str +r', steps params $= '+str(displacement_parameters)+'$'
    
    probabilities = np.load(main_dir+'/pd_'+str(tmax-1)+'.npy')
    statistics_file = open(main_dir+'/statistics.txt','r')
    coin_statistics_file = open(main_dir+'/coin_statistics.txt','r')
    
    statistics = []
    cstatistics = []    

    for x in statistics_file.readlines():
        t_stat = []
        for y in x.split('\t'):
            if y != '\n': t_stat.append(float(y))
        statistics.append(t_stat)

    for x in coin_statistics_file.readlines():
        ent_data = []
        for y in x.split('\t'):
            if y != '\n': ent_data.append(float(y))
        cstatistics.append(ent_data)
 
    statistics = np.array(statistics)
    cstatistics = np.array(cstatistics)

    # List of the positions in the lattice for the plot.
    positions = []
    for x in range(-(size//2),(size//2)+1):
        positions.append(x)
    positions = np.array(positions)

    # Function to the log(variance) fiting. As we just want the t expoent, 
    # we fit in a straight line.
    def linear_func(x,a,b):
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
        if dimension > 1:
            k,= plt.plot(positions,np.sum(probabilities, axis=dimension-i-1),
                         lw=2,label='Simulation', color = 'Blue')
        else:
            k,= plt.plot(positions, probabilities,lw=2,label='Simulation',
                         color = 'Blue')
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
            if j != 0. and j != 'inf':
                log_variance.append(np.log(j))
            else:
                log_variance.append(0)

        log_variance = np.array(log_variance) 

        try:        
            fit_params, pcov = optimize.curve_fit(linear_func,log_times,
                                                  log_variance)
        except:
            fit_params = [0,0]
            pcov = [[0,0],[0,0]]

        a = round(fit_params[0],5)
        b = round(fit_params[1],5)
        

        plt.title(title_str,fontsize=16)
        l, = plt.plot(log_times,log_variance,label = 'Simulation',lw=2,
                      color = 'Blue')
        fit_label = str(a)+r'$log(t)$'+ '+' +str(b)
        m, = plt.plot(log_times,linear_func(log_times,*fit_params),
                      label = fit_label, ls='--', color = 'Black')
        plt.grid(linestyle='--')
        plt.xlabel(r'log(t)',fontsize=16)
        plt.ylabel(r'$log(\sigma_{'+label_dimension+'}^{2}$(t))',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(handles=[l,m],fontsize=14)
        plt.savefig(main_dir+'/'+label_dimension+'_variance',
                   bbox_inches='tight')
        plt.clf()
        
        plt.title(r'$'+coin_dimension+'$'+', '+title_str,fontsize=16)
        n, = plt.plot(time_steps,cstatistics[:,i],
                      label = 'Entanglement Entropy',lw=2)
        plt.grid(linestyle='--')
        plt.xlabel(r't',fontsize=16)
        plt.ylabel(r'$S_{E}^{('+str(i+1)+')}$(t)',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(handles=[n],fontsize=14)
        plt.savefig(main_dir+'/'+coin_dimension+'_entropy',bbox_inches='tight')
        plt.clf()

    statistics_file.close()
    coin_statistics_file.close()

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

        fig, ax1 = plt.subplots(figsize=(16,9),dpi=200) 
        plt.title(title_str,fontsize=16)
        ax1.plot(time_steps,trace_dist_vector, lw=2, label='Trace Distance', 
                 color = 'black')
        ax1.grid(linestyle='--')
        ax1.set_xlabel(r'$t$',fontsize=16)
        ax1.tick_params(axis = 'x', labelsize = 14)
        ax1.tick_params(axis = 'y', labelsize = 14)
        ax1.set_ylabel(r'$D(\rho,\rho^{\perp})$',fontsize=16)
        ax1.legend(fontsize=14)

        ax2 = fig.add_axes([0.55,0.5,0.3,0.3])
        ax2.plot(time_steps,trace_dist_derivate,lw=2,
                label='Trace Distance Derivative', color = 'red')
        ax2.grid(linestyle='--')
        ax2.legend(fontsize = 14)
        ax2.set_xlabel(r'$t$',fontsize=16)
        ax2.tick_params(axis = 'x', labelsize=14)
        ax2.tick_params(axis = 'y', labelsize=14)
        ax2.set_ylabel(r'$d/dt D(\rho,\rho^{\perp})$',fontsize=16)
        save_str = main_dir+'/trace_distance'
        plt.savefig(save_str, bbox_inches = 'tight')
        plt.clf()

