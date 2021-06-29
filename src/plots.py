import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def plot(main_dir, dimension, size, thetas, in_pos_var, bloch_angle, 
        phase_angle, q, trace_entang, tmax):
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
        k,= plt.plot(positions,probabilities[i,:],lw=2,label='Simulation', 
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
            if j != '0' and j != 'inf':
                log_variance.append(np.log(j))

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
        n, = plt.plot(time_steps,entang_entrop[:,i],
                      label = 'Entanglement Entropy',lw=2)
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
