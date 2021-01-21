import numpy as np
import matplotlib.pyplot as plt
import qwalk
import statistics
import time
import os
from scipy import optimize
from datetime import datetime
from shutil import copy 

def common_qwalk(dimension,size,f,thetas,coin_init_state):

    main_dir = 'data/'+str(dimension)+'D_qwalks'
    date_time = datetime.now().strftime('%d%m%Y-%H:%M:%S')
    try:
        os.mkdir(main_dir+'/'+date_time)
        main_dir = main_dir+'/'+date_time
    except:
        os.mkdir(main_dir)
        main_dir = main_dir+'/'+date_time
        os.mkdir(main_dir)

    statistics_file = open(main_dir+'/statistics.txt','w+')
    statistics_file.write('#time \t mean position \t variances\n')

    copy('common.cfg', main_dir+'/parameters.txt')

    start_time = time.time()
    L = qwalk.Lattice(dimension,size)
    S = qwalk.FermionShiftOperator(L,f)
    c = qwalk.FermionCoin(thetas)
    W = qwalk.Walker(coin_init_state,L)

    for t in range(0,size//2):

        ps,mp,msq,sq = statistics.position_statistics(W.density,L,2)

        statistics_file.write('%f\t' %t)
        statistics_file.writelines('%f\t' %c for c in mp[0])
        statistics_file.writelines('%f\t' %c for c in sq[0])

        prob_dist_file = open(main_dir+'/pd_'+str(t),'w+')

        for i in range(0,dimension):
            prob_dist_file.writelines('%f\t' %c for c in ps[i])
            prob_dist_file.write('\n')

        W.walk(c,S,L,False)
#        print(np.trace(W.density.todense()))
        print('time: ',t, end='\r')   
    
    print("--- %s seconds ---" % (time.time() - start_time))


def elephant_qwalk(dimension,size,f,thetas,coin_init_state,q,p):
    
    main_dir = 'data/'+str(dimension)+'D_Eqwalks'
    date_time = datetime.now().strftime('%d%m%Y-%H:%M:%S')
    try:
        os.mkdir(main_dir+'/'+date_time)
        main_dir = main_dir+'/'+date_time
    except:
        os.mkdir(main_dir)
        main_dir = main_dir+'/'+date_time
        os.mkdir(main_dir)

    statistics_file = open(main_dir+'/statistics.txt','w+')
    statistics_file.write('#time \t mean position \t variances\n')

    copy('elephant.cfg', main_dir+'/parameters.txt')

    start_time = time.time()
    L = qwalk.Lattice(dimension,size)
    c = qwalk.FermionCoin(thetas)
    W = qwalk.ElephantWalker(coin_init_state,L,q,p)
    
    for t in range(0,size//2):
 
        ps,mp,msq,sq = statistics.position_statistics(W.density,L,2)
        statistics_file.write('%f\t' %t)
        statistics_file.writelines('%f\t' %c for c in mp[0])
        statistics_file.writelines('%f\t' %c for c in sq[0])
        
        prob_dist_file = open(main_dir+'/pd_'+str(t),'w+')

        for i in range(0,dimension):
            prob_dist_file.writelines('%f\t' %c for c in ps[i])
            prob_dist_file.write('\n')

        W.walk(c,L,f,t)
#        print(np.trace(W.density.todense()))
        print('time: ',t,end = '\r')     
    
    print("--- %s seconds ---" % (time.time() - start_time))

    positions = []
    for x in range(-(L.size//2),(L.size//2) +1):
        positions.append(x)
    positions = np.array(positions)

       def gaussian(x,sig):
           return np.sqrt(1/(2*np.pi*sig**2))*np.exp(-(x/sig)**2)

    fig = plt.figure(figsize=(16,9),dpi=200) 
    plt.title(r'$\theta = \pi/4,|\uparrow>, t ='+str(t)+'$',fontsize=16)
    k,= plt.plot(positions,position_statistics[t][0],lw=2,label='Simulation')
    plt.grid(linestyle='--')
    plt.xlabel(r'x',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r'Pr(x)',fontsize=16)
    plt.legend(handles=[k],fontsize=14)
    plt.savefig(main_dir+'/x_position_distribuition',bbox_inches='tight')
    plt.clf()

    def general_variance(x,a,b,c,d):
        return a*x**3+b*x**2+c*x+d

    fit_params, pcov = optimize.curve_fit(general_variance,time_steps,variance_x)
    a = round(fit_params[0],5)
    b = round(fit_params[1],5)
    c = round(fit_params[2],5)
    d = round(fit_params[3],5)

    plt.title(r'$\theta =  \pi/4,|\uparrow>$',fontsize=16)
    l, = plt.plot(time_steps,variance_x,label = 'Simulation',lw=2)
    fit_label = str(a)+r'$t^{3}$'+ '+' +str(b)+r'$t^{2}$'
    fit_label = fit_label + '+' + str(c)+r'$t$' + '+' + str(d)
    m, = plt.plot(time_steps,general_variance(time_steps,*fit_params),label = fit_label,ls='--')
    plt.grid(linestyle='--')
    plt.xlabel(r't',fontsize=16)
    plt.ylabel(r'$\sigma_{x}^{2}$(t)',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(handles=[l,m],fontsize=14)
    plt.savefig(main_dir+'/variance',bbox_inches='tight')
    plt.clf()


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
    up_state = np.cos(bloch_angle)*f.up
    down_state = np.exp(1j*phase_angle)*np.sin(bloch_angle)*f.down 
    coin_init_state = np.kron(coin_init_state,up_state + down_state)

thetas = np.array(thetas)
thetas = (np.pi/180)*thetas

try:
    os.mkdir('data')
except:
    pass    
    
if qwalk_type == 'common': 
    common_qwalk(dimension,size,f,thetas,coin_init_state)
else:
    q = []
    p = []
    
    for i in range (0,dimension):
        q.append(float(params[5][i]))
        p.append(float(params[6][i]))

    elephant_qwalk(dimension,size,f,thetas,coin_init_state,q,p)

