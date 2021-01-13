import numpy as np
import matplotlib.pyplot as plt
import qwalk
import statistics
import time
import os
from scipy import optimize

qwalk_type = input('Enter the quantum walk type (common,elephant): ')

params = [x.split(' ')[2:] for x in open(qwalk_type+'.cfg').read().splitlines()]
dimension = int(params[0][0])
size = int(params[1][0])
coin_type = params[2][0]
if 'fermion' == coin_type:
    f = qwalk.FermionSpin()

thetas = []

coin_init_state = 1    

if qwalk_type == 'common':
    for i in range (0,dimension):
        thetas.append(float(params[3][i]))
        if params[4][i] == 'up':
            coin_init_state = np.kron(coin_init_state,f.up)
        else:
            coin_init_state = np.kron(coin_init_state,f.down)
    thetas = np.array(thetas)
    thetas = (np.pi/180)*thetas

    start_time = time.time()
    L = qwalk.Lattice(dimension,size)
    S = qwalk.FermionShiftOperator(L,f)
    c = qwalk.FermionCoin(thetas)
    W = qwalk.Walker(coin_init_state,L)
    position_statistics = []
    mean_position = []
    mean_sq_position = []
    variance_x = []
    time_steps = []
    #walker_negativity

    for t in range(0,size//2):
        time_steps.append(t)
        ps,mp,msq,sq = statistics.position_statistics(W.density,L,2)
        position_statistics.append(ps)
        mean_position.append(mp)
        mean_sq_position.append(msq)
        variance_x.append(sq[0][0])
#        walker_negativity = statistics.negativity(W.density,L,True,[1,2])
#        statistics_file.write('%f\t' %t)
#        statistics_file.writelines('%f\t' %c for c in mp[0])
#        statistics_file.writelines('%f\t' %c for c in sq[0])
#        statistics_file.write('%f\n' %walker_negativity)
#        for i in range(0,2):
#            prob_dist_file.write('%f\t' %t)
#            prob_dist_file.writelines('%f\t' %c for c in ps[i])
#            prob_dist_file.write('\n')
        W.walk(c,S,L,False)
        print(np.trace(W.density.todense())),
#        print('time:',t)   
    
    print("--- %s seconds ---" % (time.time() - start_time))
              
else:
    q = []
    p = []
    coin_init_state = 1    
    for i in range (0,dimension):
        thetas.append(float(params[3][i]))
        if params[4][i] == 'up':
            coin_init_state = np.kron(coin_init_state,f.up)
        else:
            coint_init_state = np.kron(coin_init_state,f.down)
        q.append(float(params[5][i]))
        p.append(float(params[6][i]))
    thetas = np.array(thetas)
    thetas = (np.pi/180)*thetas
    #try:
        #os.mkdir('2D_Eqwalks')
    #except:
        #os.mkdir('2D_Eqwalks/s'+str(size))

    #statistics_file = open('2D_Eqwalks/s'+str(21)+'/statiscs.txt','w+')
    #prob_dist_file = open('2D_Eqwalks/s'+str(21)+'/pd.txt','w+')

    #statistics_file.write('#Parameters: Size ='+str(21)+', Coins_p = pi/4,pi/4, spins = up,up.\n')
    #statistics_file.write('#time \t mean position \t variances \t negativity \n')

    #prob_dist_file.write('#Parameters: Size ='+str(21)+', Coins_p = pi/4,pi/4, spins = up,up.\n')
    #prob_dist_file.write('#time \t p(x)\n#time \t p(y)...\n')
    start_time = time.time()
    L = qwalk.Lattice(dimension,size)
    c = qwalk.FermionCoin(thetas)
    W = qwalk.ElephantWalker(coin_init_state,L,q,p)
    position_statistics = []
    mean_position = []
    mean_sq_position = []
    variance_x = []
    time_steps = []
    #walker_negativity

    for t in range(0,size//2):
        time_steps.append(t)
        ps,mp,msq,sq = statistics.position_statistics(W.density,L,2)
        position_statistics.append(ps)
        mean_position.append(mp)
        mean_sq_position.append(msq)
        variance_x.append(sq[0][0])
#        walker_negativity = statistics.negativity(W.density,L,True,[1,2])
#        statistics_file.write('%f\t' %t)
#        statistics_file.writelines('%f\t' %c for c in mp[0])
#        statistics_file.writelines('%f\t' %c for c in sq[0])
#        statistics_file.write('%f\n' %walker_negativity)
#        for i in range(0,2):
#            prob_dist_file.write('%f\t' %t)
#            prob_dist_file.writelines('%f\t' %c for c in ps[i])
#            prob_dist_file.write('\n')
        W.walk(c,L,f,t)
        print(np.trace(W.density.todense())),
        print('time:',t)
        os.system('clear')     
    
    print("--- %s seconds ---" % (time.time() - start_time))

variance_x = np.array(variance_x)
time_steps = np.array(time_steps)
positions = []
for x in range(-(L.size//2),(L.size//2) +1):
    positions.append(x)
positions = np.array(positions)

#   def gaussian(x,sig):
#       return np.sqrt(1/(2*np.pi*sig**2))*np.exp(-(x/sig)**2)

fig = plt.figure(figsize=(16,9),dpi=200) 
plt.title(r'$\theta_x,\theta_y = \pi/4,|\uparrow,\uparrow>, t = 20$',fontsize=16)
k,= plt.plot(positions,position_statistics[t][0],lw=2,label='Simulation')
plt.grid(linestyle='--')
plt.xlabel(r'x',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel(r'Pr(x)',fontsize=16)
plt.legend(handles=[k],fontsize=14)
plt.savefig('1D_qwalks/x_position_distribuition_normlzd_02',bbox_inches='tight')
plt.clf()

def general_variance(x,a,b,c,d):
    return a*x**3+b*x**2+c*x+d

fit_params, pcov = optimize.curve_fit(general_variance,time_steps,variance_x)
a = round(fit_params[0],2)
b = round(fit_params[1],2)
c = round(fit_params[2],2)
d = round(fit_params[3],2)

plt.title(r'$\theta_x , \theta_y = \pi/4,|\uparrow,\uparrow>$',fontsize=16)
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
plt.savefig('1D_qwalks/variance_normalzd_02',bbox_inches='tight')
plt.clf()
