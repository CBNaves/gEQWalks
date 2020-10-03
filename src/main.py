import numpy as np
import matplotlib.pyplot as plt
import qwalk
import statistics
import time
from scipy import optimize

start_time = time.time()

L = qwalk.Lattice(1,5)
f = qwalk.FermionSpin()
S = qwalk.FermionShiftOperator(L,f)
test = np.array(S.shift.todense())
c = qwalk.FermionCoin([np.pi/4])
W = qwalk.ElephantWalker(f.up,L)
position_statistics = []
mean_position = []
mean_sq_position = []
variance_x = []
variance_y =[]
walker_negativity = []
time_steps = []

for t in range(0,50):
    #print(t)
    time_steps.append(float(t)) 
    ps,mp,msq,sv = statistics.position_statistics(W.density,L,2)
    position_statistics.append(ps)
    mean_position.append(mp)
    mean_sq_position.append(msq)
    variance_x.append(sv[0][0])
    #variance_y.append(sv[0][1])
    walker_negativity.append(statistics.negativity(W.density,L,True,[1,2]))
    W.walk(c,L,f,t)

print("--- %s seconds ---" % (time.time() - start_time))
variance_x = np.array(variance_x)
variance_y = np.array(variance_y)
time_steps = np.array(time_steps)

positions = []
for x in range(-(L.size//2),(L.size//2) +1):
    positions.append(x)

fig = plt.figure(figsize=(16,9),dpi=200) 
plt.title(r'$\theta = \pi/4,|\uparrow>, t = 50$')
k,= plt.plot(positions,position_statistics[50][0])
plt.grid(linestyle='--')
plt.xlabel(r'x',fontsize=14)
plt.ylabel(r'Pr(x)',fontsize=14)
plt.savefig('1D_Eqwalks/e_x_position_distribuition01',bbox_inches='tight')
plt.clf()

plt.title(r'$ Entangling coin \theta_{1} = \pi/4,\theta_{2} = \pi/4,|\uparrow,\uparrow>, t = 50$')
k,= plt.plot(positions,position_statistics[50][1])
plt.grid(linestyle='--')
plt.xlabel(r'y',fontsize=14)
plt.ylabel(r'Pr(y)',fontsize=14)
plt.savefig('2D_qwalks/y_position_distribuition02',bbox_inches='tight')
plt.clf()

def general_variance(x,a,b,c,d):
    return a*x**3+b*x**2+c*x+d

fit_params, pcov = optimize.curve_fit(general_variance,time_steps,variance_x)
a = round(fit_params[0],2)
b = round(fit_params[1],2)
c = round(fit_params[2],2)
d = round(fit_params[3],2)


plt.title(r'$\theta = \pi/4,|\uparrow>')
l, = plt.plot(time_steps,variance_x,label = 'Simulation')
fit_label = str(a)+r'$t^{3}$'+ '+' +str(b)+r'$t^{2}$'
fit_label = fit_label + '+' + str(c)+r'$t$' + '+' + str(d)
m, = plt.plot(time_steps,general_variance(time_steps,*fit_params),label = fit_label)
plt.grid(linestyle='--')
plt.xlabel(r't',fontsize=14)
plt.ylabel(r'$\sigma_{x}^{2}$(t)',fontsize=14)
plt.legend(handles=[l,m])
plt.savefig('1D_Eqwalks/e_x_variance01',bbox_inches='tight')
plt.clf()

fit_params, pcov = optimize.curve_fit(general_variance,time_steps,variance_y)
a = round(fit_params[0],2)
b = round(fit_params[1],2)
c = round(fit_params[2],2)
d = round(fit_params[3],2)


plt.title(r'$ Entangling coin \theta_{1} = \pi/4,\theta_{2} = \pi/4,|\uparrow,\uparrow>')
l, = plt.plot(time_steps,variance_y,label = 'Simulation')
fit_label = str(a)+r'$t^{3}$'+ '+' +str(b)+r'$t^{2}$'
fit_label = fit_label + '+' + str(c)+r'$t$' + '+' + str(d)
m, = plt.plot(time_steps,general_variance(time_steps,*fit_params),label = fit_label)
plt.grid(linestyle='--')
plt.xlabel(r't',fontsize=14)
plt.ylabel(r'$\sigma_{y}^{2}$(t)',fontsize=14)
plt.legend(handles=[l,m])
plt.savefig('2D_qwalks/y_variance02',bbox_inches='tight')
plt.clf()


plt.title(r'$ \theta = \pi/4,|\uparrow>')
l, = plt.plot(time_steps,walker_negativity)
plt.grid(linestyle='--')
plt.xlabel(r'$t$',fontsize=14)
plt.ylabel(r'$N(\rho)$',fontsize=14)
plt.savefig('1D_Eqwalks/e_negativity01',bbox_inches='tight')
plt.clf()

