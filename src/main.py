import numpy as np
import matplotlib.pyplot as plt
import qwalk
import statistics

L = qwalk.Lattice(1,201)
f = qwalk.FermionSpin()
S = qwalk.FermionShiftOperator(L,f)
c = qwalk.FermionCoin([np.pi/4])
W = qwalk.Walker(f.up,L)
position_statistics = []
mean_position = []
mean_sq_position = []
sq_variance = []
walker_negativity = []
time_steps = []
for t in range(0,101):
    time_steps.append(t)
    ps,mp,msq,sv = statistics.position_statistics(W.density,L,2)
    position_statistics.append(ps)
    mean_position.append(mp)
    mean_sq_position.append(msq)
    sq_variance.append(sv)
    walker_negativity.append(statistics.negativity(W.density,L,True,[1,2]))
    W.walk(c,S,L)

positions = []
for x in range(-(L.size//2),(L.size//2) +1):
    positions.append(x)

fig = plt.figure(figsize=(12,12)) 
plt.title('Distribuição de probabilidade')
k,= plt.plot(positions,position_statistics[100][0])
plt.grid(linestyle='--')
plt.xlabel(r'x')
plt.ylabel(r'Pr(x)')
plt.savefig('x_position_distribuition')
plt.clf()

sq_variance = np.array(sq_variance)
plt.title(r'\sigma^{2}(t) \times t')
l, = plt.plot(time_steps,sq_variance[:,0])
plt.grid(linestyle='--')
plt.xlabel(r't')
plt.ylabel(r'\sigma^(2)(t)')
plt.savefig('variance')
plt.clf()

plt.title(r'\sigma^{2}(t) \times t')
l, = plt.plot(time_steps,walker_negativity)
plt.grid(linestyle='--')
plt.xlabel(r't')
plt.ylabel(r'N(\rho)')
plt.savefig('negativity')
plt.clf()
