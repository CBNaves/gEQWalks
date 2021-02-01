import numpy as np
import matplotlib.pyplot as plt
total = 10**3
random_uniform_dist = np.random.uniform(0,10,total)
past_t = []
for i in random_uniform_dist:
    past_t.append(int(np.ceil(i)))
unique,counts = np.unique(past_t,return_counts=True)

uniform_dist = np.asarray((unique,counts))

fig = plt.figure(figsize=(16,9),dpi=200) 
plt.title('Discrete Uniform distribuition',fontsize=16)
plt.clf()
plt.scatter(uniform_dist[0,:],(1/total)*uniform_dist[1,:],lw=2,label='Simulation')
plt.grid(linestyle='--')
plt.xlabel(r'$i$')
plt.ylabel(r'$Pr(i)$')
plt.show()

