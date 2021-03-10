import numpy as np
import matplotlib.pyplot as plt

title_str = r'$ \Theta = '+str(45)
title_str = title_str +', \Omega = '+str(45)
title_str = title_str +', \phi = '+str(0)+'$'

main_dir = 'data/1D_gEQWalks/'
q_dirs =  ['q_05', 'q_07', 'q_10','q_13']

fig = plt.figure(figsize=(16,9))
plt.title(title_str,fontsize=16)

plots = []

for i in q_dirs:

    q_label = 'q = '

    for j in i[2:]:
        if j == i[-1]:        
            q_label = q_label + '.'+ j
        else:
            q_label = q_label + j

    entanglement_file = open(main_dir+i+'/entanglement_entropy.txt','r')

    entang_entrop = []

    for x in entanglement_file.readlines():
        ent_data = []
        for y in x.split('\t'):
            if y != '\n': ent_data.append(float(y))
        entang_entrop.append(ent_data)

    time_steps = []

    for i in range(0,len(entang_entrop)):
        time_steps.append(i)

    entang_entrop = np.array(entang_entrop)

    
    n, = plt.plot(time_steps,entang_entrop[:],lw=2,label = q_label)
    plots.append(n)

plt.grid(linestyle='--')
plt.xlabel(r't',fontsize=16)
plt.ylabel(r'$S_{E}$(t)',fontsize=16)
plt.xlim((0,100))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(handles=plots,fontsize=14)
plt.savefig('entanglement_entropy_comparative',bbox_inches='tight')
plt.clf()
