import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from scipy import optimize
from scipy import constants
import matplotlib.animation as animation

def general_variance(x,a,b):
        return a*x + b

def pol_var(x,a,b,c):
    return a*x**(2) + b*x + c

title_str = r'$ \Theta(s) = '+str([45])
title_str = title_str +', \Omega(s) = '+str([45])
title_str = title_str +', \phi(s) = '+str([0])+'$'
#title_str = r'$\sigma(t)^{2} \propto t^{\alpha}$'

#main_dir = 'data/1D_gEQWalks/q_0.5_gauss_theta_45_omega_0'
#q_type = 'q_inf_inf'
#q_dirs =  ['q_0.5','q_0.6','q_0.7','q_0.8','q_0.9','q_1.0','q_1.1','q_1.2','q_1.3','q_1.4','q_1.5','q_1.6','q_1.7','q_1.8','q_1.9','q_inf']
#q_dirs = ['q_inf']
#q_dirs = ['','_md_0.3','_md_0.35','_md_0.5','_md_0.8']

fig = plt.figure(figsize=(16,5),dpi=200)
#plt.title(title_str,fontsize=16)

################################################################################
#def format_func(value, tick_number):
#    # find number of multiples of pi/2
#    N = int(np.round(2 * value / np.pi))
#    if N == 0:
#        return "0"
#    elif N == 1:
#        return r"$\pi/2$"
#    elif N == 2:
#        return r"$\pi$"
#    elif N % 2 > 0:
#        return r"${0}\pi/2$".format(N)
#    else:
#        return r"${0}\pi$".format(N // 2)
#
#thetas = [0,15,30,45,60,75]
#omegas = [0,10,20,30,40,50,80]
#params = [thetas,omegas]
#
#gs = fig.add_gridspec(1, 2, hspace=0, wspace=0.25)
#axs = gs.subplots()
#fig.suptitle(title_str,fontsize=16)
#
#for i in range(0,2):
#    mean_entanglement = []
#    mean_ent_err = []
#    for j in params[i]:
#        if i == 0: 
#            pm = '_theta_'
#            xlbl = r'$\theta\degree$'
#            ylbl = r'$<S_{E}>(\theta)$'
#            title_str = r'$\Omega(s) = '+str([45])
#            title_str = title_str +', \phi(s) = '+str([0])+'$'
#        else: 
#            pm = '_omega_'
#            xlbl = r'$\Omega\degree$'
#            ylbl = r'$<S_{E}>(\Omega)$'
#            title_str = r'$\Theta(s) = '+str([45])
#            title_str = title_str +', \phi(s) = '+str([0])+'$'
#    q_label = 'q = '
#    q_label = 'mem. dependence = '
#
#    if q_dirs[i][2:] == 'inf': q_label = q_label + r'$\infty$'
#    else:
#        for j in q_dirs[i][2:]:
#            q_label = q_label + j
#
#        if j != 45: 
#            sd = pm+str(j)
#            entanglement_file = open(main_dir+sd+'/entanglement_entropy.txt','r')
#        else:
#            entanglement_file = open(main_dir+'/entanglement_entropy.txt','r')
#
#        entang_entrop = []
#
#        for x in entanglement_file.readlines():
#            ent_data = []
#            for y in x.split('\t'):
#                if y != '\n': ent_data.append(float(y))
#            entang_entrop.append(ent_data)
#
#        entang_entrop = np.array(entang_entrop)
#        mean_entanglement.append(np.mean(entang_entrop[100:]))
#        mean_ent_err.append(np.var(entang_entrop[100:]))
#
#        time_steps = []
#
#        for j in range(0,len(entang_entrop[:,0])):
#            time_steps.append(j)
#
#    if q_dirs[i][2:] != 'inf':
#        qs.append(float(q_dirs[i][2:]))
#        if float(q_dirs[i][2:]) <= 0.9:
#            mean_entanglement.append(np.mean(entang_entrop[2000:]))
#            mean_ent_err.append(np.var(entang_entrop[2000:]))
#        elif float(q_dirs[i][2:]) > 0.9 and float(q_dirs[i][2:]) <= 1.2:
#            mean_entanglement.append(np.mean(entang_entrop[1000:]))
#            mean_ent_err.append(np.var(entang_entrop[1000:]))
#        elif float(q_dirs[i][2:]) > 1.2 and float(q_dirs[i][2:]) < 1.6:
#            mean_entanglement.append(np.mean(entang_entrop[200:]))
#            mean_ent_err.append(np.var(entang_entrop[200:]))
#        else:
#            mean_entanglement.append(np.mean(entang_entrop[100:]))
#            mean_ent_err.append(np.var(entang_entrop[100:]))
#    else:
#        mean_entanglement.append(np.mean(entang_entrop[100:]))
#        mean_ent_err.append(np.var(entang_entrop[100:]))
#
#    mean_ent_err = np.array(mean_ent_err)
#    mean_ent_err = np.sqrt(mean_ent_err)
#    mean_entanglement = np.array(mean_entanglement)
#
#    n, = plt.plot(time_steps,entang_entrop[:,0],lw=2,label = q_label,marker='o',color = colors[i])
#    plots.append(n)cd
#    axs[i].errorbar(params[i],mean_entanglement[:],yerr=mean_ent_err[:],ecolor='blue',fmt = '.k',ms=15,elinewidth=3,capsize=4)
#plt.hlines(mean_entanglement[-1],0,2,linestyles='-',color='red',label=r'q = $\infty$',lw=2)
#plt.hlines(mean_entanglement[0],0,2,linestyles='--',color='blue',label=r'q = $0.5$',lw=2)    
#    axs[i].set_title(title_str,fontsize=16)
#    axs[i].grid(True,linestyle='--')
#plt.xlabel(r't',fontsize=16)
#plt.ylabel(r'$S_{E}$(t)',fontsize=16)
#plt.xlim((0.4,2))
#    plt.ylim((0.85,1.05))
#    axs[i].set(xlabel=xlbl,ylabel=ylbl)
#    axs[i].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
#    axs[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
#    axs[i].ticklabel_format(axis='y',style='sci',scilimits = (-10,10))
#    axs[i].xaxis.get_label().set_fontsize(16)
#    axs[i].yaxis.get_label().set_fontsize(16)
#    axs[i].tick_params(axis='x',labelsize=14)
#    axs[i].tick_params(axis='y',labelsize=14)
#    axs[i].set_ylim((0.99,1.01))
#    plt.legend(fontsize=16)
#plt.savefig(main_dir+'md_entanglement_entropy_comparative',bbox_inches='tight')
#plt.savefig('mean_entanglement_entropy_x_theta&omega_deg',bbox_inches='tight')
#plt.clf()

###############################################################################
#est_list = [0,90]
#mds = [0,0.3,0.35,0.5,0.8]
#est_list = [np.zeros(16,dtype=int)]
#est_list.append([7,1095,403,148,1096,403,244,403,403,148,148,90,54,54,54,90])
#gs = fig.add_gridspec(1, 2, hspace=0, wspace=0.15)
#axs = gs.subplots()
#title_str = r'$\alpha$ $x$ $q$, $Var(x) \propto t^{\alpha}$'
#fig.suptitle(title_str,fontsize=16)
#
#for j in range(0,2):
#
#    statistics = []
#    dimension = 1
#    if j==1: main_dir = 'data/1D_gEQWalks/q_inf_gauss_theta_45_omega_0'
#    qs.append(float(q_dirs[i][2:]))
#    statistics_file = open(main_dir+'/statistics.txt','r')
#    for x in statistics_file.readlines():
#        t_stat = []
#        for y in x.split('\t'):
#            if y != '\n': t_stat.append(float(y))
#        statistics.append(t_stat)
#    statistics = np.array(statistics)
#
#    time_steps = statistics[:,0]
#    time_steps = np.array(time_steps)
#        log_times = np.log(time_steps[1:])
#    variance = statistics[:,dimension+1]
#    log_variance = np.log(variance[1:])
#
#    fit_params, pcov =  optimize.curve_fit(pol_var,time_steps[:],variance[:])
#
#    if j == 0:
#        ar_int = 9
#        br_int = 5
#        cr_int = 2
#    else:
#        ar_int = 0
#        br_int = -2
#        cr_int = -4
#
#    a = round(fit_params[0],ar_int)
#    b = round(fit_params[1],br_int)
#    c = round(fit_params[2],cr_int)
#    perr = np.sqrt(np.diag(pcov))
#    un_a = round(perr[0],ar_int)
#    un_b = round(perr[1],br_int)
#    un_c = round(perr[2],cr_int)
#
#    if j == 0:subtitle = r'$q = 0.5$'
#    else: subtitle = r'$q = \infty$'
#    axs[j].plot(time_steps,variance,label = 'Simulation',lw=4,color='blue')
#    fitting_label = r'$('+str(a)+'\pm'+str(un_a)+')t^2 + ('+str(b)+'\pm'+str(un_b)+')t + ('+str(c)+'\pm'+str(un_c)+')$'
#    axs[j].plot(time_steps,pol_var(time_steps,*fit_params),lw=3,ls='--',color='orange',label=fitting_label)
#    axs[j].set_title(subtitle,fontsize=14)
#    axs[j].fill_between(time_steps,pol_var(time_steps,a+un_a,b+un_b,c+un_c),pol_var(time_steps,a-un_a,b-un_b,c-un_c),color='red',lw=3,label='Uncertainty Region',alpha=0.5)    
#    axs[j].grid(True,linestyle='--')
#    plt.ylabel(r'$\sigma^{2}(t)$',fontsize = 16)
#    plt.xlabel(r'$t$',fontsize = 16)
#    axs[j].legend(fontsize=14)
#    axs[j].set(xlabel=r'$t$',ylabel=r'$\sigma^{2}(t)$')
#    axs[j].tick_params(axis='x',labelsize=14)
#    axs[j].tick_params(axis='y',labelsize=14)
#    axs[j].xaxis.get_label().set_fontsize(16)
#    axs[j].yaxis.get_label().set_fontsize(16)
#    axs[j].legend(fontsize=12)

#for ax in axs.flat:
#    ax.set(xlabel=r'$q$', ylabel=r'$\alpha$')
#    ax.xaxis.get_label().set_fontsize(16)
#    ax.yaxis.get_label().set_fontsize(16)
#
# Hide x labels and tick labels for top plots and y ticks for right plots.
#for ax in axs.flat:
#    ax.label_outer()
#
#plt.savefig('var_theta_45_omega_0_gauss',bbox_inches='tight')
#plt.clf()
#################################################################################
#size = 50001
#tmax = 322
#pairs = [[0,0],[0,1],[1,0],[1,1]]
#pairs = np.array(pairs)
#gs = fig.add_gridspec(1, 1, hspace=0, wspace=0.2)
#fig.suptitle(title_str,fontsize=16)
#ax = gs.subplots()
#save_str = 'qs_pos_pd'

#positions = []
#for x in range(-(size//2),(size//2)+1):
#    positions.append(x)
#
#prob_dist_file = open(main_dir+'/pd_70','r')
#probabilities = []
#for x in prob_dist_file.readlines():
#    d_prob = []
#    for y in x.split('\t'):
#        if y != '\n': d_prob.append(float(y))
#    probabilities.append(d_prob)
#
#probabilities = np.array(probabilities)
#im = ax.imshow((np.ones(30),np.ones(30)),cmap='gray',interpolation='nearest')
#def update_img(n):
#    positions = []
#    for x in range(-(size//2),(size//2)+1):
#        positions.append(x)
#    
#    prob_dist_file = open(main_dir+'/pd_'+str(n),'r')
#    probabilities = []
#    for x in prob_dist_file.readlines():
#        d_prob = []
#        for y in x.split('\t'):
#            if y != '\n': d_prob.append(float(y))
#        probabilities.append(d_prob)
#
#    probabilities = np.array(probabilities)
#    im.set_data((positions,probabilities[0,:]))
#    return im
#
#ani = animation.FuncAnimation(fig,update_img,300,interval=tmax)
#writer = animation.writers['imagemagick'](fps=30)
#
#ani.save('demo.mp4',writer=writer)

#import matplotlib.animation as animation
#import numpy as np
#from pylab import *
#
#
#dpi = 100
#
#def ani_frame():
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.set_aspect('equal')
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#
#    positions = []
#    size = 50000
#    for x in range(-(size//2),(size//2)+1):
#        positions.append(x)
#    
#    prob_dist_file = open(main_dir+'/pd_0','r')
#    probabilities = []
#    for x in prob_dist_file.readlines():
#        d_prob = []
#        for y in x.split('\t'):
#            if y != '\n': d_prob.append(float(y))
#        probabilities.append(d_prob)
#
#    probabilities = np.array(probabilities)
#    im = ax.imshow((positions,probabilities[0,:]))
#    im.set_clim([0,1])
#    fig.set_size_inches([5,5])
#
#
#    tight_layout()
#
#
#    def update_img(n):
#        positions = []
#        size = 50000
#        for x in range(-(size//2),(size//2)+1):
#            positions.append(x)
#    
#        prob_dist_file = open(main_dir+'/pd_'+str(n),'r')
#        probabilities = []
#        for x in prob_dist_file.readlines():
#            d_prob = []
#            for y in x.split('\t'):
#                if y != '\n': d_prob.append(float(y))
#            probabilities.append(d_prob)
#
#        probabilities = np.array(probabilities)
#        im.set_data((positions,probabilities[0,:]))
#        return im
#
#    #legend(loc=0)
#    ani = animation.FuncAnimation(fig,update_img,300,interval=300)
#    writer = animation.writers['ffmpeg'](fps=30)
#
#    ani.save('demo.mp4',writer=writer,dpi=dpi)
#    return ani
#
#ani = ani_frame()



#for i in range(0,1):
#
#    pair = pairs[i] 
#
#    main_qdir = main_dir + q_dirs[i]
#    title_str = r'$q = '+q_dirs[i][2:]+'$'
#    label_dimension = 'x_'+str(i+1)
#
#    plt.title(title_str,fontsize=16)
#    plt.plot(positions,probabilities[0,:],lw=2,color='blue')
#    plt.grid(linestyle='--')
#    plt.xlabel(r'$'+label_dimension+'$',fontsize=16)
#    plt.yticks(fontsize=16)
#    plt.xticks(fontsize=16)
#    plt.xlim((-500,500))
#    plt.ylabel(r'$Pr('+label_dimension+')$',fontsize=16)
#    plt.legend(fontsize=16)

#    axs[i].set_title(title_str,fontsize=16)
#
#    title_str = title_str + 't = '+str(tmax)+'$'
#    axs[i].plot(positions,probabilities[i,:],lw=2,color='blue')
#    axs[i].grid(True)
#    axs[i].tick_params(axis='x',labelsize=14)
#    axs[i].tick_params(axis='y',labelsize=14)
#    axs[i].set_ylim((0,0.033))
#    axs[i].set(xlabel=r'$'+label_dimension+'$', ylabel=r'$Pr('+label_dimension+')$')
#    axs[i].xaxis.get_label().set_fontsize(16)
#    axs[i].yaxis.get_label().set_fontsize(16)

#for ax in axs.flat:
#    ax.set(xlabel=r'$'+label_dimension+'$', ylabel=r'$Pr('+label_dimension+')$')
#    ax.xaxis.get_label().set_fontsize(16)
#    ax.yaxis.get_label().set_fontsize(16)
#
# Hide x labels and tick labels for top plots and y ticks for right plots.
#for ax in axs.flat:
#    ax.label_outer()

#plt.savefig('q_inf_gaussian_70',bbox_inches='tight')
################################################################################

#time_steps = []
#trace_dist_vector = []
#trace_dist_file = open(main_dir+'/trace_distance.txt','r')
#     
#for x in trace_dist_file.readlines():
#    for y in x.split('\n'):
#        if y != '': trace_dist_vector.append(float(y))
#
# Loop to calculate the trace distance derivative.
# Here we use the middle point rule for mid points, and difference
# for the terminals.
#for i in range(0,len(trace_dist_vector)):
#    time_steps.append(i)
#time_steps = np.array(time_steps)
#
#trace_dist_derivate = []
#for i in time_steps[:]:
#    j = int(i)
#    if i == 0:
#        derivate = (trace_dist_vector[j+1] - trace_dist_vector[j])
#    elif i == time_steps[-1]:
#        derivate = (trace_dist_vector[j] - trace_dist_vector[j-1])                
#    else:
#        derivate = (trace_dist_vector[j+1] - trace_dist_vector[j-1])/2
#
#    trace_dist_derivate.append(derivate)
#
#trace_dist_file.close()
#left, bottom, width, height = [0.25,0.25,0.25,0.25]
#left, bottom, width, height = [0.55, 0.55, 0.3, 0.3]
#gs = fig.add_gridspec(1, 1, hspace=0, wspace=0)
#axs1 = gs.subplots()
#axs2 = fig.add_axes([left,bottom,width,height])
#
#axs1.plot(time_steps[:],trace_dist_vector[:],color='black',lw=3,marker='o')
#axs1.set_title(title_str,fontsize=16)
#axs1.set_xlim((-1,200))
#axs1.tick_params(axis='x',labelsize=16)
#axs1.tick_params(axis='y',labelsize=16)
#axs1.grid(True)
#axs1.set(ylabel=r'$D(\rho,\rho^{\perp})(t)$',xlabel=r'$t$')
#axs1.xaxis.get_label().set_fontsize(16)
#axs1.yaxis.get_label().set_fontsize(16)
#
#axs2.plot(time_steps[:],trace_dist_derivate[:],color='blue',lw=2)
#axs2.set_xlim((-1,200))
#axs2.grid(True)
#axs2.set(ylabel=r'$d/dt D(\rho,\rho^{\perp})(t)$',xlabel=r'$t$')
#axs2.tick_params(axis='x',labelsize=14)
#axs2.tick_params(axis='y',labelsize=14)
#axs2.xaxis.get_label().set_fontsize(14)
#axs2.yaxis.get_label().set_fontsize(14)
#
#plt.savefig('q_inf_trace_dist',bbox_inches='tight')

###############################################################################

thetas_array = np.arange(0,100,10)
blochs_array = np.arange(0,100,10)
initial_angles_set = np.meshgrid(thetas_array,blochs_array)
xs,ys = initial_angles_set
initial_angles_set = np.array(initial_angles_set).T.reshape(-1,2)
mean_entanglement = []
for angles in initial_angles_set:
    main_dir = 'data/1D_gEQWalks/q_inf_gauss_theta_'+str(angles[0])
    main_dir = main_dir + '_omega_'+str(angles[1])

    entanglement_file = open(main_dir+'/entanglement_entropy.txt','r')

    entang_entrop = []

    for x in entanglement_file.readlines():
        ent_data = []
        for y in x.split('\t'):
            if y != '\n': ent_data.append(float(y))
        entang_entrop.append(ent_data)

    entang_entrop = np.array(entang_entrop)
    mean = np.mean(entang_entrop[250:])
    mean_entanglement.append(mean)
    
mean_entanglement = np.array(mean_entanglement).reshape(10,10)
ax = plt.axes(projection = '3d')
surface = ax.plot_surface(xs,ys,mean_entanglement,cmap="jet",vmin=0,vmax=1)
cb = plt.colorbar(surface,pad=0.05)
ax.set_zlim(0,1)
ax.set_xlabel(r'$\theta$',fontsize=12)
ax.set_ylabel(r'$\Omega$',fontsize=12)
ax.set_zlabel(r'$< S_E >$',fontsize=12)

for angle in range(0,360,15):
    ax.view_init(20, angle)
    plt.savefig('mean_entanglement_'+str(angle),bbox_inches='tight')
