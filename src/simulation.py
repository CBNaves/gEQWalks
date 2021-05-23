import os
import time
import subprocess

q = [0.5,1000]
for i in range(0,2):
    cfg_file = open('gEQW.cfg','w+')
    cfg_file.write('dimension = 2\n')
    cfg_file.write('size = 501\n')
    cfg_file.write('coin_type = fermion fermion\n')
    cfg_file.write('thetas = 45 45\n')
    cfg_file.write('in_pos_var = 0 0\n')
    cfg_file.write('coin_init_state = 45 0 45 0\n')
    cfg_file.write('q = '+str(q[i])+' '+str(q[i])+'\n')
    cfg_file.write('mem_dependence = 1 0 0 1\n')
    cfg_file.write('trace_dist = False\n')
    cfg_file.write('entang = False')
    cfg_file.close()

    procs = []
    for i in range(0,3):
        proc = subprocess.Popen(['python3 main.py'], shell=True)
        procs.append(proc)
        time.sleep(3)
    pid = procs[2].pid
    os.waitid(os.P_PID,pid,os.WEXITED)    
print('Done!\n')
