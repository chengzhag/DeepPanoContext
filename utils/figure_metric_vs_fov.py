import matplotlib
 
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np

fovs = np.array([360, 180, 120, 90, 60, 30])

mAP_wo_ro = np.array([30.91, 27.4, 24.77, 25.9, 26.16, 25.41])

mAP_full = np.array([33.92, 26.09, 24.34, 23.01, 21.72, 18.52])

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.figure()

plt.plot(fovs, mAP_wo_ro, lw=1.5, c='red', marker='s', ms=6, clip_on = False, zorder=100, label='w/o. RO')  
plt.plot(fovs, mAP_full, lw=1.5, c='g', marker='s', ms=6, clip_on = False, zorder=100, label='Full')  
# plt-style 

plt.xticks(fovs)  
plt.xlim(0, 360)  
plt.ylim(15, 35)  

plt.xlabel('Field of View (degree)')
plt.ylabel('mAP')  
plt.grid(ls=':')
plt.gca().invert_xaxis()
# plt.legend(loc='upper right', borderpad=2)  
plt.legend()
plt.savefig('./mAP_vs_fov.eps', dpi=600, format='eps')  

# legend = {'w/o. RO', 'Full'}
# xtitle = 'Field of View (degree)'
# ytitle = 'mAP'
