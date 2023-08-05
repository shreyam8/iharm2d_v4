import numpy as np
import h5py, sys, glob, psutil, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib import grid_DCSspec
#from mpl_toolkits.axes_grid_DCS1 import make_axes_locatable
import multiprocessing as mp


#####################################################################################
#                                       DUMPS
#####################################################################################

outputdir = sys.argv[1]
dumpsdir_GR = sys.argv[2]
dumpsdir_DCS= sys.argv[3]
dumpsdir_EDGB= sys.argv[4]

if not os.path.exists(outputdir):
    os.makedirs(outputdir)

grid_GR_file = os.path.join(dumpsdir_GR, "grid")
grid_DCS_file = os.path.join(dumpsdir_DCS, "grid")
grid_EDGB_file = os.path.join(dumpsdir_EDGB, "grid")

# Load the grid files
grid_GR = np.loadtxt(grid_GR_file)
grid_DCS = np.loadtxt(grid_DCS_file)
grid_EDGB = np.loadtxt(grid_EDGB_file)

# Reading in grid values
header_file = os.path.join(dumpsdir_GR, "dump_00000000")
with open(header_file, 'r') as header:
    firstline = header.readline().split()
    metric = firstline[9]
    n1 = int(firstline[11])
    n2 = int(firstline[12])
    ndim = int(firstline[22])
    t = float(firstline[29])


#####################################################################################
#                                    PRIMITIVES
#####################################################################################
# GR
prims_g = np.loadtxt(os.path.join(dumpsdir_GR,'dump_0000{0:04d}'.format(0)),skiprows=1)
rho_g = prims_g[:,0].reshape((n1,n2))
uu_g = prims_g[:,1].reshape((n1,n2))
u1_g = prims_g[:,2].reshape((n1,n2))
u2_g = prims_g[:,3].reshape((n1,n2))
u3_g = prims_g[:,4].reshape((n1,n2))
B1_g = prims_g[:,5].reshape((n1,n2))
B2_g = prims_g[:,6].reshape((n1,n2))
B3_g = prims_g[:,7].reshape((n1,n2))
logrho_g = np.log10(rho_g)

# DCS 
prims_d = np.loadtxt(os.path.join(dumpsdir_DCS,'dump_0000{0:04d}'.format(0)),skiprows=1)
rho_d = prims_d[:,0].reshape((n1,n2))
uu_d = prims_d[:,1].reshape((n1,n2))
u1_d = prims_d[:,2].reshape((n1,n2))
u2_d = prims_d[:,3].reshape((n1,n2))
u3_d = prims_d[:,4].reshape((n1,n2))
B1_d = prims_d[:,5].reshape((n1,n2))
B2_d = prims_d[:,6].reshape((n1,n2))
B3_d = prims_d[:,7].reshape((n1,n2))
logrho_d = np.log10(rho_d)

# EDGB
prims_e = np.loadtxt(os.path.join(dumpsdir_EDGB,'dump_0000{0:04d}'.format(0)),skiprows=1)
rho_e = prims_e[:,0].reshape((n1,n2))
uu_e = prims_e[:,1].reshape((n1,n2))
u1_e = prims_e[:,2].reshape((n1,n2))
u2_e = prims_e[:,3].reshape((n1,n2))
u3_e = prims_e[:,4].reshape((n1,n2))
B1_e = prims_e[:,5].reshape((n1,n2))
B2_e = prims_e[:,6].reshape((n1,n2))
B3_e = prims_e[:,7].reshape((n1,n2))
logrho_e = np.log10(rho_e)

prims_g = np.loadtxt(os.path.join(dumpsdir_GR,'dump_0000{0:04d}'.format(0)),skiprows=1)
prims_d = np.loadtxt(os.path.join(dumpsdir_DCS,'dump_0000{0:04d}'.format(0)),skiprows=1)
prims_e = np.loadtxt(os.path.join(dumpsdir_EDGB,'dump_0000{0:04d}'.format(0)),skiprows=1)

NPRIMS = 8
print("\n Prims check \n")
for n in range(NPRIMS):
    print(n, np.amax(np.fabs(prims_g[:,n].reshape((n1,n2)) - prims_e[:,n].reshape((n1,n2)))))


#####################################################################################
#                                   GRID VALUES
#####################################################################################
# GRID GR DATA 
x_g = grid_GR[:,0].reshape((n1,n2))
z_g = grid_GR[:,1].reshape((n1,n2))
r_g = grid_GR[:,2].reshape((n1,n2))
th_g = grid_GR[:,3].reshape((n1,n2))
x1_g = grid_GR[:,4].reshape((n1,n2))
x2_g = grid_GR[:,5].reshape((n1,n2))
gdet_g = grid_GR[:,6].reshape((n1,n2))
gcon_g = grid_GR[:,8:24].reshape((n1, n2, ndim, ndim))
gcov_g = grid_GR[:,24:].reshape((n1, n2, ndim, ndim))

# GRID DCS (GR) DATA
x_d = grid_DCS[:,0].reshape((n1,n2))
z_d = grid_DCS[:,1].reshape((n1,n2))
r_d = grid_DCS[:,2].reshape((n1,n2))
th_d = grid_DCS[:,3].reshape((n1,n2))
x1_d = grid_DCS[:,4].reshape((n1,n2))
x2_d = grid_DCS[:,5].reshape((n1,n2))
gdet_d = grid_DCS[:,6].reshape((n1,n2))
gcon_d = grid_DCS[:,8:24].reshape((n1, n2, ndim, ndim))
gcov_d = grid_DCS[:,24:].reshape((n1, n2, ndim, ndim))

# GRID EDGB (GR) DATA
x_e = grid_EDGB[:,0].reshape((n1,n2))
z_e = grid_EDGB[:,1].reshape((n1,n2))
r_e = grid_EDGB[:,2].reshape((n1,n2))
th_e = grid_EDGB[:,3].reshape((n1,n2))
x1_e = grid_EDGB[:,4].reshape((n1,n2))
x2_e = grid_EDGB[:,5].reshape((n1,n2))
gdet_e = grid_EDGB[:,6].reshape((n1,n2))
gcon_e = grid_EDGB[:,8:24].reshape((n1, n2, ndim, ndim))
gcov_e = grid_EDGB[:,24:].reshape((n1, n2, ndim, ndim))


#####################################################################################
#                                   DIFFERENCES
#####################################################################################
 
X_diff = x_g - x_e
Z_diff = z_g - z_e
r_diff = r_g - r_e
th_diff = th_g - th_e
x1_diff = x1_g - x1_e
x2_diff = x2_g - x2_e
gdet_diff = gdet_g - gdet_e
gcon_diff = gcon_g - gcon_e
gcov_diff = gcov_g - gcov_e

#gcon_diff = np.array(gcon_diff)
#gcov_diff = np.array(gcov_diff)

print("\n Grid Check \n")
print ("\n X diff \n ", np.amax(np.fabs(X_diff)))
print ("\n Z diff \n ", np.amax(np.fabs(Z_diff)))
print ("\n R diff \n ", np.amax(np.fabs((r_diff))))
print ("\n Theta diff \n ", np.amax(np.fabs((th_diff))))
print ("\n X1 diff \n ", np.amax(np.fabs((x1_diff))))
print ("\n X2 diff \n ", np.amax(np.fabs((x2_diff))))

#print ("\n Gdet diff \n ", np.unravel_index(np.amax(np.fabs(gdet_diff)), gdet_diff.shape))
print ("\n Gcon diff \n ", np.amax(np.fabs(gcon_diff)))
print ("\n Gcov diff \n ", np.amax(np.fabs(gcov_diff)))



#####################################################################################
#                                    RADIAL PLOTS  
#####################################################################################
font_title = {'family': 'serif'}
font_labels = {'family': 'serif'}
plt.rc('font', family='serif')
#plt.rcParams.update({font.size: 16})	 ## font size
#plt.rcParams.update({lines.markersize: 10}) ## scatter point size. I think default is 6?

fig, subs = plt.subplots(1,2,figsize=(16,6))

subs[0].plot(r_g[:,0], np.abs(gcov_g[:,128,0,0]),label="GR", linewidth=4, alpha=0.5, linestyle='--')
subs[0].plot(r_e[:,0], np.abs(gcov_e[:,128,0,0]),label="EDGB [$\zeta$=0.05]")
subs[0].plot(r_d[:,0], np.abs(gcov_d[:,128,0,0]),label="DCS [$\zeta$=0.05]", color='purple')
subs[0].set_xlabel("R",fontsize=17, fontdict=font_labels)
subs[0].set_xlim(0,5)
subs[0].set_ylim(10e-3,1)
subs[0].set_ylabel("$|g_{tt}|$", fontsize=17, fontdict=font_labels)
subs[0].set_yscale('log')


subs[0].legend()
subs[0].set_title('$g_{tt}$ COMPONENT ANALYSIS',fontsize=13, fontdict=font_labels)

subs[1].plot(r_g[:,0], np.abs(gcov_g[:,128,1,1]),label="GR",linewidth=4, alpha=0.5, linestyle='--')
subs[1].plot(r_e[:,0], np.abs(gcov_e[:,128,1,1]),label="EDGB [$\zeta$=0.05]")
subs[1].plot(r_d[:,0], np.abs(gcov_d[:,128,1,1]),label="DCS [$\zeta$=0.05]", color='purple')
subs[1].set_xlabel("R",fontsize=17, fontdict=font_labels)
subs[1].set_xlim(0,5)
subs[1].set_ylim(pow(10,0),pow(10,2))
subs[1].set_ylabel("$g_{rr}$",fontsize=17, fontdict=font_labels)
subs[1].set_yscale('log')

subs[1].legend()
subs[1].set_title('$g_{rr}$ COMPONENT ANALYSIS', fontsize=13, fontdict=font_labels)

#fig.suptitle('METRIC COMPONENT ANALYSIS IN THE MIDPLANE', fontsize=18, fontdict=font_title)

plt.savefig('Metric comparisons for poster', dpi=300)

#####################################################################################
#                    HEATMAPS OF DIFFERENCES IN GCOV COMPONENTS
#####################################################################################

# fig, axs = plt.subplots(2, 2, figsize=(13, 10))

# tt = axs[0, 0].pcolormesh(x_g, z_g, gcov_diff[:,:,0,0])
# axs[0, 0].set_title('DCS[zeta=0.1] - GR : TT diff in MKS coords')
# axs[0, 0].set_xlabel('X')
# axs[0, 0].set_ylabel('Z')
# fig.colorbar(tt, ax=axs[0, 0])

# rr = axs[0, 1].pcolormesh(x_g, z_g, gcov_diff[:,:,1,1])
# axs[0, 1].set_title('DCS[zeta=0.1] - GR : RR diff in MKS coords')
# axs[0, 1].set_xlabel('X')
# axs[0, 1].set_ylabel('Z')
# fig.colorbar(tt, ax=axs[0, 1])

# tp = axs[1, 0].pcolormesh(x_g, z_g, gcov_diff[:,:,0,3])
# axs[1, 0].set_title('DCS[zeta=0.1] - GR : Tϕ diff in MKS coords')
# axs[1, 0].set_xlabel('X')
# axs[1, 0].set_ylabel('Z')
# fig.colorbar(tt, ax=axs[1, 0])

# pp = axs[1, 1].pcolormesh(x_g, z_g, gcov_diff[:,:,3,3])
# axs[1, 1].set_title('DCS[zeta=0.1] - GR : ϕϕ diff in MKS coords')
# axs[1, 1].set_xlabel('X')
# axs[1, 1].set_ylabel('Z')
# fig.colorbar(tt, ax=axs[1, 1])

# plt.savefig("actual diff")


#####################################################################################
#                           CONTRAVARIANT METRIC PLOTTING
#####################################################################################

#                         GR GCOV METRIC COMPONENT HEATMAPS 

# fig1, subG1 = plt.subplots(nrows=2, ncols=2,figsize=(16,9))
# plt.subplots_adjust(hspace=0.3)
# fig1.suptitle("COV Metric components, GR (spin=0.5) in MKS coordinates", fontsize=16)

# pcg1 = subG1[0, 0].pcolormesh(x_g, z_g, gcov_g[:,:,0,0],cmap='jet',shading='gouraud')
# subG1[0, 0].set_title('TT component')
# subG1[0, 0].grid(True)
# subG1[0, 0].set_aspect('equal')
# plt.colorbar(pcg1, ax=subG1[0, 0])

# pcg2 = subG1[0, 1].pcolormesh(x_g, z_g, gcov_g[:,:,0,3],cmap='jet',shading='gouraud')
# subG1[0, 1].set_title('$Tϕ$ component')
# subG1[0, 1].grid(True)
# subG1[0, 1].set_aspect('equal')
# plt.colorbar(pcg2, ax=subG1[0, 1])


# pcg3 = subG1[1, 0].pcolormesh(x_g, z_g, gcov_g[:,:,1,1],cmap='jet',shading='gouraud')
# subG1[1, 0].set_title('RR component')
# subG1[1, 0].grid(True)
# subG1[1, 0].set_aspect('equal')
# plt.colorbar(pcg3, ax=subG1[1, 0])

# pcg4 = subG1[1, 1].pcolormesh(x_g, z_g, gcov_g[:,:,3,3],cmap='jet',shading='gouraud')
# subG1[1, 1].set_title('$ϕϕ$ component')
# subG1[1, 1].grid(True)
# subG1[1, 1].set_aspect('equal')
# plt.colorbar(pcg4, ax=subG1[1, 1])


# fig1.savefig('COV GR Metric comparisons')

#####################################################################################
#                         DCS GCOV METRIC COMPONENT HEATMAPS 

# fig3, subD1 = plt.subplots(nrows=2, ncols=2,figsize=(16,9))
# plt.subplots_adjust(hspace=0.3)
# fig3.suptitle("COV Metric components, DCS [zeta=0.1] (spin=0.5) in MKS coordinates", fontsize=16)

# pc1 = subD1[0, 0].pcolormesh(x_d, z_d, gcov_d[:,:,0,0],cmap='jet',shading='gouraud')
# subD1[0, 0].set_title('TT component')
# subD1[0, 0].grid(True)
# subD1[0, 0].set_aspect('equal')
# plt.colorbar(pcg1, ax=subD1[0, 0])

# pc2 = subD1[0, 1].pcolormesh(x_d, z_d, gcov_d[:,:,0,3],cmap='jet',shading='gouraud')
# subD1[0, 1].set_title('$Tϕ$ component')
# subD1[0, 1].grid(True)
# subD1[0, 1].set_aspect('equal')
# plt.colorbar(pcg2, ax=subD1[0, 1])


# pc3 = subD1[1, 0].pcolormesh(x_d, z_d, gcov_d[:,:,1,1],cmap='jet',shading='gouraud')
# subD1[1, 0].set_title('RR component')
# #subD1[1, 0].axvline(x=35,color='red',linestyle='--')
# subD1[1, 0].grid(True)
# subD1[1, 0].set_aspect('equal')
# plt.colorbar(pcg3, ax=subD1[1, 0])

# pc4 = subD1[1, 1].pcolormesh(x_d, z_d, gcov_d[:,:,3,3],cmap='jet',shading='gouraud')
# subD1[1, 1].set_title('$ϕϕ$ component')
# subD1[1, 1].grid(True)
# subD1[1, 1].set_aspect('equal')
# plt.colorbar(pcg4, ax=subD1[1, 1])

# fig3.savefig('COV DCS Metric comparisons')

########################################################################################
#                         EDGB GCOV METRIC COMPONENT HEATMAPS 

# fig5, subE1 = plt.subplots(nrows=2, ncols=2,figsize=(16,9))
# plt.subplots_adjust(hspace=0.3)
# fig1.suptitle("COV Metric components, EDGB [zeta=0.0](spin=0.5) in MKS coordinates", fontsize=16)

# pcg1 = subE1[0, 0].pcolormesh(x_e, z_e, gcov_e[:,:,0,0],cmap='jet',shading='gouraud')
# subE1[0, 0].set_title('TT component')
# subE1[0, 0].grid(True)
# subE1[0, 0].set_aspect('equal')
# plt.colorbar(pcg1, ax=subE1[0, 0])

# pcg2 = subE1[0, 1].pcolormesh(x_e, z_e, gcov_e[:,:,0,3],cmap='jet',shading='gouraud')
# subE1[0, 1].set_title('$Tϕ$ component')
# subE1[0, 1].grid(True)
# subE1[0, 1].set_aspect('equal')
# plt.colorbar(pcg2, ax=subE1[0, 1])


# pcg3 = subE1[1, 0].pcolormesh(x_e, z_e, gcov_e[:,:,1,1],cmap='jet',shading='gouraud')
# subE1[1, 0].set_title('RR component')
# subE1[1, 0].grid(True)
# subE1[1, 0].set_aspect('equal')
# plt.colorbar(pcg3, ax=subE1[1, 0])

# pcg4 = subE1[1, 1].pcolormesh(x_e, z_e, gcov_e[:,:,3,3],cmap='jet',shading='gouraud')
# subE1[1, 1].set_title('$ϕϕ$ component')
# subE1[1, 1].grid(True)
# subE1[1, 1].set_aspect('equal')
# plt.colorbar(pcg4, ax=subE1[1, 1])


# fig1.savefig('COV EDGB Metric comparisons')

########################################################################################
#                           CONTRAVARIANT METRIC PLOTTING
########################################################################################
#                         DCS GCON METRIC COMPONENT HEATMAPS 

# fig4, subD2 = plt.subplots(nrows=2, ncols=2,figsize=(16,9))
# plt.subplots_adjust(hspace=0.3)
# fig4.suptitle("CON Metric components, DCS in MKS coordinates", fontsize=16)

# subD2[0, 0].pcolormesh(x_d, z_d, gcon_d[:,:,0,0],cmap='jet',shading='gouraud')
# subD2[0, 0].set_title('TT component')

# subD2[0, 1].pcolormesh(x_d, z_d, gcon_d[:,:,0,3],cmap='jet',shading='gouraud')
# subD2[0, 1].set_title('$Tϕ$ component')

# subD2[1, 0].pcolormesh(x_d, z_d, gcon_d[:,:,1,1],cmap='jet',shading='gouraud')
# subD2[1, 0].set_title('RR component')

# im4 = subD2[1, 1].pcolormesh(x_d, z_d, gcon_d[:,:,3,3],cmap='jet',shading='gouraud')
# subD2[1, 1].set_title('$ϕϕ$ component')

# cbar = fig4.colorbar(im4, ax=subD2, shrink=0.95, aspect=40, pad=0.05)

# fig4.savefig('CON DCS Metric comparisons')

########################################################################################
#                         GR GCON METRIC COMPONENT HEATMAPS 

# fig2, subGR2 = plt.subplots(nrows=2, ncols=2,figsize=(16,9))
# plt.subplots_adjust(hspace=0.3)
# fig2.suptitle("CON Metric components, GR in MKS coordinates", fontsize=16)

# subGR2[0, 0].pcolormesh(x_g, z_g, gcon_g[:,:,0,0],cmap='jet',shading='gouraud')
# subGR2[0, 0].set_title('TT component')

# subGR2[0, 1].pcolormesh(x_g, z_g, gcon_g[:,:,0,3],cmap='jet',shading='gouraud')
# subGR2[0, 1].set_title('$Tϕ$ component')

# subGR2[1, 0].pcolormesh(x_g, z_g, gcon_g[:,:,1,1],cmap='jet',shading='gouraud')
# subGR2[1, 0].set_title('RR component')

# im2 = subGR2[1, 1].pcolormesh(x_g, z_g, gcon_g[:,:,3,3],cmap='jet',shading='gouraud')
# subGR2[1, 1].set_title('$ϕϕ$ component')

# cbar = fig2.colorbar(im2, ax=subGR2, shrink=0.95, aspect=40, pad=0.05)
# fig2.savefig('CON GR Metric comparisons')

########################################################################################




