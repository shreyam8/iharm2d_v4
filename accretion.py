import numpy as np
import sys, glob, psutil, os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing as mp

# global dictionaries
params = {}

# plotting parameters
mpl.rcParams['figure.dpi']        = 200
mpl.rcParams['savefig.dpi']       = 200
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize']    = (14,5)
mpl.rcParams['axes.titlesize']    = 18
mpl.rcParams['axes.labelsize']    = 16
mpl.rcParams['xtick.labelsize']   = 14
mpl.rcParams['ytick.labelsize']   = 14
mpl.rcParams['axes.grid']         = True
mpl.rcParams['axes.xmargin']      = 0.02
mpl.rcParams['axes.ymargin']      = 0.1
mpl.rcParams['legend.fontsize']   = 'large'
# mpl.rcParams['text.usetex']       = True
# mpl.rcParams['font.family']       = 'serif'


params['dumpsdir_gr']   = sys.argv[1]
params['dumpsdir_dcs']  = sys.argv[2]
params['dumpsdir_edgb'] = sys.argv[3]
params['outputdir']     = sys.argv[4]
if not os.path.exists(params['outputdir']):
	os.makedirs(params['outputdir'])

# # function to parallelize plotting
# def run_parallel(function, dlist,	nthreads):
# 	pool = mp.Pool(nthreads)
# 	pool.map_async(function, dlist).get(720000)
# 	pool.close()
# 	pool.join()


# # function to generate 4-potential from magnetic field
# def plotting_bfield_lines(ax, B1, B2, xp, zp, n1, n2, dx1, dx2, gdet, nlines=20):
# 	AJ_phi = np.zeros([n1,n2]) 
# 	for j in range(n2):
# 			for i in range(n1):
# 					AJ_phi[i,j] = (np.trapz(gdet[:i,j]*B2[:i,j],dx=dx1) - np.trapz(gdet[i,:j]*B1[i,:j],dx=dx2))
# 	AJ_phi -=AJ_phi.min()
# 	levels = np.linspace(0,AJ_phi.max(),nlines*2)
# 	ax.contour(xp, zp, AJ_phi, levels=levels, colors='k')


# generate reductions for mdot
# def reductions(dumpno):	
# 	print("Analyzing {} dump {:04d}".format(params['model'], dumpno))

#     # header info
# 	header    = open(os.path.join(params['dumpsdir_{}'.format(model.lower())], 'dump_0000{0:04d}'.format(dumpno)), 'r')
# 	firstline = header.readline()
# 	header.close()
# 	firstline = firstline.split()

# 	madtype = int(firstline[0])
# 	metric  = firstline[9]
# 	n1      = int(firstline[11])
# 	n2      = int(firstline[12])

# 	gam  = float(firstline[15])
# 	dx1  = float(firstline[20])
# 	dx2  = float(firstline[21])
# 	ndim = int(firstline[22])	
# 	if metric == 'FMKS':
# 		rEH = float(firstline[28])
# 		a   = float(firstline[31])
# 		t   = float(firstline[32])
# 	elif metric == 'MKS':
# 		rEH = float(firstline[25])
# 		a   = float(firstline[28])
# 		t   = float(firstline[29])

# 	# loading prims
# 	prims = np.loadtxt(os.path.join(params['dumpsdir_{}'.format(model.lower())], 'dump_0000{0:04d}'.format(dumpno)),skiprows=1)
# 	rho = prims[:,0].reshape((n1,n2))
# 	U1  = prims[:,2].reshape((n1,n2))
# 	U2  = prims[:,3].reshape((n1,n2))
# 	U3  = prims[:,4].reshape((n1,n2))
# 	B1  = prims[:,5].reshape((n1,n2))
# 	B2  = prims[:,6].reshape((n1,n2))
# 	B3  = prims[:,7].reshape((n1,n2))

# 	# reading grid file
# 	grid  = np.loadtxt(os.path.join(params['dumpsdir_{}'.format(model.lower())],'grid'))
# 	x     = grid[:,0].reshape((n1,n2))
# 	z     = grid[:,1].reshape((n1,n2))
# 	r     = grid[:,2].reshape((n1,n2))
# 	th    = grid[:,3].reshape((n1,n2))
# 	x1    = grid[:,4].reshape((n1,n2))
# 	x2    = grid[:,5].reshape((n1,n2))
# 	gdet  = grid[:,6].reshape((n1,n2))
# 	lapse = grid[:,7].reshape((n1,n2))
# 	gcon  = grid[:,8:24].reshape((n1, n2, ndim, ndim))
# 	gcov  = grid[:,24:].reshape((n1, n2, ndim, ndim))

# 	# compute four velocity

# 	uvec   = np.append(np.append(U1[Ellipsis,None], U2[Ellipsis, None], axis=-1), U3[Ellipsis, None], axis=-1)
# 	gti    = gcon[Ellipsis,0,1:4]
# 	gij    = gcov[Ellipsis,1:4,1:4]
# 	beta_i = np.einsum('ijs,ij->ijs', gti, lapse**2)
# 	qsq    = np.einsum('ijy,ijy->ij', np.einsum('ijxy,ijx->ijy', gij, uvec), uvec)
# 	gamma  = np.sqrt(1 + qsq)
# 	ui     = uvec - np.einsum('ijs,ij->ijs', beta_i, gamma / lapse)
# 	ut     = gamma / lapse
# 	ucon   = np.append(ut[Ellipsis,None], ui, axis=-1)

# 	# compute accretion rate
# 	rEH_ind = np.argmin(abs(r[:,0] - rEH))
# 	mdot    = -np.sum((rho * ucon[Ellipsis,1])[rEH_ind,:] * gdet[rEH_ind,:], axis=-1) * dx2

# 	# save reductions
# 	# print(t, mdot)
# 	np.savetxt(os.path.join(params['outputdir'], 'reduction_{}_{:04d}.txt'.format(model.lower(), dumpno)), [t, mdot])


# if __name__=="__main__":
	# for m, model in enumerate(['GR', 'dCS', 'EdGB']):
	# 	params['model']   = model
	# 	params['dfirst']  = int(sorted(glob.glob(os.path.join(params['dumpsdir_{}'.format(model.lower())], 'dump*')))[0][-4:])
	# 	params['dlast']   = int(sorted(glob.glob(os.path.join(params['dumpsdir_{}'.format(model.lower())], 'dump*')))[-1][-4:])
	# 	params['dlist']   = range(params['dfirst'], params['dlast']+1)

		# ncores   = psutil.cpu_count(logical=True)
		# pad      = 0.5
		# nthreads = int(ncores * pad)
		# run_parallel(reductions, params['dlist'], nthreads)
        
time_gr, mdot_gr = [], []
time_dcs, mdot_dcs = [], []
time_edgb, mdot_edgb = [], []

for m, model in enumerate(['GR', 'dCS', 'EdGB']):
	data_files = sorted(glob.glob(os.path.join(params['outputdir'], 'reduction_{}*.txt'.format(model.lower()))))
	for file in data_files:
		with open(file, 'r') as f:
			lines = f.readlines()
			time = float(lines[0].strip())
			mdot = float(lines[1].strip())

		if model == 'GR':
			time_gr.append(time)
			mdot_gr.append(mdot)
		elif model == 'dCS':
			time_dcs.append(time)
			mdot_dcs.append(mdot)
		else:
			time_edgb.append(time)
			mdot_edgb.append(mdot)


# Plotting
plt.figure(figsize=(11,6))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(time_gr, mdot_gr, label='GR')
plt.plot(time_dcs, mdot_dcs, label='dCS')
plt.plot(time_edgb, mdot_edgb, label='EdGB')
plt.xlabel('T [$\\frac{GM}{c^3}$]', fontsize=15)
plt.ylabel(r'$\dot{M}$', fontsize=15)
plt.xlim(0,4000)
plt.ylim(0,0.2)
plt.legend()
plt.title('Accretion Rate for GR, dCS, and EdGB')
plt.savefig('accretion_rate.png')
plt.close()