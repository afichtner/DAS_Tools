#==================================================================================================
#= Packages and figure embellishments. ============================================================
#==================================================================================================

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({'font.size': 70})
plt.rcParams['xtick.major.pad']='12'
plt.rcParams['ytick.major.pad']='12'


#==================================================================================================
#= f-k transform ==================================================================================
#==================================================================================================

def fk(d, dt, dx, distance_scaling=0.0, fmin_plot=0.0, fmax_plot=100.0, kmin_plot=0.0, kmax_plot=0.4, taper=0.05, plot=True, flip_axes=False, f_pick=[], k_pick=[], saturation=0.1, filename=None):
	"""
	Compute and plot the f-k transform of a 2D space-time data array.

	Input:
	------
	d: data matrix organised as d[space indices, time indices]
	dt: time increment [s]
	dx: space increment [m]
	distance_scaling: apply a scaling to the data proportional to r^distance_scaling, for no scaling set to 0
	fmin_plot, fmax_plot: minimum and maximum frequencies for plotting [Hz], can be negative
	kmin_plot, kmax_plot: minimum and maximum wavenumbers for plotting [1/m], can be negative
	taper: fraction of the time and space length for tapering prior to the f-k transform
	plot: flag for plotting
	flip_axes: flip x- and y-axis, default is f on the x-axis
	f_pick, k_pick: array of discrete samples of frequency and wavenumber along a dispersion curve, just for plotting on top of the f-k plot, set to False by default
	saturation: fraction of the maximum spectral amplitude where colour scale saturates 
	filename: provide filename for figure saving

	Output:
	-------
	f, k: frequency and wave number arrays
	d_fk_r: f-k transformed data, rolled such that the discrete Fourier transform is interpretable with continuous wave numbers and frequencies

	"""

	#- Preparations. ==============================================================================

	nx=np.shape(d)[0]
	nt=np.shape(d)[1]


	#- Apply distance scaling if wanted. ==========================================================

	if distance_scaling>0.0:

		for i in range(nx): d[i,:]=(dx*float(i))**0.5 * d[i,:]


	#- Apply taper in space and time. =============================================================

	# Apply a temporal taper to the data.
	width=int(nt*taper)
	for i in range(width): 
	    d[:,i]=(np.float(i+1)/np.float(width+1))*d[:,i]
	    d[:,nt-i-1]=(np.float(i+1)/np.float(width+1))*d[:,nt-i-1]
	    
	# Apply a spatial taper to the data.
	width=int(nx*taper)
	for i in range(width): 
	    d[i,:]=(np.float(i+1)/np.float(width+1))*d[i,:]
	    d[nx-i-1,:]=(np.float(i+1)/np.float(width+1))*d[nx-i-1,:]


	#- Compute f-k transform. =====================================================================

	# 2D Fourier transform.
	d_fk=np.fft.fft2(d)*dt*dx
	# Roll in order to plot in physical f-k domain.
	d_fk_r=np.roll(np.roll(d_fk,int((nt-1)/2),axis=1),int((nx-1)/2),axis=0)
	# Make frequency and wavenumber axes.
	f=np.linspace(-0.5/dt,0.5/dt,nt)
	k=np.linspace(-np.pi/dx,np.pi/dx,nx)
	ff,kk=np.meshgrid(f,k)


	#- Plotting. ==================================================================================

	if plot:

		# Compute minimum and maximum indices for plotting. This takes into account that in this application the k-axis is reversed.
		ifmin_plot=np.where(np.abs(f-fmin_plot)==np.min(np.abs(f-fmin_plot)))[0][0]
		ifmax_plot=np.where(np.abs(f-fmax_plot)==np.min(np.abs(f-fmax_plot)))[0][0]

		ikmax_plot=np.where(np.abs(-k-kmin_plot)==np.min(np.abs(-k-kmin_plot)))[0][0]
		ikmin_plot=np.where(np.abs(-k-kmax_plot)==np.min(np.abs(-k-kmax_plot)))[0][0]

		# Plot f-k domain amplitude spectrum.
		plt.subplots(1, figsize=(30,30))
		if flip_axes:
			plt.pcolor(-kk[ikmin_plot:ikmax_plot,ifmin_plot:ifmax_plot],ff[ikmin_plot:ikmax_plot,ifmin_plot:ifmax_plot],np.abs(d_fk_r[ikmin_plot:ikmax_plot,ifmin_plot:ifmax_plot]),cmap='Greys')
		else:
			plt.pcolor(ff[ikmin_plot:ikmax_plot,ifmin_plot:ifmax_plot],-kk[ikmin_plot:ikmax_plot,ifmin_plot:ifmax_plot],np.abs(d_fk_r[ikmin_plot:ikmax_plot,ifmin_plot:ifmax_plot]),cmap='Greys')

		# Superimpose lines of constant phase velocity.
		if flip_axes:
			for c in np.arange(500.0,4500.0,500.0):
			    plt.plot(2.0*np.pi*f/c,f,'--b',linewidth=5.0,alpha=0.25)
		else:
			for c in np.arange(500.0,4500.0,500.0):
			    plt.plot(f,2.0*np.pi*f/c,'--b',linewidth=5.0,alpha=0.25)

		# Superimpose the dispersion curve picks.
		if flip_axes:
			if len(f_pick)>0 and len(k_pick)>0:
				plt.plot(k_pick,f_pick,'ro',markersize=15)
		else:
			if len(f_pick)>0 and len(k_pick)>0:
				plt.plot(f_pick,k_pick,'ro',markersize=15)
			
		# Embellish figure.
		if flip_axes:
			plt.ylim(fmin_plot,fmax_plot)
			plt.xlim(kmin_plot,kmax_plot)
			plt.ylabel('f [Hz]',labelpad=30)
			plt.xlabel('k [1/m]',labelpad=30)
		else:
			plt.xlim(fmin_plot,fmax_plot)
			plt.ylim(kmin_plot,kmax_plot)
			plt.xlabel('f [Hz]',labelpad=30)
			plt.ylabel('k [1/m]',labelpad=30)
		
		plt.colorbar()

		max_color=saturation*np.max(np.abs(d_fk_r[ikmin_plot:ikmax_plot,ifmin_plot:ifmax_plot]))
		plt.clim([0.0,max_color])
		
		plt.minorticks_on()
		plt.grid(which='major',color='k',linewidth=4.0)
		plt.grid(which='minor',color='k',linewidth=1.5)
		plt.tight_layout()

		if filename: plt.savefig(filename,format='png',dpi=200)

		plt.show()

	#= Return. ====================================================================================

	return f, k, d_fk_r


#==================================================================================================
#= f-k filtering along dispersion curve ===========================================================
#==================================================================================================

def fk_filter_dispersion(d_fk_r, f, k, dt, dx, f_pick, k_pick, n_smooth=10, fmin_plot=0.0, fmax_plot=100.0, kmin_plot=0.0, kmax_plot=0.4, plot=True):
	"""
	f-k filtering along a dispersion curve.

	Input:
	------
	d_fk_r, f, k: f-k transform, frequency and wave number arrays provided by fk()
	dt, dx: time increment [s] and space increment [m]
	f_pick, k_pick: array of discrete samples of frequency and wavenumber along a dispersion curve
	n_smooth: number of times the mask is smoothed by averaging over neighbouring pixles in f-k space
	fmin_plot, fmax_plot: minimum and maximum frequencies for plotting [Hz], can be negative
	kmin_plot, kmax_plot: minimum and maximum wavenumbers for plotting [1/m], can be negative
	plot: plot the f-k filter mask with the dispersion curve samples superimposed

	Output:
	-------
	d_filtered: f-k filtered version of the data in the time-space domain

	"""

	#- Preparations. ==============================================================================

	# Initiate the filter mask.
	mask=np.zeros(np.shape(d_fk_r),dtype='complex64')

	# Number of discrete wave numbers and frequencies.
	nk=np.shape(d_fk_r)[0]
	nf=np.shape(d_fk_r)[1]


	#- Compute f-k filter mask. ===================================================================

	# Interpolation of the dispersion curve picks.
	interpolate_k=interpolate.interp1d(f_pick,k_pick,kind='cubic',fill_value="extrapolate")

	# Start and end indices in frequency of the dispersion curve.
	if_start=np.where(np.min(np.abs(f-f_pick[0]))==np.abs(f-f_pick[0]))[0][0]
	if_end=np.where(np.min(np.abs(f-f_pick[-1]))==np.abs(f-f_pick[-1]))[0][0]

	# March through all the frequencies to assign mask values along the curve.
	for i in np.arange(if_start,if_end+1):
	    
	    # Interpolate to find the wave number at that frequency.
	    k_interp=interpolate_k(f[i])
	    
	    # Find the index for that wave number.
	    ik=np.where(np.min(np.abs(-k-k_interp))==np.abs(-k-k_interp))[0][0]
	    
	    # Assign value 1 to the mask.
	    mask[ik-1:ik+1,i]=1.0

	# Smooth the mask.
	for l in range(n_smooth): mask[1:nk-1,1:nf-1]=(mask[1:nk-1,1:nf-1]+mask[0:nk-2,1:nf-1]+mask[2:nk,1:nf-1]+mask[1:nk-1,0:nf-2]+mask[1:nk-1,2:nf])/5.0


    #- Plot the f-k filter mask. ==================================================================

	if plot:
    	# Compute minimum and maximum indices for plotting. This takes into account that in this application the k-axis is reversed.
		ifmin_plot=np.where(np.abs(f-fmin_plot)==np.min(np.abs(f-fmin_plot)))[0][0]
		ifmax_plot=np.where(np.abs(f-fmax_plot)==np.min(np.abs(f-fmax_plot)))[0][0]

		ikmax_plot=np.where(np.abs(-k-kmin_plot)==np.min(np.abs(-k-kmin_plot)))[0][0]
		ikmin_plot=np.where(np.abs(-k-kmax_plot)==np.min(np.abs(-k-kmax_plot)))[0][0]

		# Plot f-k domain mask.
		ff,kk=np.meshgrid(f,k)
		plt.subplots(1, figsize=(30,30))
		plt.pcolor(ff[ikmin_plot:ikmax_plot,ifmin_plot:ifmax_plot],-kk[ikmin_plot:ikmax_plot,ifmin_plot:ifmax_plot],np.abs(mask[ikmin_plot:ikmax_plot,ifmin_plot:ifmax_plot]),cmap='Greys')

		# Superimpose the (interpolated) dispersion curve.
		plt.plot(f_pick,k_pick,'ro',markersize=15)
		f_interp=np.linspace(f_pick[0],f_pick[-1],100)
		k_interp=interpolate_k(f_interp)
		plt.plot(f_interp,k_interp,'r',linewidth=3)

		# Embellish the figure.
		plt.xlim(fmin_plot,fmax_plot)
		plt.ylim(kmin_plot,kmax_plot)
		plt.xlabel('f [Hz]',labelpad=30)
		plt.ylabel('k [1/m]',labelpad=30)
		plt.colorbar()

		plt.minorticks_on()
		plt.grid(which='major',color='k',linewidth=4.0)
		plt.grid(which='minor',color='k',linewidth=1.5)
		plt.tight_layout()

		plt.show()


	#- Apply mask, invert transform and return. ===================================================

	# Apply the f-k domain filter.
	d_fk_r_filtered=d_fk_r*mask

	# Roll back.
	nt=len(f)
	nx=len(k)
	d_fk_filtered=np.roll(np.roll(d_fk_r_filtered,-int((nt-1)/2),axis=1),-int((nx-1)/2),axis=0)

	# Inverse 2D Fourier transform.
	d_filtered=np.real(np.fft.ifft2(d_fk_filtered))/(dt*dx)

	# Return.
	return d_filtered



#==================================================================================================
#= f-k filtering along dispersion curve ===========================================================
#==================================================================================================

def fk_filter_velocity(d_fk_r, f, k, dt, dx, c_min=300.0, c_max=4000.0, freqmin=5.0, freqmax=80.0, n_smooth=10, fmin_plot=0.0, fmax_plot=100.0, kmin_plot=0.0, kmax_plot=0.4, plot=True):
	"""
	f-k filtering according to velocity and frequency boundaries.

	Input:
	------
	d_fk_r, f, k: f-k transform, frequency and wave number arrays provided by fk()
	dt, dx: time increment [s] and space increment [m]
	c_min, c_max: minimum and maximum phase velocities [m/s]
	freqmin, freqmax: minimum and maximum frequencies [Hz]
	n_smooth: number of times the mask is smoothed by averaging over neighbouring pixles in f-k space
	fmin_plot, fmax_plot: minimum and maximum frequencies for plotting [Hz], can be negative
	kmin_plot, kmax_plot: minimum and maximum wavenumbers for plotting [1/m], can be negative
	plot: plot the f-k filter mask

	Output:
	-------
	d_filtered: f-k filtered version of the data in the time-space domain

	"""

	#- Preparations. ==============================================================================

	# Initiate the filter mask.
	mask=np.ones(np.shape(d_fk_r),dtype='complex64')

	# Number of discrete wave numbers and frequencies.
	nx=np.shape(d_fk_r)[0]
	nt=np.shape(d_fk_r)[1]


	#- Compute f-k filter mask. ===================================================================

	# Find frequency indices for which the mask is non-zero.
	ifmin=np.where(np.abs(f-freqmin)==np.min(np.abs(f-freqmin)))[0][0]
	ifmax=np.where(np.abs(f-freqmax)==np.min(np.abs(f-freqmax)))[0][0]

	# Set mask outside these frequencies to zero.
	mask[:,0:ifmin]=0.0
	mask[:,ifmax:-1]=0.0

	# Compute the mask.
	for i in range(nx):
	    for j in np.arange(ifmin,ifmax):
	        
	        # Compute phase velocity.
	        if np.abs(k[i]):
	            c=2.0*np.pi*f[j]/k[i]
	        else:
	            c=1.0e9
	            
	        # Maximum and minimum absolute phase velocities.
	        if (np.abs(c)>c_max) or (np.abs(c)<c_min):
	            mask[i,j]=0.0
	        # Forward or backward propagation.
	        if c>0.0: mask[i,j]=0.0
	        # Bandpass filter.
	        if np.abs(f[j])>freqmax: 
	            mask[i,j]=0.0
	        if np.abs(f[j])<freqmin: 
	            mask[i,j]=0.0

	# Smooth the mask.
	for l in range(n_smooth): mask[1:nx-1,1:nt-1]=(mask[1:nx-1,1:nt-1]+mask[0:nx-2,1:nt-1]+mask[2:nx,1:nt-1]+mask[1:nx-1,0:nt-2]+mask[1:nx-1,2:nt])/5.0


	#- Plot the f-k filter mask. ==================================================================

	if plot:

	    # Compute minimum and maximum indices for plotting. This takes into account that in this application the k-axis is reversed.
		ifmin_plot=np.where(np.abs(f-fmin_plot)==np.min(np.abs(f-fmin_plot)))[0][0]
		ifmax_plot=np.where(np.abs(f-fmax_plot)==np.min(np.abs(f-fmax_plot)))[0][0]

		ikmax_plot=np.where(np.abs(-k-kmin_plot)==np.min(np.abs(-k-kmin_plot)))[0][0]
		ikmin_plot=np.where(np.abs(-k-kmax_plot)==np.min(np.abs(-k-kmax_plot)))[0][0]

		# Plot f-k domain mask.
		ff,kk=np.meshgrid(f,k)
		plt.subplots(1, figsize=(30,30))
		plt.pcolor(ff[ikmin_plot:ikmax_plot,ifmin_plot:ifmax_plot],-kk[ikmin_plot:ikmax_plot,ifmin_plot:ifmax_plot],np.abs(mask[ikmin_plot:ikmax_plot,ifmin_plot:ifmax_plot]),cmap='Greys')

		# Embellish the figure.
		plt.xlim(fmin_plot,fmax_plot)
		plt.ylim(kmin_plot,kmax_plot)
		plt.xlabel('f [Hz]',labelpad=30)
		plt.ylabel('k [1/m]',labelpad=30)
		plt.colorbar()

		plt.minorticks_on()
		plt.grid(which='major',color='k',linewidth=4.0)
		plt.grid(which='minor',color='k',linewidth=1.5)
		plt.tight_layout()

		plt.show()

	#- Apply mask, invert transform and return. ===================================================

	# Apply the f-k domain filter.
	d_fk_r_filtered=d_fk_r*mask

	# Roll back.
	nt=len(f)
	nx=len(k)
	d_fk_filtered=np.roll(np.roll(d_fk_r_filtered,-int((nt-1)/2),axis=1),-int((nx-1)/2),axis=0)

	# Inverse 2D Fourier transform.
	d_filtered=np.real(np.fft.ifft2(d_fk_filtered))/(dt*dx)

	# Return.
	return d_filtered


#==================================================================================================
#= f-s transform ==================================================================================
#==================================================================================================

def fs(d_fk_r, f, k, f_pick, k_pick, fmin_plot=0.0, fmax_plot=100.0, smin_plot=0.0002, smax_plot=0.003, plot=True, flip_axes=False, saturation=0.1, filename=None):
	"""
	Compute and visualise the frequency-slowness transform based on the f-k transform obtained from fk().

	Input:
	------
	d_fk_r, f, k: f-k transform, frequency and wave number arrays provided by fk()
	f_pick, k_pick: array of discrete samples of frequency and wavenumber along a dispersion curve
	fmin_plot, fmax_plot: minimum and maximum frequencies for plotting [Hz], can be negative
	smin_plot, smax_plot: minimum and maximum slownesses for plotting [s/m], can be negative
	plot: flag for plotting
	flip_axes: flip x- and y-axis, default is f on the x-axis
	saturation: fraction of the maximum spectral amplitude where colour scale saturates 
	filename: provide filename for figure saving
	
	Output:
	-------
	f,s: frequency and slowness arrays
	d_fs: frequency-slowness transform
	"""

	#- Preparations. ==============================================================================

	nt=np.shape(d_fk_r)[1]


	#- Compute frequency-slowness transform from f-k transform. ===================================

	s=np.linspace(smin_plot,smax_plot,len(k))
	d_fs=np.zeros(np.shape(d_fk_r))

	for i in np.arange(int(nt/2),nt):
	    # Initiate the interpolation function.
	    f_interp=interpolate.interp1d(k, np.abs(d_fk_r[:,i]), kind='cubic', bounds_error=False, fill_value=0.0)
	    # Array of k values that corresponds to the s values.
	    k_interp=-2.0*np.pi*s*f[i]
	    # Interpolate.
	    d_fs[:,i]=f_interp(k_interp)


	#- Plot. ======================================================================================

	if plot:

		# Compute indices for plotting.
		ifmin_plot=np.where(np.abs(f-fmin_plot)==np.min(np.abs(f-fmin_plot)))[0][0]
		ifmax_plot=np.where(np.abs(f-fmax_plot)==np.min(np.abs(f-fmax_plot)))[0][0]
		ismin_plot=np.where(np.abs(s-smin_plot)==np.min(np.abs(s-smin_plot)))[0][0]
		ismax_plot=np.where(np.abs(s-smax_plot)==np.min(np.abs(s-smax_plot)))[0][0]

		# Plot.
		ff,ss=np.meshgrid(f,s)

		plt.subplots(1, figsize=(30,30))
		if flip_axes:
			plt.pcolor(ss[ismin_plot:ismax_plot,ifmin_plot:ifmax_plot],ff[ismin_plot:ismax_plot,ifmin_plot:ifmax_plot],d_fs[ismin_plot:ismax_plot,ifmin_plot:ifmax_plot],cmap='Greys')
		else:
			plt.pcolor(ff[ismin_plot:ismax_plot,ifmin_plot:ifmax_plot],ss[ismin_plot:ismax_plot,ifmin_plot:ifmax_plot],d_fs[ismin_plot:ismax_plot,ifmin_plot:ifmax_plot],cmap='Greys')
		
		if len(f_pick)>0 and len(k_pick)>0:
			if flip_axes:
				plt.plot(k_pick/(2.0*np.pi*f_pick),f_pick,'ro',markersize=15)
			else:
				plt.plot(f_pick,k_pick/(2.0*np.pi*f_pick),'ro',markersize=15)

		if flip_axes:
			plt.ylim(fmin_plot,fmax_plot)
			plt.xlim(smin_plot,smax_plot)
			plt.ylabel('f [Hz]',labelpad=30)
			plt.xlabel('s [s/m]',labelpad=30)
		else:
			plt.xlim(fmin_plot,fmax_plot)
			plt.ylim(smin_plot,smax_plot)
			plt.xlabel('f [Hz]',labelpad=30)
			plt.ylabel('s [s/m]',labelpad=30)
		
		max_color=saturation*np.max(np.abs(d_fs[ismin_plot:ismax_plot,ifmin_plot:ifmax_plot]))
		plt.clim([0.0,max_color])
		
		plt.minorticks_on()
		plt.grid(which='major',color='k',linewidth=4.0)
		plt.grid(which='minor',color='k',linewidth=1.5)
		plt.tight_layout()
		
		if filename: plt.savefig(filename,format='png',dpi=200)

		plt.show()

	#= Return. ====================================================================================

	return f, s, d_fs


#==================================================================================================
#= f-c transform ==================================================================================
#==================================================================================================

def fc(d_fk_r, f, k, f_pick, k_pick, fmin_plot=0.0, fmax_plot=100.0, cmin_plot=200.0, cmax_plot=5000.0, plot=True, flip_axes=False, saturation=0.1, filename=None):
	"""
	Compute and visualise the frequency-velocity transform based on the f-k transform obtained from fk().

	Input:
	------
	d_fk_r, f, k: f-k transform, frequency and wave number arrays provided by fk()
	f_pick, k_pick: array of discrete samples of frequency and wavenumber along a dispersion curve
	fmin_plot, fmax_plot: minimum and maximum frequencies for plotting [Hz], can be negative
	cmin_plot, cmax_plot: minimum and maximum velocities for plotting [m/s], can be negative
	plot: flag for plotting
	flip_axes: flip x- and y-axis, default is f on the x-axis
	saturation: fraction of the maximum spectral amplitude where colour scale saturates 
	filename: provide filename for figure saving
	
	Output:
	-------
	f,c: frequency and slowness arrays
	d_fs: frequency-slowness transform
	"""

	#- Preparations. ==============================================================================

	nt=np.shape(d_fk_r)[1]


	#- Compute frequency-slowness transform from f-k transform. ===================================

	s=np.linspace(1.0/cmax_plot,1.0/cmin_plot,len(k))
	d_fs=np.zeros(np.shape(d_fk_r))

	for i in np.arange(int(nt/2),nt):
	    # Initiate the interpolation function.
	    f_interp=interpolate.interp1d(k, np.abs(d_fk_r[:,i]), kind='cubic', bounds_error=False, fill_value=0.0)
	    # Array of k values that corresponds to the s values.
	    k_interp=-2.0*np.pi*s*f[i]
	    # Interpolate.
	    d_fs[:,i]=f_interp(k_interp)


	#- Plot. ======================================================================================

	if plot:

		# Compute indices for plotting.
		c=1.0/s
		ifmin_plot=np.where(np.abs(f-fmin_plot)==np.min(np.abs(f-fmin_plot)))[0][0]
		ifmax_plot=np.where(np.abs(f-fmax_plot)==np.min(np.abs(f-fmax_plot)))[0][0]
		icmax_plot=np.where(np.abs(c-cmin_plot)==np.min(np.abs(c-cmin_plot)))[0][0]
		icmin_plot=np.where(np.abs(c-cmax_plot)==np.min(np.abs(c-cmax_plot)))[0][0]

		# Plot.
		ff,ss=np.meshgrid(f,s)
		cc=1.0/ss

		plt.subplots(1, figsize=(30,30))

		if flip_axes:
			plt.pcolor(cc[icmin_plot:icmax_plot,ifmin_plot:ifmax_plot],ff[icmin_plot:icmax_plot,ifmin_plot:ifmax_plot],d_fs[icmin_plot:icmax_plot,ifmin_plot:ifmax_plot],cmap='Greys')
		else:
			plt.pcolor(ff[icmin_plot:icmax_plot,ifmin_plot:ifmax_plot],cc[icmin_plot:icmax_plot,ifmin_plot:ifmax_plot],d_fs[icmin_plot:icmax_plot,ifmin_plot:ifmax_plot],cmap='Greys')

		if len(f_pick)>0 and len(k_pick)>0:
			if flip_axes:
				plt.plot((2.0*np.pi*f_pick)/k_pick,f_pick,'ro',markersize=15)
			else:
				plt.plot(f_pick,(2.0*np.pi*f_pick)/k_pick,'ro',markersize=15)
				
		if flip_axes:
			plt.ylim(fmin_plot,fmax_plot)
			plt.xlim(cmin_plot,cmax_plot)
			plt.ylabel('f [Hz]',labelpad=30)
			plt.xlabel('c [m/s]',labelpad=30)
		else:
			plt.xlim(fmin_plot,fmax_plot)
			plt.ylim(cmin_plot,cmax_plot)
			plt.xlabel('f [Hz]',labelpad=30)
			plt.ylabel('c [m/s]',labelpad=30)

		max_color=saturation*np.max(np.abs(d_fs[icmin_plot:icmax_plot,ifmin_plot:ifmax_plot]))
		plt.clim([0.0,max_color])

		plt.minorticks_on()
		plt.grid(which='major',color='k',linewidth=4.0)
		plt.grid(which='minor',color='k',linewidth=1.5)
		plt.tight_layout()
		
		if filename: plt.savefig(filename,format='png',dpi=200)

		plt.show()

	#= Return. ====================================================================================

	return f, c, d_fs