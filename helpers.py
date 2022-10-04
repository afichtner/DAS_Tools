#==================================================================================================
#= Packages and figure embellishments. ============================================================
#==================================================================================================

import obspy
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from obspy.signal.filter import bandpass

plt.rcParams["font.family"] = "Times"
plt.rcParams.update({'font.size': 70})
plt.rcParams['xtick.major.pad']='12'
plt.rcParams['ytick.major.pad']='12'


#==================================================================================================
#= Plot record section. ===========================================================================
#==================================================================================================

def plot_section(d,x_start,nx,dx,nt,dt,vmax=0,tmin=0,tmax=0,interpolation='bicubic',plot=True,filename=None):
    """
    d: data matrix
    x_start: starting position of the first channel [m]
    nx: number of space samples
    dx: spatial sampling [m]
    nt: number of time samples
    dt: temporal sampling [s]
    vmax: maximum of color scale, autoscaling when vmax=0
    tmin, tmax: minimum and maximum of time axis for plotting, full range when vmin=vmax=0
    interpolation: matplotlib image interpolation, default is 'bicubic'
    plot: show image on screen, default is True, when False, image is still saved
    filename: output filename for image, no saving if set to None (default)
    """
    
    if vmax==0: 
    	scale=2.0*np.sqrt(np.mean(d**2))
    else:
    	scale=vmax
    
    if tmax==0: 
    	tmax=nt*dt
    	imax=nt
    else:
    	imax=np.int(tmax/dt)

    if tmin>0:
    	imin=np.int(tmin/dt)
    else:
    	imin=0

    fig=plt.figure(figsize=(60,20))
    im=plt.imshow(d[:,imin:imax], cmap='seismic', interpolation=interpolation, aspect='auto', vmin=-scale, vmax=scale, extent=(tmin,tmax,x_start+nx*dx,x_start))
    cbar = fig.colorbar(im)
    cbar.set_label('nanostrain rate [nanostrain/s]', rotation=270, labelpad=80)
    plt.xlabel('time [s]',labelpad=30)
    plt.ylabel('distance along cable [m]',labelpad=30)
    plt.minorticks_on()
    plt.grid(which='major',color='k',linewidth=2.0)
    plt.grid(which='minor',color='k',linewidth=0.5)
    plt.tight_layout()

    if filename: plt.savefig(filename,format='png', dpi=100)

    if plot==False: plt.close()
    if plot==True: plt.show()


#==================================================================================================
#= Crop record section in space and time. =========================================================
#==================================================================================================

def crop(d, t_start, t_end, dt, x_start, x_end, dx, x0):
    """
    Crop a record section in space and time.

    Input:
    ------
    d: data array organised as d[space indices, time indices]
    t_start, t_end: starting and ending times [s]
    x_start, x_end: starting and ending positions [m]
    dt, dx: time increment [s] and space increment [m]
    x0: starting position along the cable [m]

    Output:
    -------
    d_crop: cropped data array
    """

    # Starting and ending time indices.
    it_start=int(t_start/dt)
    it_end=int(t_end/dt)

    # Starting and ending space indices.
    ix_start=int((x_start-x0)/dx)
    ix_end=int((x_end-x0)/dx)

    # New number of time samples.
    nt=it_end-it_start
    
    # New number of space samples.
    nx=ix_end-ix_start

    # Trim the data.
    d_crop=d[ix_start:ix_end,it_start:it_end]

    # Return.
    return d_crop, nt, nx

#==================================================================================================
#= Compare pairs of traces in the time and frequency domains. =====================================
#==================================================================================================

def compare_traces(trace1, trace2, nt, dt, freqmin, freqmax, tmin_plot=0.0, tmax_plot=0.0):
    
    # Frequency-domain comparison. ===============================================
        
    # Make frequency axis.
    f=np.linspace(-0.5/dt,0.5/dt,nt)

    # Compute discrete Fourier transform.
    df1=np.abs(np.fft.fft(trace1))
    df2=np.abs(np.fft.fft(trace2))
    # Smooth spectra.
    for k in range(3): 
        df1[1:nt-1]=(df1[1:nt-1]+df1[0:nt-2]+df1[2:nt])/3.0
        df2[1:nt-1]=(df2[1:nt-1]+df2[0:nt-2]+df2[2:nt])/3.0
    # Roll in order to plot in the physical frequency domain.
    df1=np.roll(df1,int((nt-1)/2))
    df2=np.roll(df2,int((nt-1)/2))

    # Plot.
    plt.figure(figsize=(25,15))
    plt.loglog(f,df1,'k')
    plt.loglog(f,df2,'r',alpha=0.5)
    plt.xlim([freqmin,freqmax])
    plt.xlabel('f [Hz]')
    plt.minorticks_on()
    plt.grid(which='major',color='k',linewidth=2.0)
    plt.grid(which='minor',color='k',linewidth=0.5)
    plt.grid()
    plt.show()
    
    # Time-domain comparison. ====================================================
    
    # Make time axis.
    t=np.linspace(0.0,nt*dt,nt)
    
    # Apply a temporal taper to the data.
    width=np.int(freqmin*dt)
    for i in range(width): 
        trace1[i]=(np.float(i+1)/np.float(width+1))*trace1[i]
        trace1[nt-i-1]=(np.float(i+1)/np.float(width+1))*trace1[nt-i-1]
        trace2[i]=(np.float(i+1)/np.float(width+1))*trace2[i]
        trace2[nt-i-1]=(np.float(i+1)/np.float(width+1))*trace2[nt-i-1]
    
    # Frequency-domain filtering.
    di1_filt=bandpass(trace1[:],freqmin=freqmin,freqmax=freqmax,df=1.0/dt,corners=4,zerophase=True)
    di2_filt=bandpass(trace2[:],freqmin=freqmin,freqmax=freqmax,df=1.0/dt,corners=4,zerophase=True)

    # Plot.
    plt.figure(figsize=(25,15))
    plt.plot(t,di1_filt,'k')
    plt.plot(t,di2_filt,'r',alpha=0.5)
    plt.grid()
    
    if tmax_plot>0.0:
        plt.xlim(tmin_plot,tmax_plot)
    else:
        plt.xlim(t[0],t[-1])
    
    plt.xlabel('t [s]')
    plt.show()