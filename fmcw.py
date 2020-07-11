#Radar Specifications 
#

# Frequency of operation = 77GHz
# Max Range = 200m
# Range Resolution = 1 m
# Max Velocity = 100 m/s
#%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Modified by the following matlab resource:

#https://www.mathworks.com/matlabcentral/answers/3131-fmcw-radar-range-doppler
#https://github.com/godloveliang/SFND-Radar-Target-Generation-and-Detection-

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

c=3e8 #speed of light
fc=77e9 #carrier freq
d_res = 1
Rmax = 200

deltaF=c/(2*d_res) #sweep freq
T=5.5*2*Rmax/c #one period
print(T)
alph=deltaF/T #sweep rate

R=80#initial distance of the target
#td=2*R/(c)#initial delay of returned signal
v=-50 #speed of the target (give some value between 0 and 10)
D=128 # #of doppler cells OR #of sent periods
N=1024 #for length of time
t=np.linspace(0,D*T,D*N)#total time

nT=len(t)/D #length of one period
a=np.zeros(len(t))#transmitted signal
b=np.zeros(len(t))#received signal
r_t=np.zeros(len(t))
ta=np.zeros(len(t))
r1=R
f0=fc
Mix = np.zeros(len(t))
for i in range(len(t)):
    r_t[i]=r1+v*t[i] # range of the target in terms of its velocity and initial range
    ta[i]=2*r_t[i]/c # delay for received signal
    a[i]=math.sin(2*math.pi*(f0*t[i]+.5*alph*t[i]**2)); #transmitted signal
    b[i]=math.sin(2*math.pi*(f0*(t[i]-ta[i])+.5*alph*(t[i]-ta[i])**2)); #received signal
    Mix[i] = a[i] * b[i]


#-------------------------------

print(Mix.shape)
print('nT',nT)
sig_fft = np.fft.fft(Mix,1024) / 1024
sig_fft = np.abs(sig_fft)  # Take the absolute value of FFT output
print(sig_fft.shape)
sig_fft = sig_fft[0:1024]

fig = plt.figure()
plt.plot(sig_fft)
plt.pause(1)


#------------------------------


m1=np.reshape(Mix,(1024,D)) #generating matrix ---> each row showing range info for one period AND each column showing number of periods
[Ny,My]=m1.shape

win=np.hamming(Ny)
win = win[:,np.newaxis]
print(m1.shape, (win*np.ones(My)).shape)
m2=np.conj(m1)*(win*np.ones(My)) #taking conjugate and applying window for sidelobe reduction (in time domain)
Win=np.fft.fft(np.hamming(My),D);
M2=(np.fft.fft(m2,2*N)); #First FFT for range information
M3=np.fft.fftshift(np.fft.fft(M2.T,2*D)) #Second FFT for doppler information
[My,Ny]=M3.shape
doppler=np.linspace(-D,D,My)
range1=np.linspace(-N,N,Ny)

X2, Y2 = np.meshgrid(range(Ny), range(My))

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X2,Y2,abs(M3), cmap='jet')
plt.show()


