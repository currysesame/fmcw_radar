import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

# test code
# this code mainly from 
# https://github.com/PreSenseRadar/OpenRadar/tree/master/Presense%20Applied%20Radar\
# https://www.mathworks.com/matlabcentral/answers/3131-fmcw-radar-range-doppler
# https://github.com/godloveliang/SFND-Radar-Target-Generation-and-Detection-

# Read in frame data

frame = np.load('doppler_example_1.npy')

# Manually cast to signed ints
frame.real = frame.real.astype(np.int16)
frame.imag = frame.imag.astype(np.int16)

print('Shape of frame: {frame.shape}')


frame = np.transpose(frame, (0, 2, 1))


frame_ = frame

# for i in range(128):
# 	for j in range(8):
# 		# when i = 0 ~ 15 in string first
# 		frame_[j + (i%16) * 8 ,:,i//16] = frame[i,:,j]

frame_ = np.reshape(frame_, (128*128*8))
print(frame_.shape)
frame_ = frame_[0:(128*128*8 - 100*128*8)]
print(frame_.shape)
frame_ = np.reshape(frame_, (28,128,8))
frame_ = np.concatenate((frame_, np.zeros((100,128,8))), axis = 0)


# Meta data about the data
num_chirps = 128 # Number of chirps in the frame
num_samples = 128 # Number of ADC samples per chirp

num_tx = 2
num_rx = 4
num_vx = num_tx * num_rx # Number of virtual antennas

range_plot = np.fft.fft(frame, axis=1)

# Visualize Results
# plt.imshow(np.abs(range_plot.sum(1)).T)
# plt.ylabel('Range Bins')
# plt.title('Interpreting a Single Frame - Range')
# plt.show()

range_doppler = np.fft.fft(range_plot, axis=0)
range_doppler = np.fft.fftshift(range_doppler, axes=0)

# Visualize Results
# plt.imshow(np.log(np.abs(range_doppler).T).sum(1))
# plt.xlabel('Doppler Bins')
# plt.ylabel('Range Bins')
# plt.title('Interpreting a Single Frame - Doppler')
# plt.show()



num_angle_bins = 64
""" REMOVE """
padding = ((0,0), (0,0), (0,num_angle_bins-range_doppler.shape[2]))
print(range_doppler.shape)
range_azimuth = np.pad(range_doppler, padding, mode='constant')
range_azimuth = np.fft.fft(range_azimuth, axis=2)



# Visualize Results
plt.imshow(np.log(np.abs(range_azimuth).sum(0)))
plt.xlabel('Azimuth (Angle) Bins')
plt.ylabel('Range Bins')
plt.title('Interpreting a Single Frame - Azimuth')
plt.show()


sum_r = np.log(np.abs(range_azimuth).sum(0))
[My, Ny] = sum_r.shape
print(sum_r.shape)
X2, Y2 = np.meshgrid(range(Ny), range(My))

print(X2.shape, Y2.shape)
#X2, Y2 = np.meshgrid(doppler_axis, range_axis)
#print(X2.shape, Y2.shape)
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X2,Y2,sum_r, cmap='jet')
plt.show()








range_plot = np.fft.fft(frame_, axis=1)

# Visualize Results
# plt.imshow(np.abs(range_plot.sum(1)).T)
# plt.ylabel('Range Bins')
# plt.title('Interpreting a Single Frame - Range')
# plt.show()

range_doppler = np.fft.fft(range_plot, axis=0)
range_doppler = np.fft.fftshift(range_doppler, axes=0)

# Visualize Results
# plt.imshow(np.log(np.abs(range_doppler).T).sum(1))
# plt.xlabel('Doppler Bins')
# plt.ylabel('Range Bins')
# plt.title('Interpreting a Single Frame - Doppler')
# plt.show()



num_angle_bins = 64
""" REMOVE """
padding = ((0,0), (0,0), (0,num_angle_bins-range_doppler.shape[2]))
print(range_doppler.shape)
range_azimuth = np.pad(range_doppler, padding, mode='constant')
range_azimuth = np.fft.fft(range_azimuth, axis=2)



# Visualize Results
plt.imshow(np.log(np.abs(range_azimuth).sum(0)))
plt.xlabel('Azimuth (Angle) Bins')
plt.ylabel('Range Bins')
plt.title('Interpreting a Single Frame - Azimuth')
plt.show()


sum_r = np.log(np.abs(range_azimuth).sum(0))
[My, Ny] = sum_r.shape
print(sum_r.shape)
X2, Y2 = np.meshgrid(range(Ny), range(My))

print(X2.shape, Y2.shape)
#X2, Y2 = np.meshgrid(doppler_axis, range_axis)
#print(X2.shape, Y2.shape)
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X2,Y2,sum_r, cmap='jet')
plt.show()

