import numpy as np
import matplotlib.pyplot as plt

plt.ion()

# s(t)
T = 50.0
sigma = 100.0
dt = 1.0
t = np.arange(-1000.0, 1000.0, dt)
s = s = np.cos(2*np.pi*t / T) * np.exp(- t**2 / (2*sigma**2))

plt.plot(t, s)

# Fourier transform, S(f)
S = np.fft.fft(s)
#S = np.abs(S)
f = np.fft.fftfreq(n=len(s), d=dt)

# omega
w = 2 * np.pi * f
dw = w[1] - w[0]

# integral{S(w).exp(iwt).dw}
ss = np.zeros(len(t), dtype='complex')
for i in range(len(t)):
    ss[i] = (S[:]*np.exp(1j*w[:]*t[i])*dw).sum()

plt.figure()
plt.plot(t, ss)