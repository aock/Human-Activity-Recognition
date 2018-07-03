import numpy as np
import matplotlib.pyplot as plt

n = 500
seq_len = 200
step = 5

x = []
y = []

c = 0
for c,i in enumerate( range(0,n-seq_len+step,step) ):
    y.append([])
    x.append([])
    y[-1] = np.ones(seq_len) * c
    x[-1] = np.arange(i, i+200)


plt.figure(1)
for i in range(c):
    plt.plot(x[i],y[i])

plt.title('Anzahl Sequenzen: %d' % c)
plt.xlabel('Sensordatum')
plt.ylabel('Sequenz Nummer')
plt.show()


