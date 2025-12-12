import random 
import matplotlib.pyplot as plt 
	

plt.figure(figsize = (8, 4))
plt.hist([random.uniform(4, 32) for i in range(100000)], bins = 100)
plt.grid()
plt.savefig("uniform_hist.png")