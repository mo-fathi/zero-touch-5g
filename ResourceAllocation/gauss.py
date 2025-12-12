import random 
import matplotlib.pyplot as plt 
	
# # store the random numbers in a list 
# nums = [] 
mu = 0
sigma = 0.2
	
# for i in range(100): 
# 	temp = random.gauss(mu, sigma) 
# 	nums.append(temp) 
		
# # plotting a graph 
# plt.plot(nums) 
# plt.savefig("gauass.png")



plt.figure(figsize = (8, 4))
plt.hist([random.gauss(mu, sigma) for i in range(100000)], bins = 100)
plt.grid()
plt.savefig("gauss_hist.png")