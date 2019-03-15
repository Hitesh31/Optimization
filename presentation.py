import numpy as np
import cvxpy as cp
import random
m=40			#initial given sensors
k=15			#sensors to select
n=3
a=[]
b=[]
c=np.zeros((3,3))
z=cp.Variable((m,1))				
for i in range (0,m,1):
	a.append(np.random.randn(3,1))		#vectors a[i] taken from gaussian distribution
	b.append(np.matmul(a[i],a[i].T))	#computing matrix for each info (error covariance matrix)
	c=c+b[i]*z[i]
constraints=[z[0]<=1,z[0]>=0]
for i in range (1,m,1):
	constraints=constraints+[z[i]<=1,z[i]>=0] #constraints are linear
obj=cp.Maximize(cp.log_det(c))		#objective function
prob = cp.Problem(obj, constraints)
prob.solve() 
print("status:", prob.status)
print("optimal value", prob.value)