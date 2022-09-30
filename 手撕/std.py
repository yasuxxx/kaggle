import numpy as np
arr = [2,3,5,7,9]
xhat = np.mean(arr)
xhat2 = sum(arr)/len(arr)
xvar = np.var(arr)
xvar2 = sum([(i-xhat2)**2 for i in arr])/len(arr)
print(xvar,xvar2)
xstd = np.std(arr)