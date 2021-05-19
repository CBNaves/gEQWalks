import numpy as np

a = np.array([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]],[[1,2],[3,4],[5,6]]])
b = a[0:2,1:3]
b = b.reshape(4,2)
for i in b: print(i)
