import numpy as np
from facemath.geometry import generalized_procrustes, epsilon_isometry_distortion
from facemath.deformation import fit_thin_plate_spline

A=np.array([[0,0],[2,0],[1,1],[.5,2],[1.5,2],[1,3]],float)
B=A.copy(); B[3:,0]+=np.array([-.1,.1,0])
proc=generalized_procrustes(A,B)
eps,_=epsilon_isometry_distortion(proc.aligned_a,proc.aligned_b)
print({"procrustes":proc.distance,"epsilon_isometry":eps})
