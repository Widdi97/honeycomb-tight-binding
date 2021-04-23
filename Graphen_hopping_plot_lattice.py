import numpy as np
import matplotlib.pyplot as plt

def eigenenergies(delta):
    hopping = 0.3
    
    onsite1 = 1 - delta # sublattice A
    onsite2 = 1 + delta # sublattice B
    
    L = 4 # hight
    K = 5 # width
    
    a = 1
    basis_vec = a/2*np.array([1,3**(-0.5)])
    norm = np.linalg.norm(basis_vec)
    
    coords = []
    sublattice_A = []
    sublattice_B = []
    for l in range(L):
        for k in range(K):
            coords.append([k*a+a/2*(l%2),l*a/2*3**0.5])
            coords.append([k*a+a/2*(l%2)+basis_vec[0],l*a/2*3**0.5+basis_vec[1]])
            sublattice_A.append([k*a+a/2*(l%2),l*a/2*3**0.5])
            sublattice_B.append([k*a+a/2*(l%2)+basis_vec[0],l*a/2*3**0.5+basis_vec[1]])
    
    coords_trimmed = []
    for m in range(len(coords)):
        x1,y1 = coords[m] # site
        neighbours = 0
        for n in range(len(coords)):
            x2,y2 = coords[n] # test
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if norm*0.999 < dist and dist < norm*1.001:
                neighbours += 1
        if neighbours > 1:
            coords_trimmed.append(coords[m])
    global ploted
    if not ploted:
        for x,y in coords_trimmed:
            if [x,y] in sublattice_A:
                plt.scatter(x,y,color="r")
            else:
                plt.scatter(x,y,color="b")
        plt.show()
        ploted = True
    
    N = len(coords_trimmed)
    
    H = np.zeros((N,N))
    
    for m in range(N):
        x1,y1 = coords_trimmed[m] # site
        if coords_trimmed[m] in sublattice_A:
            H[m,m] = onsite1
        else:
            H[m,m] = onsite2
        for n in range(N):
            x2,y2 = coords_trimmed[n] # test
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if norm*0.999 < dist and dist < norm*1.001:
                H[m,n] = hopping
                H[n,m] = hopping
    
    eigensys = np.linalg.eig(H)
    return np.sort(np.real(eigensys[0]))


ploted = False
es = []
deltas = [-0.5 + 0.01*k for k in range(101)]
for delta in deltas:
    es.append(eigenenergies(delta))
es = np.array(es)
# for k in range(len(deltas)):
#     plt.scatter([deltas[k] for n in range(len(es[0]))],es[k])
# plt.show()
for k in range(len(es[0])):
    plt.plot(deltas,es.T[k],color="k")
plt.show()
    
    
    