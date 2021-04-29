cuda = False

import numpy as np
if cuda:
    import cupy as cp
else:
    class cuda_ph:
        def __init__(self):
            self.zeros = np.zeros
            self.linalg = np.linalg
            self.sort = np.sort
            self.real = np.real
        
        def asnumpy(self,x):
            return x
    cp = cuda_ph()
import matplotlib.pyplot as plt
import time


class Honeycomb:
    def __init__(self,K,L,a):
        self.K = K
        self.L = L
        self.a = a
        self.basis_vec = a/2*np.array([1,3**(-0.5)])
        self.norm = np.linalg.norm(self.basis_vec)
        coords = []
        self.sublattice_A = []
        self.sublattice_B = []
        for l in range(L):
            for k in range(K):
                coords.append([k*a+a/2*(l%2),l*a/2*3**0.5])
                coords.append([k*a+a/2*(l%2)+self.basis_vec[0],l*a/2*3**0.5+self.basis_vec[1]])
                self.sublattice_A.append([k*a+a/2*(l%2),l*a/2*3**0.5])
                self.sublattice_B.append([k*a+a/2*(l%2)+self.basis_vec[0],l*a/2*3**0.5+self.basis_vec[1]])
        self.coords_trimmed = []
        for m in range(len(coords)):
            x1,y1 = coords[m] # site
            neighbours = 0
            for n in range(len(coords)):
                x2,y2 = coords[n] # test
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if self.norm*0.999 < dist and dist < self.norm*1.001:
                    neighbours += 1
            if neighbours > 1:
                self.coords_trimmed.append(coords[m])
    
    def plot(self):
        for x,y in self.coords_trimmed:
            if [x,y] in self.sublattice_A:
                plt.scatter(x,y,color="r")
            else:
                plt.scatter(x,y,color="b")
        plt.axis('off')
        # plt.savefig("lattice.png")
        plt.show()

class Hamiltonian:
    def __init__(self,lattice,onsite1,onsite2,hopping):
        
        self.lattice = lattice
        self.onsite1 = onsite1
        self.onsite2 = onsite2
        self.hopping = hopping
        self.generate_H_new()
        
    def generate_H(self):
        self.t0_h = time.time()
        N = len(self.lattice.coords_trimmed)
        
        H = cp.zeros((N,N))
        
        for m in range(N):
            x1,y1 = self.lattice.coords_trimmed[m] # site
            if self.lattice.coords_trimmed[m] in self.lattice.sublattice_A:
                H[m,m] = self.onsite1
            else:
                H[m,m] = self.onsite2
            for n in range(N):
                x2,y2 = self.lattice.coords_trimmed[n] # test
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if self.lattice.norm*0.999 < dist and dist < self.lattice.norm*1.001:
                    H[m,n] = self.hopping
                    H[n,m] = self.hopping
        self.H = H
        # print("H gen time",time.time() - self.t0_h)
        
    def generate_H_new(self):
        self.t0_hn = time.time()
        N = len(self.lattice.coords_trimmed)
        K = self.lattice.K
        H = cp.zeros((N,N))
        
        for m in range(N):
            x1,y1 = self.lattice.coords_trimmed[m] # site
            if self.lattice.coords_trimmed[m] in self.lattice.sublattice_A:
                H[m,m] = self.onsite1
            else:
                H[m,m] = self.onsite2
            for n in range(m-2*K,m+2*K):
                if n<=0 or n>=N:
                    pass
                else:
                    x2,y2 = self.lattice.coords_trimmed[n] # test
                    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if self.lattice.norm*0.999 < dist and dist < self.lattice.norm*1.001:
                        H[m,n] = self.hopping
                        H[n,m] = self.hopping
        self.H = H
        # print("H gen time",time.time() - self.t0_hn)
    
    def eigensys(self):
        eigensys = cp.linalg.eigh(self.H)
        return eigensys
    
    def eigenvals(self):
        t0_ev = time.time()
        eigensys = self.eigensys()
        eigensys = cp.sort(cp.real(eigensys[0]))
        # print("EVals time",time.time() - t0_ev)
        return eigensys

lattice = Honeycomb(10,10 ,1)
lattice.plot()

# ham = Hamiltonian(lattice,0.9,1.1,0.3)
# vals = ham.eigensys()
# vals1 = ham.eigenvals()


es = []
deltas = [-0.5 + 0.01*k for k in range(101)]
t0 = time.time()
for delta in deltas:
    ham = Hamiltonian(lattice,1-delta,1+delta,0.3)
    es.append(cp.asnumpy(ham.eigenvals()))
es = np.array(es)
print("runtime",time.time()-t0)
for k in range(len(es[0])):
    plt.plot(deltas,es.T[k],color="k")
plt.xlabel("on-site energy offset")
plt.ylabel("energy")
# plt.savefig("energy.png")
plt.show()

# ham1 = Hamiltonian(lattice,0.9,1.1,0.3)
# ham2 = Hamiltonian(lattice,0.9,1.1,0.3)
# ham2.generate_H_new()
# print(False in list((ham2.H == ham1.H).flatten()))
