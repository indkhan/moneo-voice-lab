p='routers.py'
s=open(p,encoding='utf-8').read()
old="                np.add.at(E, ids, -self.elr * dh0)  # plain SGD on the sparse rows\n"
new='''                u, c = np.unique(ids, return_counts=True)  # sparse Adagrad on the embedding rows that were used
                g = c[:, None] * dh0[None, :]
                G[u] += g * g
                E[u] -= self.elr * g / (np.sqrt(G[u]) + 1e-6)
'''
assert old in s
s=s.replace(old,new)
s=s.replace("        lr, step = 0.01, 0\n","        lr, step = 0.01, 0\n        G = np.zeros_like(E)\n")
s=s.replace("seed=0, elr=0.05):","seed=0, elr=0.1):")
open(p,'w',encoding='utf-8').write(s)
