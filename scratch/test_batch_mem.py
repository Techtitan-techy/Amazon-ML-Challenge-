import scipy.sparse as sp
import numpy as np

# Create synthetic sparse matrix similar to France
n_targets = 1434993
n_features = 100000

# Suppose 1.43M targets have ~15 tokens each
print("Creating target matrix...")
t_mat = sp.random(n_targets, n_features, density=0.00015, format='csr', dtype=np.float32)

print(f"Target matrix memory: {t_mat.data.nbytes / (1024*1024):.2f} MB")

# Test batch size 250 vs 2000
for b in [250, 1000]:
    q_mat = sp.random(b, n_features, density=0.0005, format='csr', dtype=np.float32)
    res = q_mat.dot(t_mat.T)
    print(f"Batch {b} dot result nnz: {res.nnz:,}, data size: {res.data.nbytes / (1024*1024):.2f} MB")
