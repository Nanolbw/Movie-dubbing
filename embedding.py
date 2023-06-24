import os
import numpy as np
from scipy import linalg

def compute_kernel_bias(vecs, n_components=256):
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    # np.nan_to_num(cov)
    # cov[np.isnan(cov)] = 0
    # cov[np.isinf(cov)] = 0
    #cov[np.where(cov==0)] = 1e-2
    cov += 1e-10
    #print(cov)
    u, s, vh = linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :n_components], -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """ 最终向量标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

# v_data = './preprocessed_data/prompt_copy/Bossbaby@BossBaby_00_0191_00.npy'
# v_data = np.load(v_data)[0]
# print(np.mean(v_data))
# v_data = np.array(v_data)    
# kernel,bias=compute_kernel_bias(v_data,256)
# v_data=transform_and_normalize(v_data, kernel=kernel, bias=bias)
# print(v_data.shape)
# print(np.mean(v_data))

path = './preprocessed_data/prompt/'
for file_name in os.listdir(path):
    print(file_name)
    v_data = np.load(os.path.join(path, file_name))[0]
    v_data = np.array(v_data)

    kernel,bias=compute_kernel_bias(v_data,256)
    v_data=transform_and_normalize(v_data, kernel=kernel, bias=bias)
    np.save(os.path.join(path, file_name), v_data)
