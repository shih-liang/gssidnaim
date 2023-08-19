import numpy as np
from scipy.signal import hilbert
from scipy.stats import zscore
import fbpca
from scipy.signal import medfilt2d

maskfile = "demodata/mask.npy"
ncomp = 20

def cpca(input_data, n_comps, n_iter=20, l = 10):
    l += n_comps
    n_samples = input_data.shape[0]
    (U, s, Va) = fbpca.pca(input_data, k = n_comps, n_iter = n_iter, l = l)
    explained_variance_ = (s ** 2) / (n_samples - 1)
    total_var = explained_variance_.sum()
    VaH = Va.conjugate().T
    pc_scores = input_data @ VaH
    loadings =  VaH @ np.diag(s)
    loadings /= np.sqrt(input_data.shape[0] - 1)
    output_dict = {
                   'U': U,
                   's': s,
                   'Va': Va,
                   'loadings': loadings.conjugate().T,
                   'exp_var': explained_variance_,
                   'pc_scores': pc_scores,
                   'total_var': total_var
                   }
    return output_dict

def img2matrix(mask, v):
    msize = np.count_nonzero(mask)
    result = np.zeros((v.shape[2], msize))
    m = 0
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if mask[i,j]:
                result[:,m] = v[i,j]
                m += 1
    return result

def matrix2img(mask, v):
    result = np.full((v.shape[0], mask.shape[0], mask.shape[1]), np.nan, dtype = 'complex_')
    t = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                result[:,i,j] = v[:,t]
                t += 1
    return result

def matrix2img2(mask, v):
    result = np.full((mask.shape[0], mask.shape[1]), np.nan, dtype = 'complex_')
    t = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                result[i,j] = v[t]
                t += 1
    return result

def reconstruct_ts(pca_res, n, real=True):
  U = pca_res['U'][:,n][0:2000,np.newaxis]
  s = np.atleast_2d(pca_res['s'][n])
  Va = pca_res['Va'][n,:].conj()[np.newaxis, :]
  recon_ts = U @ s @ Va
  if real:
    recon_ts = np.real(recon_ts)
  else:
    recon_ts = np.imag(recon_ts)
  return recon_ts

def group_norm_append(vs):
    return zscore(np.concatenate(vs,axis=0))

def cpca_loading(data, n, mask):
  loading = data['loadings'][n]
  return matrix2img2(mask, loading)

def cpca_reconstruct(pca_res, n, mask):
  recon_ts = reconstruct_ts(pca_res, n)
  return np.real(matrix2img(mask, recon_ts).transpose((1,2,0)))

def cpca_strength(data, n):
  if n < 0:
    return 0
  else:
    return data['exp_var'][n] / 9305 / 2

def psnr(v):
  real = np.real(v)
  imag = np.imag(v)
  real_filted = medfilt2d(real, kernel_size=7)
  imag_filted = medfilt2d(imag, kernel_size=7)
  r_mse = np.nanmean((real - real_filted) ** 2)
  r_psnr = 10 * np.log10((np.nanmax(real) ** 2) / r_mse)
  a_mse = np.nanmean((imag - imag_filted) ** 2)
  a_psnr = 10 * np.log10((np.nanmax(imag) ** 2) / a_mse)
  return r_psnr + a_psnr

# %%
def cpca_pipeline(nv, maskfile=maskfile):
  mask = np.load(maskfile)
  data = [np.load(v) for v in nv]
  pdata = group_norm_append([img2matrix(mask, i) for i in data])
  complex_data = hilbert(pdata, axis = 0)
  pca = cpca(complex_data, ncomp)
  return pca
