import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

maskfile = "demodata/mask.npy"

seed_1 = [('M2 3',      50,      127-75),
          ('M2 2',      40,      55),
          ('M2 1',      30,      50),
          ('M1 1',      40,      40),
          ('BF 2',      70,      10),
          ('BF 1',      50,      10),
          ('FL',        40,      25),
          ('SSp-m',     40,      13),
          ('SSp-tr 2',  75,      127-110),
          ('HL',        60,      35),
          ('SSp-un',    90,      10),
          ('M1 2',      60,      45),
          ('SSp-tr 3',  80,      30),
          ('SSp-tr 1',  75,      40),
          ('M1 3',      75,      127-80),
          ('M2 4',      80,      55),
          ('PtA',       90,      40),
          ('RSP',       100,     127-75),
          ('VISam',     100,     40),
          ('VISp',      100,     25),
          ('VISal',     100,     12)]
seed_2 = [(i[0]+"'", i[1], 127 - i[2]) for i in seed_1]

seed_points = []
for i, j in zip(seed_1, seed_2):
  seed_points.append(i)
  seed_points.append(j)

def FDR_correlate(ps):
  reject, p_corrected, _, _ = multipletests(ps, method="fdr_bh")
  return reject, p_corrected

def cell_seed_significant(c14, c28, c56):
  seed_zs = [c14, c28, c56]
  t14, p14 = stats.ttest_1samp(seed_zs[0], 0)
  t28, p28 = stats.ttest_1samp(seed_zs[1], 0)
  t56, p56 = stats.ttest_1samp(seed_zs[2], 0)
  t14_28, p14_28 = stats.ttest_ind(seed_zs[0], seed_zs[1], equal_var=False)
  t28_56, p28_56 = stats.ttest_ind(seed_zs[1], seed_zs[2], equal_var=False)
  t14_56, p14_56 = stats.ttest_ind(seed_zs[0], seed_zs[2], equal_var=False)
  ps = [p14,p28,p56,p14_28,p28_56,p14_56]
  rejected, corrected = FDR_correlate(ps)
  return rejected, ps, corrected

def seed_image(seed, mask, d):
  def correlation(var1, var2):
    return np.corrcoef(var1, var2)[0, 1]
  si = seed[1]
  sj = seed[2]
  var1 = d[si, sj]
  result = np.full(mask.shape, np.nan)
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
      if mask[i,j]:
        result[i,j] = correlation(var1, d[i,j])
  result[result>=1] -= 1e-9
  fisher = np.arctanh(result)
  freenom = var1.shape[0]
  return result, fisher, freenom

def full_image(mask, d):
  size = mask.shape[0] * mask.shape[1]
  m = d.reshape((size, -1))
  result = np.corrcoef(m)
  fisher = np.arctanh(result)
  freenom = m.shape[0]
  return result, fisher, freenom

def group_level_image(mask, image_fisher_images_p14, image_fisher_images_p28, image_fisher_images_p56):
  image_ps_p14    = np.full(mask.shape, np.nan)
  image_ps_p28    = np.full(mask.shape, np.nan)
  image_ps_p56    = np.full(mask.shape, np.nan)
  image_ps_p14_28 = np.full(mask.shape, np.nan)
  image_ps_p28_56 = np.full(mask.shape, np.nan)
  image_ps_p14_56 = np.full(mask.shape, np.nan)

  image_rejected_p14    = np.full(mask.shape, False)
  image_rejected_p28    = np.full(mask.shape, False)
  image_rejected_p56    = np.full(mask.shape, False)
  image_rejected_p14_28 = np.full(mask.shape, False)
  image_rejected_p28_56 = np.full(mask.shape, False)
  image_rejected_p14_56 = np.full(mask.shape, False)

  image_corrected_p14    = np.full(mask.shape, np.nan)
  image_corrected_p28    = np.full(mask.shape, np.nan)
  image_corrected_p56    = np.full(mask.shape, np.nan)
  image_corrected_p14_28 = np.full(mask.shape, np.nan)
  image_corrected_p28_56 = np.full(mask.shape, np.nan)
  image_corrected_p14_56 = np.full(mask.shape, np.nan)

  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
      if mask[i,j]:
        c14 = [t[i, j] for t in image_fisher_images_p14]
        c28 = [t[i, j] for t in image_fisher_images_p28]
        c56 = [t[i, j] for t in image_fisher_images_p56]
        rejected, ps, corrected = cell_seed_significant(c14, c28, c56)

        image_ps_p14[i,j]    = ps[0]
        image_ps_p28[i,j]    = ps[1]
        image_ps_p56[i,j]    = ps[2]
        image_ps_p14_28[i,j] = ps[3]
        image_ps_p28_56[i,j] = ps[4]
        image_ps_p14_56[i,j] = ps[5]

        image_rejected_p14[i,j]    = rejected[0]
        image_rejected_p28[i,j]    = rejected[1]
        image_rejected_p56[i,j]    = rejected[2]
        image_rejected_p14_28[i,j] = rejected[3]
        image_rejected_p28_56[i,j] = rejected[4]
        image_rejected_p14_56[i,j] = rejected[5]

        image_corrected_p14[i,j]    = corrected[0]
        image_corrected_p28[i,j]    = corrected[1]
        image_corrected_p56[i,j]    = corrected[2]
        image_corrected_p14_28[i,j] = corrected[3]
        image_corrected_p28_56[i,j] = corrected[4]
        image_corrected_p14_56[i,j] = corrected[5]

  return ((image_ps_p14, image_ps_p28, image_ps_p56, image_ps_p14_28, image_ps_p28_56, image_ps_p14_56),
          (image_rejected_p14, image_rejected_p28, image_rejected_p56, image_rejected_p14_28, image_rejected_p28_56, image_rejected_p14_56),
          (image_corrected_p14, image_corrected_p28, image_corrected_p56, image_corrected_p14_28, image_corrected_p28_56, image_corrected_p14_56))

def seed_pipeline(seed, data_p14, data_p28, data_p56, maskfile=maskfile):
  def age_pipeline(seed, nv):
    mask = np.load(maskfile)
    data = [np.load(v) for v in nv]
    return mask, [seed_image(seed, mask, d) for d in data]
  mask_c14, data_c14 = age_pipeline(seed, data_p14)
  mask_c28, data_c28 = age_pipeline(seed, data_p28)
  mask_c56, data_c56 = age_pipeline(seed, data_p56)
  p_images, rejected_images, corrected_images = group_level_image(mask_c56, [d[1] for d in data_c14], [d[1] for d in data_c28], [d[1] for d in data_c56])
  return (mask_c14, data_c14, data_c28, data_c56), (p_images, rejected_images, corrected_images)

def full_image_pipeline(data_p14, data_p28, data_p56, maskfile=maskfile):
  def mask2coefmask(mask):
    m = mask.reshape((-1))
    result = np.full((m.shape[0], m.shape[0]), False)
    for i in range(m.shape[0]):
      for j in range(m.shape[0]):
        if m[i] == 1 and m[j] == 1:
          result[i,j] = True
    return result
  def age_pipeline(nv):
    mask = np.load(maskfile)
    data = [np.load(v) for v in nv]
    return mask2coefmask(mask), [full_image(mask, d) for d in data]
  mask_c14, data_c14 = age_pipeline(data_p14)
  mask_c28, data_c28 = age_pipeline(data_p28)
  mask_c56, data_c56 = age_pipeline(data_p56)
  p_images, rejected_images, corrected_images = group_level_image(mask_c56, [d[1] for d in data_c14], [d[1] for d in data_c28], [d[1] for d in data_c56])
  return (mask_c14, data_c14, data_c28, data_c56), (p_images, rejected_images, corrected_images)
