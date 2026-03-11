import numpy as np
import glob
import matplotlib.pyplot as plt
import skimage
import os
from PIL import Image
from skimage.filters import meijering, sato, frangi, hessian
import copy

from scipy import ndimage as ndi
from skimage import exposure, filters, morphology, measure, util
from skimage.filters import sato, frangi, apply_hysteresis_threshold
from skimage.segmentation import clear_border


def normalize01(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, (.1, 99.9))
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    return x


def make_brain_mask(I, sigma_bg=8, min_size=20000, close_radius=15):
    """
    粗脑组织mask：对I平滑→Otsu→形态学闭运算→填洞→去小连通域
    """
    Is = filters.gaussian(I, sigma=sigma_bg, preserve_range=True)
    th = filters.threshold_otsu(Is)
    m = Is > th

    m = morphology.binary_closing(m, morphology.disk(close_radius))
    m = ndi.binary_fill_holes(m)
    m = morphology.remove_small_objects(m, min_size=min_size)
    # 保留最大连通域（通常脑组织是最大的那块）
    lab = measure.label(m)
    if lab.max() > 0:
        areas = np.bincount(lab.ravel())
        areas[0] = 0
        m = lab == areas.argmax()
    return m



root_path = os.path.join(os.getcwd(),'Light-sheet/241030_652__11-18-58')
paths = glob.glob(os.path.join(root_path, '1_original/8bit/*'))
print(paths)

volumn = []
for path in paths:
    im = Image.open(path)
    imarray = np.array(im)
    volumn.append(imarray)


volumn = np.array(volumn).astype(np.float32) / 255.0
print(volumn.shape)


#volumn_ = skimage.filters.frangi(volumn, sigmas=range(1, 15, 2), scale_range=None, scale_step=None, 
#                                 alpha=0.5, beta=0.5, gamma=None, black_ridges=True, mode='reflect', cval=0)
#print('frangi finished')

save_dir = os.path.join(root_path, 'frangi')
os.makedirs(save_dir, exist_ok=True)
for i, arr_ in enumerate(volumn[[0]]):


    arr = copy.deepcopy(arr_)
    arr_norm = arr / arr.max() if arr.max() != 0 else arr
    arr_uint8 = (arr_norm * 255).astype(np.uint8)

    thr = 100
    mask_thr = arr_uint8>thr
    #mask = apply_hysteresis_threshold(arr_uint8, low=80, high=200)
    mask_thr = morphology.binary_closing(mask_thr, morphology.disk(1))
    mask_thr = morphology.remove_small_objects(mask_thr, min_size=15)

    arr_uint8[~mask_thr] = 0
    
    #arr = sato(arr_uint8, sigmas=[1,2,3,4,5,6,7,8,9,10], black_ridges=False)
    #arr_norm = arr / arr.max() if arr.max() != 0 else arr
    #arr_uint8 = (arr_norm * 255).astype(np.uint8)

    img = Image.fromarray(arr_uint8)
    save_path = os.path.join(save_dir, f'thr_{i:04d}.tif')
    img.save(save_path)





    #arr = frangi(arr_, sigmas=range(1, 30, 3), scale_range=None, scale_step=None, 
    #                             alpha=0.5, beta=0.5, gamma=None, black_ridges=True, mode='reflect', cval=0)
    
    arr = filters.gaussian(arr_, sigma=1, preserve_range=True)
    #arr = frangi(arr, sigmas=np.arange(.1,20,1), black_ridges=False)
    arr = sato(arr, sigmas=np.arange(.01,4,.2), black_ridges=False)
    #arr = meijering(arr, sigmas=np.arange(.01,4,.2), black_ridges=False)
    #arr = hessian(ararrr_, sigmas=[1,2,3,3,5])
    print(i)

    # Normalize per-slice (or globally if preferred)
    arr_norm = arr / arr.max() if arr.max() != 0 else arr
    arr_uint8 = (arr_norm * 255).astype(np.uint8)

    img = Image.fromarray(arr_uint8)
    save_path = os.path.join(save_dir, f'sato_{i:04d}.tif')
    img.save(save_path)


    thr = 15
    mask = arr_uint8>thr
    #mask = apply_hysteresis_threshold(arr_uint8, low=15, high=100)
    #mask = filters.threshold_local(arr_uint8, block_size=11,method='gaussian',offset=-.01)
    mask = morphology.binary_closing(mask, morphology.disk(1))
    mask = morphology.remove_small_objects(mask, min_size=15)
    
    mask = mask*mask_thr
    
    arr_[~mask] = 0    
    img = Image.fromarray(arr_)
    save_path = os.path.join(save_dir, f'sato_thr_{i:04d}.tif')
    img.save(save_path)

    mask = (mask * 255).astype(np.uint8)
    img = Image.fromarray(mask)
    save_path = os.path.join(save_dir, f'sato_thr_mask_{i:04d}.tif')
    img.save(save_path)







    #I = exposure.equalize_adapthist(arr_, clip_limit=0.01)
    #img = Image.fromarray(I)
    #save_path = os.path.join(save_dir, f'brain_{i:04d}.tif')
    #img.save(save_path)

