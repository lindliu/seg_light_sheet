


import numpy as np
import glob
import matplotlib.pyplot as plt
import skimage
import os
from PIL import Image
from skimage.filters import meijering, sato, frangi, hessian
import copy
import qim3d
import re
import tifffile as tiff

from scipy import ndimage as ndi
from skimage import exposure, filters, morphology, measure, util
from skimage.filters import sato, frangi, apply_hysteresis_threshold
from skimage.segmentation import clear_border



import open3d as o3d
import matplotlib.cm as cm
def plot_3d_show(masks, value_map=None, cmap_name="turbo"):
    points = np.argwhere(masks > 0)  ## masks (Z,X,Y), points (Z,X,Y)
    
    # max_pts = 200000
    # if points.shape[0] > max_pts:
    #     points = points[np.random.choice(points.shape[0], max_pts, replace=False)]

    if value_map is not None:
        vals = value_map[masks > 0].astype(np.float64)
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
        
        norm = (vals - vmin) / (vmax - vmin)
        cmap = cm.get_cmap(cmap_name)
        colors = cmap(norm)[:, :3].astype(np.float64)

        # colors = np.repeat(norm[:, None], 3, axis=1)
        # colors = np.clip(colors, 0.0, 1.0)
    else:        
        # 自定义颜色，每个点对应一个 RGB 值（范围 [0, 1]）
        colors = np.ones_like(points, dtype=float)*.5   

    # 构建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 可视化
    o3d.visualization.draw_geometries([pcd])


import plotly.graph_objects as go
def plot_3d_save(mask, spacing_zyx=[1,1,1], value_map=None, save_html=None, seed=0):
    pts = np.argwhere(mask)
    print("points:", pts.shape[0])

    max_pts = 200000
    if pts.shape[0] > max_pts:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pts.shape[0], max_pts, replace=False)
        pts = pts[idx]
    else:
        idx = None

    if value_map is not None:
        value_map = np.asarray(value_map)
        val = value_map[pts[:, 0], pts[:, 1], pts[:, 2]].astype(np.float32)
        # val = value_map[mask]
    else:
        val = np.ones(pts.shape[0], dtype=np.float32)

    pz, py, px = spacing_zyx
    z = pts[:,0]*pz
    y = pts[:,1]*py
    x = pts[:,2]*px

    fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode="markers",
                                marker=dict(size=1, opacity=0.5, color=val,
                                            colorscale="Turbo")))
    fig.update_layout(scene=dict(aspectmode="data"))


        # ---- saving ----
    if save_html is not None:
        fig.write_html(save_html)
        print("Saved HTML:", save_html)

    # fig.show()


def get_loc_thichness(paths, step=1):
    volumn = []
    for path in paths[::step]:
        array = np.array(Image.open(path))>0
        array = array[::step,::step]
        volumn.append(array)
    volumn = np.array(volumn)
    loc_thichness = qim3d.processing.local_thickness(volumn, visualize=False, axis=0)
    return loc_thichness


num_re = re.compile(r'(\d+)(?!.*\d)')

# root_path = glob.glob(os.path.join(os.getcwd(),'data/241030_652__11-18-58'))[0]
root_paths = glob.glob(os.path.join(os.getcwd(),'data/*'))
print(root_paths)


for root_path in root_paths[:1]:
    
    print(root_path)
    tif_paths = glob.glob(os.path.join(root_path, '2_8bit/*'))
    tif_paths = sorted(tif_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

    filter_dir = '3_filter_mask_99.4'
    filter_mask_paths = glob.glob(os.path.join(root_path, f'{filter_dir}/*.png'))
    filter_mask_paths = sorted(filter_mask_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

    thr_dir = '3_thr_mask_99.0'
    thr_mask_paths = glob.glob(os.path.join(root_path, f'{thr_dir}/*.png'))
    thr_mask_paths = sorted(thr_mask_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))



    save_mask_dir = os.path.join(root_path, '4_mask')
    save_thickness_dir = os.path.join(root_path, '5_thickness')
    os.makedirs(save_mask_dir, exist_ok=True)
    os.makedirs(save_thickness_dir, exist_ok=True)

    save_tif = True  #  False # 
    i = 0
    for filter_mask_path, thr_mask_path in zip(filter_mask_paths, thr_mask_paths):
        print(os.path.split(filter_mask_path)[1], os.path.split(thr_mask_path)[1])

        mask_filter = np.array(Image.open(filter_mask_path))>0
        mask_thr = np.array(Image.open(thr_mask_path))>0

        ############ final vessel mask by mask_filter+mask_thr ##############
        mask = (mask_filter + mask_thr)>0

        if save_tif:
            mask = mask.astype(np.uint8) * 255
            save_path = os.path.join(save_mask_dir, f'mask_{i:04d}.tif')
            tiff.imwrite(save_path, mask)
        else:
            save_path = os.path.join(save_mask_dir, f'mask_{i:04d}.png')
            img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
            img.save(save_path)
        i += 1
        #####################################################################
    
    if save_tif:
        final_mask_paths = glob.glob(os.path.join(save_mask_dir, f'*.tif'))
    else:
        final_mask_paths = glob.glob(os.path.join(save_mask_dir, f'*.png'))
    final_mask_paths = sorted(final_mask_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

    step = 4
    ############ loc thichness of final combined mask #############
    final_loc = get_loc_thichness(final_mask_paths, step=1)
    save_html = os.path.join(root_path, 'loc_final.html')
    plot_3d_save(final_loc[::step,::step,::step]>0, value_map=final_loc[::step,::step,::step], save_html=save_html)

    for i in range(final_loc.shape[0]):
        save_path = os.path.join(save_thickness_dir, f'loc_{i:04d}.tif')
        tiff.imwrite(save_path, final_loc[i])
