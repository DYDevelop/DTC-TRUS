import io, math, numpy as np, nibabel as nib, imageio.v2 as imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from PIL import Image
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

id = '15213598'
mode = 'img'
type ='axi'
nii = os.path.join(PROJECT_ROOT, "video_results", type, f"{id}_{mode}.nii.gz")

# 1) 메쉬 생성(간소화로 빠르게)
vol = nib.load(nii).get_fdata()
mask = (vol > 0.5).astype(np.uint8)
verts, faces, _, _ = marching_cubes(mask, level=0.5, step_size=3)  # step_size=2~4 권장(빠름)
verts = verts[:, [2,1,0]]
C = verts.mean(0); X = verts - C
_,_,Vt = np.linalg.svd(X, full_matrices=False)
R = Vt.T; aligned = X @ R

def build_ax(dpi=120, cmap_name='magma', bg='black', edge=True):
    fig = plt.figure(figsize=(6,6), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # 높이(=z) 기준 per-face 색상
    tri = aligned[faces]                  # (F, 3, 3)
    zmean = tri[:, :, 2].mean(axis=1)     # (F,)
    zmin, zmax = zmean.min(), zmean.max()
    znorm = (zmean - zmin) / (zmax - zmin + 1e-8)
    cmap = plt.get_cmap(cmap_name)
    facecolors = cmap(znorm)              # RGBA (F, 4)

    ec = ('white' if bg=='black' else 'black') if edge else 'none'
    lw = 0.2 if edge else 0.0

    mesh = Poly3DCollection(tri, facecolors=facecolors, edgecolor=ec, linewidths=lw, alpha=1.0)
    ax.add_collection3d(mesh)

    ax.set_xlim(aligned[:,0].min(), aligned[:,0].max())
    ax.set_ylim(aligned[:,1].min(), aligned[:,1].max())
    ax.set_zlim(aligned[:,2].min(), aligned[:,2].max())
    ax.set_axis_off()

    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    return fig, ax

def save_spin_gif(axis='z', frames=90, elev=15, fps=24, out_path='prostate.gif'):
    fig, ax = build_ax(dpi=120)
    with imageio.get_writer(out_path, mode='I', duration=1/fps) as gif:
        for t in range(frames):
            a = 360.0 * t / frames
            if axis == 'z':
                ax.view_init(elev=elev, azim=a)
            elif axis == 'x':
                ax.view_init(elev=-90 + a, azim=-90)
            else:  # 'y'
                ax.view_init(elev=elev, azim=a)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', facecolor=fig.get_facecolor(),
                        bbox_inches=None, pad_inches=0)
            gif.append_data(np.array(Image.open(buf)))
            buf.close()
    plt.close(fig)
    print(f"saved: {out_path}")

# 사용 예시(한 축만 저장)
# 저장 경로: 3D_recon/{환자이름}/{날짜시간}/파일명.gif
save_dir = os.path.join(SCRIPT_DIR, "3D_recon", id)
date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(save_dir, date_time_str)
os.makedirs(save_dir, exist_ok=True)
out_path = os.path.join(save_dir, f"id_{id}_{type}_{mode}.gif")
save_spin_gif(axis='z', out_path=out_path)