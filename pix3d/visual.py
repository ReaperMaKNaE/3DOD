def dcpn(variable):
    return variable.detach().cpu().numpy()

num = 1
import open3d as o3d
from pytorch3d.ops import cubify
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d

font_size = 18
ph,pw = 1,3
elev = -60; azim = -90
elev_nocs = -60;azim_nocs = -90

mesh = o3d.io.read_triangle_mesh('/vscode/pix3d/model/bed/IKEA_BEDDINGE/model.obj')



plt.figure(figsize=(25, 10))
plt.subplot(ph,pw,1)
plt.imshow(rgb[num].permute(1,2,0).detach().cpu())

ax = plt.subplot(ph, pw, 2, projection='3d')
vert = dcpn(target_meshes.verts_list()[num])
face = dcpn(target_meshes.faces_list()[num])

pc = art3d.Poly3DCollection(vert[face], edgecolor="black", rasterized=True)
ax.add_collection(pc)

ax.set(xlim=[vert.min(0)[0], vert.max(0)[0]], ylim=[vert.min(0)[1], vert.max(0)[1]],
       zlim=[vert.min(0)[2], vert.max(0)[2]])

ax.set(xlabel='xlabel', ylabel='ylabel', zlabel='zlabel')

ax.set_title('pred_meshes 0', fontdict={'fontsize': font_size})
ax.view_init(elev=elev, azim=azim)

# ax.set(xlim=[-1.0, 1.0], ylim=[-1.0, 1.0], zlim=[-1.0, 1.0])

ax = plt.subplot(ph, pw, 3, projection='3d')

tmp = dcpn(target_meshes.verts_list()[num])

ax.scatter(tmp[:, 0], tmp[:, 1], tmp[:, 2], c=tmp[:, 2], cmap='viridis', linewidth=0.5)
ax.set(xlabel='xlabel', ylabel='ylabel', zlabel='zlabel')
ax.set_title('prior', fontdict={'fontsize': font_size})
ax.view_init(elev=elev_nocs, azim=azim_nocs)
# ax.set(xlim=[-0.5, 0.5], ylim=[-0.5, 0.5], zlim=[-0.5, 0.5])
plt.show()