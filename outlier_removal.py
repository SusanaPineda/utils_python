import open3d as o3d
import argparse 

def radius_outlier_removal(point_cloud, min_number_points, radius):
    return point_cloud.remove_radius_outlier(nb_points=int(points), radius=float(radius))


##Leer Argumentos de entrada
ap = argparse.ArgumentParser()
ap.add_argument("--ipc", required = True, help = "path of pcd file")
ap.add_argument("--points", required = True, help = "number of points")
ap.add_argument("--radius", required = True, help = "radius")
ap.add_argument("--opc", required = True, help = "path of resulting pcd file")

args = vars(ap.parse_args())

inputURL = args['ipc']
points = args['points']
radius = args['radius']
outputURL = args['opc']


pcd = o3d.io.read_point_cloud(inputURL)
# Visualización de los datos de entrada
o3d.visualization.draw_geometries([pcd])


cl, ind = radius_outlier_removal(pcd, points, radius)

# Salida por pantalla del resultado
inlier_cloud = pcd.select_down_sample(ind)
outlier_cloud = pcd.select_down_sample(ind, invert=True)
outlier_cloud.paint_uniform_color([1, 0, 0])
inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

o3d.io.write_point_cloud(outputURL,cl)