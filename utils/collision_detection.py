import numpy as np
import numba as nb
from tqdm import trange
from pathlib import Path
from numba import cuda, prange, njit
import math
import torch


SCENE_IDS = [
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11, 12, 13, 14, 15,
    16, 17, 18, 20, 21,
    22, 23, 24, 25, 26
]


@cuda.jit(nogil=True, device=True)
def EuclideanDistance3(point_a: np.array, point_b: np.array) -> float:
    """
    Euclidean distance of two given point (3D)
    """
    # return float(np.linalg.norm((point_a - point_b), ord=2))
    return np.sqrt(np.sum(np.square(np.subtract(point_a, point_b))))


@nb.njit(nogil=True)
def VerticesPreprocess(vertices: np.array, xyz_limit: np.array):
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    xx = x[np.logical_and(z > xyz_limit[5], z < xyz_limit[4])]
    yy = y[np.logical_and(z > xyz_limit[5], z < xyz_limit[4])]
    zz = z[np.logical_and(z > xyz_limit[5], z < xyz_limit[4])]

    xxx = xx[np.logical_and(xx > xyz_limit[1], xx < xyz_limit[0])]
    yyy = yy[np.logical_and(xx > xyz_limit[1], xx < xyz_limit[0])]
    zzz = zz[np.logical_and(xx > xyz_limit[1], xx < xyz_limit[0])]

    xxxx = xxx[np.logical_and(yyy > xyz_limit[3], yyy < xyz_limit[2])]
    yyyy = yyy[np.logical_and(yyy > xyz_limit[3], yyy < xyz_limit[2])]
    zzzz = zzz[np.logical_and(yyy > xyz_limit[3], yyy < xyz_limit[2])]

    dims = np.zeros((len(xxxx), 3), dtype=np.float64)

    dims[:, 0] = xxxx
    dims[:, 1] = yyyy
    dims[:, 2] = zzzz

    return dims


# @nb.njit(nogil=True)
def VerticesProcessInParallel(all_vertices, center_points, length):
    vertices = []
    for idx in trange(len(center_points)):
        center_point = center_points[idx]
        # [x_max, x_min, y_max, y_min, z_max, z_min]
        x_max, y_max = [center_point[idx] + length / 2 for idx in range(2)]
        x_min, y_min = [center_point[idx] - length / 2 for idx in range(2)]
        z_max = 10e15; z_min = -10e15
        xyz_limit = np.array([x_max, x_min, y_max, y_min, z_max, z_min], dtype=np.float64)
        _vertices = VerticesPreprocess(all_vertices, xyz_limit)
        _vertices = np.array(_vertices, dtype=np.float64)
        vertices.append(_vertices)

    return vertices


class Simulator:
    def __init__(self, args) -> None:
        self.config = args
        self.all_vertices = self.BasicVertices()
        self.hierarchical_vertices = {}

    def BasicVertices(self, ):
        vertices_filename = Path(self.config.vertices_filepath) / 'vertices_{}.npy'.format(int(self.config.scene))
        all_vertices = np.load(vertices_filename)
        print('Converting AirSim mesh...')
        if self.config.xyz_limit:
            print('\Screen Vertices by xyz limit...\n')
            xyz_limit = np.array([10e3, -10e3, 10e3, -10e3, 10e2, -10e2], dtype=np.float64)
            all_vertices = VerticesPreprocess(all_vertices, xyz_limit)

        return all_vertices

    def HierarchicalVertices(
        self,
        potential_points,
        step_length=200
    ):
        x_min, y_min, z_min = (min(potential_points[:, dim]) for dim in range(3))
        x_max, y_max, z_max = (max(potential_points[:, dim]) for dim in range(3))

        grid = np.mgrid[  # [start : end) : step
            int(x_min - step_length / 2):int(x_max + step_length / 2 + 1):int(step_length),
            int(y_min - step_length / 2):int(y_max + step_length / 2 + 1):int(step_length),
        ]
        grid_x, grid_y = (grid[idx].flatten().reshape((-1, 1)) for idx in range(2))
        grid_z = np.array(
            [
                (z_max - z_min) / 2
                for idx in range(len(grid_x))
            ],
            dtype=np.float64
        ).reshape((-1, 1))
        num_zone = len(grid_x)
        center_points = np.concatenate(
            [grid_x, grid_y, grid_z],
            axis=1
        )

        all_vertices = self.all_vertices
        vertices = VerticesProcessInParallel(all_vertices, center_points, step_length)

        count = 0
        for idx in range(len(vertices)):
            count = count + len(vertices[idx])

        print('There are', int(len(all_vertices) - count), "points remained with", len(all_vertices), "in total")
        # assert (len(all_vertices) - count)<0.2*len(all_vertices), 'mismatch'

        hierarchical_vertices = {
            'Centers': center_points,
            'Vertices': vertices,
        }

        self.hierarchical_vertices = hierarchical_vertices

        return hierarchical_vertices


@cuda.jit(device=True)
def IsPointInPolyWoker(
    point_a: np.array,
    point_b: np.array,
    vertice_point: np.array,
) -> bool:
    distance_z = abs(float(point_a[2] - point_b[2]))
    distance_xy = (float(point_a[0] - point_b[0]) ** 2 + float(point_a[1] - point_b[1]) ** 2) ** (0.5)

    tolorance = float(0.1)
    radius = float(0.2)

    if distance_z < tolorance:
        if abs(point_a[1] - point_b[1]) < tolorance:
            x_min = min(point_a[0], point_b[0])
            x_max = max(point_a[0], point_b[0])
            y_min = float(point_a[1] - radius)
            y_max = float(point_a[1] + radius)
            if (x_min < vertice_point[0]) and (x_max > vertice_point[0]) and \
                    ((y_min < vertice_point[1]) and (y_max > vertice_point[1])):
                # print('True')
                return True
            else:
                return False
        elif abs(point_a[0] - point_b[0]) < tolorance:
            y_min = min(point_a[1], point_b[1])
            y_max = max(point_a[1], point_b[1])
            x_min = float(point_a[0] - radius)
            x_max = float(point_a[0] + radius)
            if (x_min < vertice_point[0]) and (x_max > vertice_point[0]) and \
                    ((y_min < vertice_point[1]) and (y_max > vertice_point[1])):
                return True
            else:
                return False
        else:
            x_min = min(point_a[0], point_b[0])
            x_max = max(point_a[0], point_b[0])
            y_min = min(point_a[1], point_b[1])
            y_max = max(point_a[1], point_b[1])
            if (x_min < vertice_point[0]) and (x_max > vertice_point[0]) and \
                    ((y_min < vertice_point[1]) and (y_max > vertice_point[1])):
                return True
            else:
                return False
    elif distance_xy < tolorance:
        x_min = float(point_a[0] - radius)
        x_max = float(point_a[0] + radius)
        y_min = float(point_a[1] - radius)
        y_max = float(point_a[1] + radius)
        z_min = min(point_a[2], point_b[2])
        z_max = max(point_a[2], point_b[2])
        if (x_min < vertice_point[0]) and (x_max > vertice_point[0]) and \
                ((y_min < vertice_point[1]) and (y_max > vertice_point[1])) and \
                ((z_min < vertice_point[2]) and (z_max > vertice_point[2])):
            return True
        else:
            return False
    else:
        return False


@cuda.jit
def IsPointInPolyKernel(point_a, point_b, all_vertice, results):
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    xStride = cuda.blockDim.x * cuda.gridDim.x

    if tx < len(all_vertice):
        vertice_point = all_vertice[tx]

        distance_z = abs(float(point_a[2] - point_b[2]))
        distance_xy = (float(point_a[0] - point_b[0]) ** 2 + float(point_a[1] - point_b[1]) ** 2) ** (0.5)

        tolorance = float(0.1)
        radius = float(0.2)

        if distance_z < tolorance:
            if abs(point_a[1] - point_b[1]) < tolorance:
                x_min = min(point_a[0], point_b[0])
                x_max = max(point_a[0], point_b[0])
                y_min = float(point_a[1] - radius)
                y_max = float(point_a[1] + radius)
                if (x_min < vertice_point[0]) and (x_max > vertice_point[0]) and \
                        ((y_min < vertice_point[1]) and (y_max > vertice_point[1])):
                    # return True
                    results[tx] = True
            #         else:
            #             # return False
            #             results[tx] = False
            elif abs(point_a[0] - point_b[0]) < tolorance:
                y_min = min(point_a[1], point_b[1])
                y_max = max(point_a[1], point_b[1])
                x_min = float(point_a[0] - radius)
                x_max = float(point_a[0] + radius)
                if (x_min < vertice_point[0]) and (x_max > vertice_point[0]) and \
                        ((y_min < vertice_point[1]) and (y_max > vertice_point[1])):
                    # return True
                    results[tx] = True
            #         else:
            #             # return False
            #             results[tx] = False
            else:
                x_min = min(point_a[0], point_b[0])
                x_max = max(point_a[0], point_b[0])
                y_min = min(point_a[1], point_b[1])
                y_max = max(point_a[1], point_b[1])
                if (x_min < vertice_point[0]) and (x_max > vertice_point[0]) and \
                        ((y_min < vertice_point[1]) and (y_max > vertice_point[1])):
                    # return True
                    results[tx] = True
        #         else:
        #             # return False
        #             results[tx] = False
        elif distance_xy < tolorance:
            x_min = float(point_a[0] - radius)
            x_max = float(point_a[0] + radius)
            y_min = float(point_a[1] - radius)
            y_max = float(point_a[1] + radius)
            z_min = min(point_a[2], point_b[2])
            z_max = max(point_a[2], point_b[2])
            if (x_min < vertice_point[0]) and (x_max > vertice_point[0]) and \
                    ((y_min < vertice_point[1]) and (y_max > vertice_point[1])) and \
                    ((z_min < vertice_point[2]) and (z_max > vertice_point[2])):
                # return True
                results[tx] = True
        #     else:
        #         # return False
        #         results[tx] = False
        # else:  # 其他
        #     # return False
        #     results[tx] = False


def IsPointInPolyCuda(point_a, point_b, all_vertice, results):
    point_a = cuda.to_device(point_a)
    point_b = cuda.to_device(point_b)
    all_vertice_gpu = cuda.to_device(all_vertice)

    threadsperblock = (1024)
    blockspergrid_x = int(math.ceil(len(all_vertice) / threadsperblock))
    blockspergrid = (blockspergrid_x)

    assert blockspergrid_x <= 65535, 'Cuda Error, not capable'

    cuda.synchronize()
    IsPointInPolyKernel[blockspergrid, threadsperblock](point_a, point_b, all_vertice_gpu, results)
    cuda.synchronize()

    results = results.copy_to_host()
    # print(results)
    return results


class Collision:
    def __init__(self, vertices_path, load_scenes=SCENE_IDS, SlicedMap=False, potential_points=-1) -> None:
        self.vertices_path = str(vertices_path)

        if SlicedMap:
            config = ''
            self.config = config
            assert potential_points != -1, 'points on path required'
            sim = Simulator(self.config)
            h_v = sim.HierarchicalVertices(potential_points, self.config.step_length)
            self.centres = h_v['Centers']
            self.vertices = h_v['Vertices']
        else:
            self.vertices = self._LoadFiles(scene_ids=load_scenes)

        results = np.array([False for idx in range(len(self.vertices))], dtype=np.bool_)
        results = cuda.to_device(results)
        self.result_array = results

    def _LoadFiles(self, scene_ids=SCENE_IDS):
        meshes = {}

        for scene_id in scene_ids:
            file_name_vertices = Path(str(self.vertices_path)) / 'vertices_{}.npy'.format(scene_id)
            mesh = np.load(str(file_name_vertices))
            meshes.update({
                int(scene_id): mesh
            })

        return meshes

    def IsCollision(
        self,
        point_source: np.array,
        point_target: np.array,
        scene_id: int
    ) -> bool:
        if torch.cuda.is_available():
            with torch.cuda.device(0):
                torch.cuda.empty_cache()

        results = IsPointInPolyCuda(point_source, point_target, self.vertices[int(scene_id)], self.result_array)
        if (results == True).any():
            return True
        else:
            return False


if __name__ == '__main__':
    import argparse
    import json
    import random
    import time
    random.seed(1)

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--vertices_path', type=str, default=str('./DATA/data/disceret/scene_meshes'))
    parser.add_argument('--dagger_mode_load_scene', nargs='+', default=[17])
    args = parser.parse_args()

    collision = Collision(args.vertices_path, load_scenes=args.dagger_mode_load_scene)

    scene_id = int(args.dagger_mode_load_scene[0])
    token_dict_path = './DATA/data/disceret/processed/token_dict_10'
    file_name_dict = Path(token_dict_path) / 'TokenDict_{}.json'.format(scene_id)
    with open(file_name_dict, 'r') as file:
        content_dict = json.load(file)
    nav_points = list(content_dict.values())

    for _ in range(16):
        time_start = time.time()
        result = []
        for i in range(16):
            point_source = nav_points[random.choice(range(len(nav_points)))]
            point_target = nav_points[random.choice(range(len(nav_points)))]
            is_collision = collision.IsCollision(point_source, point_target, scene_id)
            result.append(is_collision)
        time_end = time.time()
        print('time: {}'.format(time_end - time_start))
        print(result)

