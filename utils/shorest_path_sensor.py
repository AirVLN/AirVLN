import os
import tqdm
import time
import numpy as np
import numba as nb
import networkx as nx
import json
from pathlib import Path
import igraph as ig


SCENE_IDS = [
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11, 12, 13, 14, 15,
    16, 17, 18, 20, 21,
    22, 23, 24, 25, 26
]


@nb.njit(nogil=True, cache=True)
def EuclideanDistance3(point_a: np.array, point_b: np.array) -> float:
    """
    Euclidean distance of two given point (3D)
    """
    # return float(np.linalg.norm((point_a - point_b), ord=2))
    return float(np.sqrt(np.sum(np.square(np.subtract(point_a, point_b)))))


@nb.njit(nogil=True, cache=True)
def EuclideanDistance1(point_a: np.array, point_b: np.array) -> float:
    """
    Euclidean distance of two given point (1D)
    """
    return float(np.sqrt(np.square(float(point_a - point_b))))


def Distance(edge, termial, token_dict):
    point_a = np.array(token_dict[edge])
    point_b = np.array(token_dict[termial])
    distance = EuclideanDistance3(point_a, point_b)
    return float(distance)


class ShortestPathSensor:
    def __init__(self, nav_graph_path, token_dict_path, load_scenes=SCENE_IDS) -> None:
        self.nav_graph_path = nav_graph_path
        self.token_dict_path = token_dict_path

        self.load_scenes = load_scenes

        # build
        # self.graphs, self.token_dicts = self._BuildNXGraphs(scene_ids=load_scenes)

        # load only
        self.graphs, self.token_dicts = self._LoadNXGraphs(scene_ids=load_scenes)

    def _LoadOriFiles(self, scene_idx):
        file_name_graph = Path(args.ori_nav_graph_path) / 'nav_graph_dict_{}.json'.format(scene_idx)
        file_name_dict = Path(args.ori_token_dict_path) / 'TokenDict_{}.json'.format(scene_idx)

        with open(file_name_graph, 'r', encoding='utf-8') as file:
            content_graph = json.load(file)
        with open(file_name_dict, 'r', encoding='utf-8') as file:
            content_dict = json.load(file)

        return content_graph, content_dict

    def _BuildNXGraphs(self, scene_ids=SCENE_IDS):
        graphs = {}
        token_dicts = {}

        for scene_id in scene_ids:
            print("BuildNXGraphs of Scene {}".format(scene_id))

            graph_dict, token_dict = self._LoadOriFiles(int(scene_id))

            G = nx.Graph()

            nodes = list(graph_dict.keys())
            G.add_nodes_from(nodes)

            edges = []
            weights_dict = {}
            pbar = tqdm.tqdm(total=len(nodes))
            for start_node in nodes:
                pbar.update()
                for end_node in graph_dict[start_node]:
                    weight = Distance(start_node, end_node, token_dict)
                    weights_dict[str('{}_{}'.format(start_node, end_node))] = weight
                    edges.append(
                        (start_node, end_node, weight)
                    )
            pbar.close()
            G.add_weighted_edges_from(edges)

            largest = max(nx.connected_components(G), key=len)
            largest_connected_subgraph = G.subgraph(largest)


            nodes = list(largest_connected_subgraph.nodes())

            token_dict_sub = {}
            for node in nodes:
                token_dict_sub.update({
                    node: token_dict[node]
                })
            token_dicts.update({
                int(scene_id): token_dict_sub
            })

            if not os.path.exists(str(Path(args.token_dict_path))):
                os.makedirs(str(Path(args.token_dict_path)), exist_ok=True)
            with open(str(Path(args.token_dict_path) / 'TokenDict_{}.json'.format(scene_id)), 'w', encoding='utf-8') as dump_f:
                json.dump(token_dict_sub, dump_f)


            new_G = ig.Graph()

            new_G.add_vertices(len(nodes))
            new_G.vs["name"] = nodes

            edges = []
            weights = []
            pbar = tqdm.tqdm(total=len(nodes))
            for start_node in nodes:
                pbar.update()
                start_vs = new_G.vs.find(name=start_node)
                try:
                    for end_node in graph_dict[start_node]:
                        end_vs = new_G.vs.find(name=end_node)
                        edges.append([start_vs, end_vs])
                        weight = weights_dict[str('{}_{}'.format(start_node, end_node))]
                        weights.append(weight)
                except Exception as e:
                    pass
            pbar.close()

            new_G.add_edges(edges)
            new_G.es["weight"] = weights

            if not os.path.exists(str(Path(args.nav_graph_path))):
                os.makedirs(str(Path(args.nav_graph_path)), exist_ok=True)
            new_G.write_pickle(str(Path(args.nav_graph_path) / 'nav_graph_dict_{}.pkl'.format(scene_id)))

            graphs[int(scene_id)] = new_G

        return graphs, token_dicts

    def _LoadNXGraphs(self, scene_ids=SCENE_IDS):
        graphs = {}
        token_dicts = {}

        for scene_id in scene_ids:
            print(
                "{}\tLoadNXGraphs of Scene {}".format(
                    str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                    scene_id
                )
            )

            G = ig.Graph.Read_Pickle(str(Path(self.nav_graph_path) / 'nav_graph_dict_{}.pkl'.format(scene_id)))
            graphs[int(scene_id)] = G

            file_name_dict = Path(self.token_dict_path) / 'TokenDict_{}.json'.format(scene_id)
            with open(file_name_dict, 'r', encoding='utf-8') as file:
                token_dict = json.load(file)
            token_dicts[int(scene_id)] = token_dict

        return graphs, token_dicts

    def get_shortest_paths(self, source: str, target: str, scene_id: int):
        assert scene_id in self.load_scenes, 'wrong scene_id of get_shortest_paths'
        graph = self.graphs[int(scene_id)]

        try:
            source_vs = graph.vs.find(name=str(source))
            target_vs = graph.vs.find(name=str(target))
        except Exception as e:
            print('wrong token of get_shortest_paths: {}'.format(e))
            raise Exception(e)

        shortest_paths = graph.get_shortest_paths(
            source_vs,
            to=target_vs,
            weights=graph.es["weight"],
            output="vpath",
        )
        if shortest_paths is None or len(shortest_paths) <= 0:
            print('get_shortest_paths error, could not found path')
            return []

        return shortest_paths[0]

    def get_vs_token(self, vs_index, scene_id: int):
        assert scene_id in self.load_scenes, 'wrong scene_id of get_vs_token'

        try:
            graph = self.graphs[int(scene_id)]

            vs_token = str(graph.vs[vs_index]['name'])
        except Exception as e:
            print('wrong vs_token of get_vs_token: {}'.format(e))
            raise Exception(e)

        return vs_token


if __name__ == '__main__':
    import argparse
    import random
    random.seed(1)

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--ori_nav_graph_path', type=str, default=str('./DATA/data/disceret/nav_graph_10/'))
    parser.add_argument('--ori_token_dict_path', type=str, default=str('./DATA/data/disceret/token_dict_10/'))
    parser.add_argument('--nav_graph_path', type=str, default=str('./DATA/data/disceret/processed/nav_graph_10/'))
    parser.add_argument('--token_dict_path', type=str, default=str('./DATA/data/disceret/processed/token_dict_10/'))
    parser.add_argument('--dagger_mode_load_scene', nargs='+', default=[4])
    args = parser.parse_args()

    # args.dagger_mode_load_scene = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26]
    sensor = ShortestPathSensor(args.nav_graph_path, args.token_dict_path, args.dagger_mode_load_scene)

    nodes = list(sensor.token_dicts[int(args.dagger_mode_load_scene[0])])

    time_start = time.time()
    for i in range(16):
        source = random.choice(nodes)
        target = random.choice(nodes)
        shortest_paths = sensor.get_shortest_paths(source, target, int(args.dagger_mode_load_scene[0]))
    time_end = time.time()
    print('time: {}'.format(time_end - time_start))

