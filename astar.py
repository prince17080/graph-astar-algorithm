import heapq
from typing import List

import numpy as np


class Graph:
    def __init__(
            self,
            point_ids
    ):
        """
        This class constructs a graph with point ids in a dictionary format.

        :param point_ids: Graph data in dictionary format as defined below:
            key: INTEGER: point id
            value: DICTIONARY containing at least below structure : {
                'point': np.array,
                'neighbours': list of point ids,
                'cell_ids': list of cell ids of the cells to which the point belongs to
            }
        """

        self.point_ids = point_ids

        self.points_float = []
        self.points_float_2d = []
        for (_, item) in self.point_ids.items():
            self.points_float.append(item['point'])
            self.points_float_2d.append(item['point'][:-1])

        self.points_float = np.asarray(self.points_float)
        self.points_float_2d = np.asarray(self.points_float_2d)
        # self.points_float = np.asarray([item['point'] for (_, item) in self.point_ids.items()])

    def get_point(
            self,
            point_id: int
    ):
        """
        Return the point coordinates of the given point id
        :param point_id: INTEGER
        :return: np.array
        """

        return self.point_ids[point_id]['point']

    def calculate_heuristic(
            self,
            point_id: int,
            sink_point_id: int,
            is_2d: bool = True,
    ):
        """
        Calculate heuristic value for the astar algorithm taking only
        the current point id and the destination/sink point id

        :param point_id: INTEGER: current point id
        :param sink_point_id: INTEGER: destination/sink point id
        :param is_2d: BOOLEAN: if the astar is applied in 2d plane, considering only x and y coordinates
        :return: FLOAT: heuristic value
        """

        point = self.get_point(point_id)
        sink_point = self.get_point(sink_point_id)
        if is_2d:
            g_value = np.linalg.norm(point[:-1] - sink_point[:-1])
        else:
            g_value = np.linalg.norm(point - sink_point)
        return g_value

    def find_astar_path_2d(
            self,
            source_point_id: int,
            sink_point_id: int,
            restrict_point_ids: List = None,
            is_2d: bool = True,
    ):
        """
        Finds the astar path between 2 points in a 3D Mesh

        :param source_point_id: INTEGER:
        :param sink_point_id: INTEGER:
        :param restrict_point_ids: LIST[INTEGERS]: List of integers to restrict
        the environment (open area to search in) for the astar algorithm
        :param is_2d: BOOLEAN: if the astar is applied in 2d plane, considering only x and y coordinates
        :return: LIST[INTEGERS]: path, containing the point ids, of the astar algorithm
        """

        if restrict_point_ids is None:
            restrict_point_ids = []

        source_value = self.calculate_heuristic(
            point_id=source_point_id,
            sink_point_id=sink_point_id,
            is_2d=is_2d
        )

        open_list = [(source_value, source_point_id)]
        closed_list = []

        parents = {source_point_id: -1}
        heapq.heapify(open_list)
        for _ in range(5000):
            (cur_value, cur_point_id) = heapq.heappop(open_list)
            closed_list.append((cur_value, cur_point_id))

            found_sink = False
            for neighbour_point_id in self.point_ids[cur_point_id]['neighbours']:
                if neighbour_point_id == sink_point_id:
                    parents[sink_point_id] = cur_point_id
                    found_sink = True
                    break

                if neighbour_point_id == source_point_id:
                    continue

                if neighbour_point_id in restrict_point_ids:
                    continue

                value = self.calculate_heuristic(
                    point_id=neighbour_point_id,
                    sink_point_id=sink_point_id,
                    is_2d=is_2d
                )

                open_search_index = None
                for i in range(len(open_list)):
                    if open_list[i][1] == neighbour_point_id:
                        open_search_index = i
                        break

                closed_search_index = None
                for i in range(len(closed_list)):
                    if closed_list[i][1] == neighbour_point_id:
                        closed_search_index = i
                        break

                if open_search_index is not None:
                    if value >= open_list[open_search_index][0]:
                        continue

                    open_list.pop(open_search_index)

                elif closed_search_index is not None:
                    if value >= closed_list[closed_search_index][0]:
                        continue

                    closed_list.pop(closed_search_index)

                heapq.heappush(open_list, (value, neighbour_point_id))
                parents[neighbour_point_id] = cur_point_id

            if found_sink:
                break

            if len(open_list) == 0:
                break

        path = []
        point_id = sink_point_id
        for _ in range(1000):
            path.append(point_id)
            if point_id not in parents:
                break

            point_id = parents[point_id]
            if point_id == -1:
                break

        return path
