from sklearn.cluster import KMeans
import math

import sumo_gym.utils.network_utils as network_utils
import numpy as np
from bisect import bisect

NO_LOADING = -1
NO_CHARGING = -1
CHARGING_STATION_LENGTH = 5
IDLE_LOCATION = -1

K_MEANS_ITERATION = 10

NEAREST_CS = True
FURTHEST_CS = False


class Vertex(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.area = -1

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"vertex ({self.x}, {self.y}, {self.area})"


class Edge(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return (self.start, self.end) == (other.start, other.end)

    def __lt__(self, other):
        return (self.start, self.end) < (other.start, other.end)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"edge ({self.start}, {self.end})"


class Demand(object):
    def __init__(self, departure, destination, earliest_time=0, latest_time=180, delivery_id=None):
        self.departure = departure
        self.destination = destination
        self.earliest_time = earliest_time  # 시간창 시작 (분)
        self.latest_time = latest_time      # 시간창 종료 (분)
        self.delivery_id = delivery_id      # 배송지 고유 ID
        self.is_completed = False           # 배송 완료 여부

    def __eq__(self, other):
        return (self.departure, self.destination, self.delivery_id) == (
            other.departure,
            other.destination,
            other.delivery_id,
        )

    def __lt__(self, other):
        return (self.departure, self.destination, self.delivery_id) < (other.departure, other.destination, other.delivery_id)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"demand ({self.departure}, {self.destination}, tw:[{self.earliest_time}-{self.latest_time}], id:{self.delivery_id})"

class ElectricVehicles(object):
    def __init__(
        self,
        id,
        speed,
        indicator,
        capacity,
        location=None,
        battery=None,
        status=None,
        bonus=None,
        responded=None,
    ):
        self.id = id
        self.speed = speed
        self.indicator = indicator
        self.capacity = capacity
        self.thresholds = list(range(0, self.capacity, int(self.capacity / 5)))

        self.location = location
        self.battery = battery
        self.status = status
        self.bonus = bonus
        self.responded = responded

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"ElectricVehicles ({self.id}, {self.location}, {self.battery}, {self.responded})"

    def get_battery_level(self):
        return bisect(self.thresholds, self.battery) - 1


class DeliveryTruck(object):
    def __init__(
        self,
        id,
        speed,
        indicator,
        capacity,
        location=None,
        cargo_count=0,
        status=None,
        current_time=0,
        start_time=0,
    ):
        self.id = id
        self.speed = speed
        self.indicator = indicator
        self.capacity = capacity
        self.max_cargo = 5  # 고정 적재량
        
        self.location = location
        self.cargo_count = cargo_count
        self.status = status
        self.current_time = current_time
        self.start_time = start_time
        self.route_history = []
        self.delivered_items = []

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"DeliveryTruck ({self.id}, loc:{self.location}, cargo:{self.cargo_count}/{self.max_cargo}, time:{self.current_time})"

    def can_load_items(self, items_count):
        return self.cargo_count + items_count <= self.max_cargo
        
    def load_items(self, items_count):
        if self.can_load_items(items_count):
            self.cargo_count += items_count
            return True
        return False
        
    def deliver_item(self, delivery_id):
        if self.cargo_count > 0:
            self.cargo_count -= 1
            self.delivered_items.append(delivery_id)
            return True
        return False


class ChargingStation(object):
    def __init__(
        self, location, indicator, charging_speed, n_slot=None, charging_vehicle=None
    ):
        self.location = location
        self.indicator = indicator
        self.charging_speed = charging_speed
        self.n_slot = n_slot
        self.charging_vehicle = charging_vehicle

    def __eq__(self, other):
        return self.location == other.location

    def __lt__(self, other):
        return self.location < other.location

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"ChargingStation ({self.location}, {self.indicator}, {self.charging_speed})"


class Loading(object):
    def __init__(self, current=-1, target=-1):
        self.current = current
        self.target = target

    def __repr__(self):
        return f"(responding {self.current}, goto respond {self.target})"


class Charging(object):
    def __init__(self, current=-1, target=-1):
        self.current = current
        self.target = target

    def __repr__(self):
        return f"(charging {self.current}, go to charge {self.target})"


class GridAction(object):
    def __init__(self, state=None):
        self.is_loading = state.is_loading
        self.is_charging = state.is_charging
        self.location = state.location

    def __repr__(self):
        return f"({self.is_loading}, {self.is_charging}, location {self.location})"


class Metrics(object):
    def __init__(self):
        self.task_finish_time = 0
        self.total_battery_consume = 0
        self.charge_waiting_time = 0
        self.respond_failing_time = 0

    def __repr__(self):
        return f"(tft {self.task_finish_time}, tbc {self.total_battery_consume}, cwt {self.charge_waiting_time}, rft {self.respond_failing_time})"


def convert_raw_vertices(raw_vertices):
    """
    Each raw vertex is [id (str), x_coord (float), y_coord (float)]
    """
    vertices = []
    vertex_dict = {}  # vertex id in SUMO to idx in vertices
    for counter, v in enumerate(raw_vertices):
        vertices.append(Vertex(v[1], v[2]))
        vertex_dict[v[0]] = counter
    return vertices, vertex_dict


def convert_raw_edges(raw_edges, vertex_dict):
    """
    Each raw edge is
    [id (str), from_vertex_id (str), to_vertex_id (str), edge_length (float)]
    """
    edges = []
    edge_dict = {}  # sumo edge_id to idx in edges
    edge_length_dict = {}  # sumo edge_id to length
    for counter, e in enumerate(raw_edges):
        new_edge = Edge(vertex_dict[e[1]], vertex_dict[e[2]])
        edges.append(new_edge)
        edge_dict[e[0]] = counter
        edge_length_dict[e[0]] = e[3]
    return edges, edge_dict, edge_length_dict


def euclidean_distance(start_x, start_y, end_x, end_y):
    """
    Compute euclidean distance between (start_x, start_y)
    and (end_x, end_y)
    """
    return (((start_x - end_x) ** 2) + ((start_y - end_y) ** 2)) ** 0.5


def convert_raw_charging_stations(
    raw_charging_stations, vertices, edges, edge_dict, edge_length_dict
):
    """
    Each raw charging station is
    [id, (x_coord, y_coord), edge_id, charging speed]
    """

    charging_station_dict = {}  # idx in charging_stations to sumo id
    charging_stations = []

    vtx_counter = len(vertices)

    for counter, charging_station in enumerate(raw_charging_stations):

        charging_station_dict[counter] = charging_station[0]

        # create new vertex with charging station's location
        x_coord, y_coord = charging_station[1]
        new_vtx = Vertex(x_coord, y_coord)
        vertices.append(new_vtx)

        # create two new edges
        # first get the start and end vertex indices of the old edge
        edge_id = charging_station[2]
        old_edge_start_idx = edges[edge_dict[edge_id]].start
        old_edge_end_idx = edges[edge_dict[edge_id]].end

        edge_length_positive_edge_cs = charging_station[4]
        edge_length_postive_edge = edge_length_dict[edge_id]

        curr_edge_count = len(edge_dict)
        edges.append(Edge(old_edge_start_idx, vtx_counter))
        edge_dict["split1_%s" % edge_id] = curr_edge_count
        edge_length_dict["split1_%s" % edge_id] = edge_length_positive_edge_cs

        curr_edge_count += 1
        edges.append(Edge(vtx_counter, old_edge_start_idx))
        edge_dict["split1_-%s" % edge_id] = curr_edge_count
        edge_length_dict["split1_-%s" % edge_id] = edge_length_positive_edge_cs

        curr_edge_count += 1
        edges.append(Edge(vtx_counter, old_edge_end_idx))
        edge_dict["split2_%s" % edge_id] = curr_edge_count
        edge_length_dict["split2_%s" % edge_id] = (
            edge_length_postive_edge
            - edge_length_positive_edge_cs
            + CHARGING_STATION_LENGTH
        )

        curr_edge_count += 1
        edges.append(Edge(old_edge_end_idx, vtx_counter))
        edge_dict["split2_-%s" % edge_id] = curr_edge_count
        edge_length_dict["split2_-%s" % edge_id] = (
            edge_length_postive_edge
            - edge_length_positive_edge_cs
            + CHARGING_STATION_LENGTH
        )

        # instantiate new ChargingStation with location set to idx in `vertices`
        charging_stations.append(ChargingStation(vtx_counter, 220, charging_station[3]))

        vtx_counter += 1

    return charging_stations, charging_station_dict, edge_length_dict


def convert_raw_electric_vehicles(raw_electric_vehicles):
    """
    Each raw electric vehicle is
    [id (str), maximum speed (float), maximumBatteryCapacity (float)]
    """

    electric_vehicles = []
    ev_dict = {}  # ev sumo id to idx in electric_vehicles
    for counter, vehicle in enumerate(raw_electric_vehicles):
        electric_vehicles.append(
            ElectricVehicles(vehicle[0], vehicle[1], 250, vehicle[2])
        )

        ev_dict[vehicle[0]] = counter

    return electric_vehicles, ev_dict


def convert_raw_departures(raw_departures, ev_dict, edges, edge_dict, num_vehicles):
    """
    Each raw departure is [vehicle_id, starting_edge_id]
    """
    departures = np.zeros(num_vehicles)
    actual_departures = np.zeros(num_vehicles)
    for dpt in raw_departures:
        actual_departures[ev_dict[dpt[0]]] = edges[edge_dict[dpt[1]]].start
        departures[ev_dict[dpt[0]]] = edges[edge_dict[dpt[1]]].end
    return departures, actual_departures


def convert_raw_demand(raw_demand, vertex_dict):
    """
    Each raw demand is [junction_id, dest_vertex_id]
    """
    demand = []
    for d in raw_demand:
        demand.append(Demand(vertex_dict[d[0]], vertex_dict[d[1]]))
    return demand


def one_step_to_destination(vertices, edges, start_index, dest_index):
    if start_index == dest_index:
        return dest_index
    visited = [False] * len(vertices)
    bfs_queue = [dest_index]
    visited[dest_index] = True

    while bfs_queue:
        curr = bfs_queue.pop(0)
        adjacent_map = network_utils.get_adj_from_list(vertices, edges)

        for v in adjacent_map[curr]:
            if not visited[v] and v == start_index:
                return curr
            elif not visited[v]:
                bfs_queue.append(v)
                visited[v] = False


def dist_between(vertices, edges, start_index, dest_index):
    if start_index == dest_index:
        return 0
    visited = [False] * len(vertices)
    bfs_queue = [[start_index, 0]]
    visited[start_index] = True
    while bfs_queue:
        curr, curr_depth = bfs_queue.pop(0)
        adjacent_map = network_utils.get_adj_to_list(vertices, edges)

        for v in adjacent_map[curr]:
            if not visited[v] and v == dest_index:
                return curr_depth + 1
            elif not visited[v]:
                bfs_queue.append([v, curr_depth + 1])
                visited[v] = False


def get_hot_spot_weight(vertices, edges, demands, demand_start):
    adjacent_vertices = np.append(
        network_utils.get_adj_to_list(vertices, edges)[demand_start], demand_start
    )
    local_demands = len([d for d in demands if d.departure in adjacent_vertices])

    return local_demands / len(demands) * 100


# k as number of clusters, i.e., count of divided areas
def cluster_as_area(vertices, k):
    vertices_loc = [[v.x, v.y] for v in vertices]
    kmeans = KMeans(
        n_clusters=k,
        init=np.asarray(_generate_initial_cluster(vertices_loc, k)),
        random_state=0,
    ).fit(vertices_loc)
    for i, v in enumerate(vertices):
        v.area = kmeans.labels_[i]

    return vertices


# get the current safe indicator
def get_safe_indicator(vertices, edges, demands, charging_stations, location, battery):
    dist_to_furthest_cs = max(
        get_dist_to_charging_stations(vertices, edges, charging_stations, location)
    )
    dist_to_finish_demands = get_dist_to_finish_demands(
        vertices, edges, demands, location
    )
    if battery <= min(dist_to_finish_demands) + dist_to_furthest_cs:
        return 0
    elif battery <= max(dist_to_finish_demands) + dist_to_furthest_cs:
        return 1
    else:
        return 2


# get the dist to finish all demands from the current location
def get_dist_to_finish_demands(vertices, edges, demands, start_index):
    dist_of_demands = get_dist_of_demands(vertices, edges, demands)
    return [
        dist_of_demands[i] + dist_between(vertices, edges, start_index, d.departure)
        for i, d in enumerate(demands)
    ]


# get the travel dist of demand
def get_dist_of_demands(vertices, edges, demands):
    return [dist_between(vertices, edges, d.departure, d.destination) for d in demands]


# get the dist to all cs from the current location
def get_dist_to_charging_stations(vertices, edges, charging_stations, start_index):
    return [
        dist_between(vertices, edges, start_index, cs.location)
        for cs in charging_stations
    ]


# roughly divide the map into a root x root grid map as initialization
def _generate_initial_cluster(vertices_loc, k):
    initial_clusters = []
    root = int(math.sqrt(k))

    x_sorted = sorted(vertices_loc, key=lambda x: x[0])
    x_start = x_sorted[0][0]
    x_step = (x_sorted[-1][0] - x_sorted[0][0]) / (root + 1)

    y_sorted = sorted(vertices_loc, key=lambda x: x[1])
    y_start = y_sorted[0][1]
    y_step = (y_sorted[-1][1] - y_sorted[0][1]) / (root + 1)
    for i in range(root):
        for j in range(root):
            initial_clusters.append(
                [x_start + (i + 1) * x_step, y_start + (j + 1) * y_step]
            )

    return initial_clusters


# VRPTW 관련 함수들
class TimeManager:
    def __init__(self, max_episode_time=180):  # 3시간 = 180분
        self.max_episode_time = max_episode_time
        
    def is_within_time_window(self, current_time, demand):
        return demand.earliest_time <= current_time <= demand.latest_time
        
    def calculate_time_violation_penalty(self, arrival_time, demand):
        if arrival_time <= demand.latest_time:
            return 0  # 시간창 내 도착
        else:
            return (arrival_time - demand.latest_time) * 10  # 초과 시간 × 10
            
    def calculate_delivery_reward(self, arrival_time, demand):
        if arrival_time <= demand.latest_time:
            # 시간창 내 배송 완료 보상
            time_bonus = max(0, (demand.latest_time - arrival_time) / 30)  # 빨리 도착할수록 보너스
            return 100 + time_bonus
        else:
            # 시간창 초과 시 페널티
            return -50
            
    def calculate_travel_time(self, vertices, edges, start_loc, end_loc, speed=30):
        # speed: km/h, 거리는 km 단위로 가정, 결과는 분 단위
        distance = dist_between(vertices, edges, start_loc, end_loc)
        travel_time_hours = distance / speed
        return travel_time_hours * 60  # 분 단위로 변환


def generate_random_time_windows(num_deliveries, max_time=180):
    """50개 배송지에 대해 랜덤 시간창 생성"""
    import random
    time_windows = []
    
    # 6개 시간 구간 (30분씩)
    time_slots = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180)]
    
    for i in range(num_deliveries):
        # 랜덤하게 시간창 선택 (겹침 허용)
        slot = random.choice(time_slots)
        time_windows.append(slot)
    
    return time_windows


def check_delivery_feasibility(truck, demand, vertices, edges, time_manager):
    """트럭이 특정 배송지를 방문할 수 있는지 확인"""
    if truck.cargo_count == 0:
        return False, "No cargo to deliver"
        
    if demand.is_completed:
        return False, "Already delivered"
        
    travel_time = time_manager.calculate_travel_time(
        vertices, edges, truck.location, demand.destination
    )
    arrival_time = truck.current_time + travel_time
    
    if arrival_time > time_manager.max_episode_time:
        return False, "Episode time limit exceeded"
        
    return True, "Feasible"


def get_available_deliveries(truck, demands, vertices, edges, time_manager):
    """트럭이 현재 상태에서 갈 수 있는 배송지 목록 반환"""
    available = []
    
    for i, demand in enumerate(demands):
        feasible, reason = check_delivery_feasibility(truck, demand, vertices, edges, time_manager)
        if feasible:
            available.append(i)
            
    return available


def calculate_hub_return_time(truck_location, hub_location, vertices, edges, speed=30):
    """허브로 복귀하는데 필요한 시간 계산"""
    distance = dist_between(vertices, edges, truck_location, hub_location)
    travel_time_hours = distance / speed
    return travel_time_hours * 60  # 분 단위
