from itertools import chain
import operator
import os
import random

# from black import prev_siblings_are  # This import appears to be incorrect
import sumo_gym.typing

import gymnasium as gym
import sumo_gym
from sumo_gym.utils.sumo_utils import SumoRender
from sumo_gym.utils.fmp_utils import *

import functools
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector

from statistics import mean


class VRPTW(object):
    def __init__(
        self,
        mode: str = None,
        net_xml_file_path: str = None,
        demand_xml_file_path: str = None,
        additional_xml_file_path: str = None,
        n_vertex: int = 0,
        n_area: int = 0,
        n_demand: int = 50,  # 50개 배송지
        n_edge: int = 0,
        n_vehicle: int = 0,
        n_delivery_truck: int = 5,  # 5대 트럭
        hub_location: int = 0,  # 허브 위치
        vertices: sumo_gym.typing.VerticesType = None,
        demands: sumo_gym.utils.fmp_utils.Demand = None,
        edges: sumo_gym.typing.EdgeType = None,
        delivery_trucks: sumo_gym.utils.fmp_utils.DeliveryTruck = None,
        departures: sumo_gym.typing.DeparturesType = None,
        time_manager: sumo_gym.utils.fmp_utils.TimeManager = None,
    ):
        if mode is None:
            raise Exception("Need a mode to identify")
        elif mode == "sumo_config":
            self.__sumo_config_init(
                net_xml_file_path, demand_xml_file_path, additional_xml_file_path
            )
        elif mode == "numerical":
            self.__numerical_init(
                n_vertex,
                n_area,
                n_demand,
                n_edge,
                n_vehicle,
                n_delivery_truck,
                hub_location,
                vertices,
                demands,
                edges,
                delivery_trucks,
                departures,
                time_manager,
            )
        else:
            raise Exception("Need a valid mode")

        if not self._is_valid():
            raise ValueError("FMP setting is not valid")

    def __numerical_init(
        self,
        n_vertex: int = 0,
        n_area: int = 0,
        n_demand: int = 0,
        n_edge: int = 0,
        n_vehicle: int = 0,
        n_delivery_truck: int = 5,
        hub_location: int = 0,
        vertices: sumo_gym.typing.VerticesType = None,
        demands: sumo_gym.utils.fmp_utils.Demand = None,
        edges: sumo_gym.typing.EdgeType = None,
        delivery_trucks: sumo_gym.utils.fmp_utils.DeliveryTruck = None,
        departures: sumo_gym.typing.DeparturesType = None,
        time_manager: sumo_gym.utils.fmp_utils.TimeManager = None,
    ):
        # number
        self.n_vertex = n_vertex
        self.n_area = n_area
        self.n_demand = n_demand
        self.n_edge = n_edge
        self.n_vehicle = n_vehicle
        self.n_delivery_truck = n_delivery_truck
        self.hub_location = hub_location

        # network
        self.vertices = sumo_gym.utils.fmp_utils.cluster_as_area(vertices, n_area)
        self.vertex_idx_area_mapping = {i: v.area for i, v in enumerate(self.vertices)}
        self.demands = demands
        self.edges = edges

        # vehicles
        self.delivery_trucks = delivery_trucks
        self.truck_name_idx_mapping = {
            f"truck_{i}": i for i in range(self.n_delivery_truck)
        }
        self.departures = departures
        self.time_manager = time_manager

        self.edge_dict = None

    def __sumo_config_init(
        self,
        net_xml_file_path: str = None,
        demand_xml_file_path: str = None,
        additional_xml_file_path: str = None,
    ):
        # VRPTW는 기존 SUMO 네트워크를 사용하지만 delivery trucks와 demands로 재구성
        (
            raw_vertices,  # [id (str), x_coord (float), y_coord (float)]
            raw_charging_stations,  # [id, (x_coord, y_coord), edge_id, charging speed] - 사용하지 않음
            raw_electric_vehicles,  # [id (str), maximum speed (float), maximumBatteryCapacity (float)] - 트럭으로 변환
            raw_edges,  # [id (str), from_vertex_id (str), to_vertex_id (str)]
            raw_departures,  # [vehicle_id, starting_edge_id] - 허브로 설정
            raw_demand,  # [junction_id, dest_vertex_id] - 배송지로 변환
        ) = sumo_gym.utils.xml_utils.decode_xml_fmp(
            net_xml_file_path, demand_xml_file_path, additional_xml_file_path
        )

        # `vertices` is a list of Vertex instances
        # `self.vertex_dict` is a mapping from vertex id in SUMO to idx in vertices
        from sumo_gym.utils.fmp_utils import convert_raw_vertices, convert_raw_edges, convert_raw_demand
        vertices, self.vertex_dict = convert_raw_vertices(raw_vertices)

        # `edges` is a list of Edge instances
        # `self.edge_dict` is a mapping from SUMO edge id to idx in `edges`
        # `self.edge_length_dict` is a dictionary mapping from SUMO edge id to edge length
        (
            edges,
            self.edge_dict,
            self.edge_length_dict,
        ) = convert_raw_edges(raw_edges, self.vertex_dict)

        # `demands` is a list of Demand instances - VRPTW 배송지로 변환
        demands = convert_raw_demand(raw_demand, self.vertex_dict)
        
        # VRPTW용으로 demands 수정 (시간창 추가)
        for i, demand in enumerate(demands):
            demand.earliest_time = i * 2  # 간단한 시간창 설정
            demand.latest_time = (i * 2) + 30
            demand.service_time = 5
            demand.is_completed = False

        # 5대 배송 트럭 생성
        from sumo_gym.utils.fmp_utils import DeliveryTruck
        self.n_delivery_truck = 5
        self.delivery_trucks = []
        
        for i in range(self.n_delivery_truck):
            truck = DeliveryTruck(
                id=i,
                speed=50,       # 속도
                indicator=0,    # 상태 지시자
                capacity=10,    # 트럭 용량
                location=0,     # 허브에서 시작
                current_time=0, # 현재 시간
                cargo_count=0,  # 적재량
                status=0        # 상태
            )
            self.delivery_trucks.append(truck)
        
        # 트럭 이름 매핑
        self.truck_name_idx_mapping = {
            f"truck_{i}": i for i in range(self.n_delivery_truck)
        }
        
        # 허브 위치 설정 (첫 번째 정점)
        self.hub_location = 0
        self.departures = [self.hub_location] * self.n_delivery_truck
        
        # 시간 관리자 생성
        from sumo_gym.utils.fmp_utils import TimeManager
        self.time_manager = TimeManager(max_episode_time=180)  # 3시간

        # VRPTW 변수 설정
        self.vertices = vertices
        self.edges = edges
        self.demands = demands

        self.n_demand = len(demands)
        self.n_vertex = len(self.vertices)
        self.n_edge = len(self.edges)
        self.n_vehicle = self.n_delivery_truck

    def _is_valid(self):
        if (
            not self.n_vertex
            or not self.n_demand
            or not self.n_edge
            or not self.n_vehicle
            or self.vertices is None
            or self.delivery_trucks is None
            or self.demands is None
            or self.edges is None
            or self.departures is None
        ):
            return False
        if len(self.vertices) != self.n_vertex:
            return False
        if len(self.edges) != self.n_edge:
            return False
        if len(self.delivery_trucks) != self.n_delivery_truck:
            return False
        if len(self.demands) != self.n_demand:
            return False
        # 허브 위치가 유효한지 확인
        if self.hub_location < 0 or self.hub_location >= self.n_vertex:
            return False
        return True


class VRPTWActionSpace(gym.spaces.Discrete):
    def __init__(self, n):
        self.n = int(n)
        super(VRPTWActionSpace, self).__init__(n)

    def sample(self) -> int:
        # 0: 허브 복귀, 1-50: 배송지, 51: 대기
        return random.randint(0, self.n - 1)


class FMPActionSpace(gym.spaces.Discrete):
    def __init__(self, n):
        self.n = int(n)
        super(FMPActionSpace, self).__init__(n)

    def sample(self) -> int:
        p_to_respond = random.uniform(0.3, 0.6)
        p_to_charge = 1.0 - p_to_respond
        return random.choices([0, 1, 2], [p_to_respond, p_to_charge, 0.0])[0]


class FMPLowerDemandActionSpace(gym.spaces.Discrete):
    def __init__(self, n):
        self.n_demand = int(n)
        super(FMPLowerDemandActionSpace, self).__init__(n)

    def sample(self) -> int:
        return random.randint(0, self.n_demand - 1)


class FMPLowerCSActionSpace(gym.spaces.Discrete):
    def __init__(self, n):
        self.n_cs = int(n)
        super(FMPLowerCSActionSpace, self).__init__(n)

    def sample(self) -> int:
        return random.randint(0, self.n_cs - 1)


class VRPTWEnv(AECEnv):
    metadata = {"render.modes": ["human"]}
    vrptw = property(operator.attrgetter("_vrptw"))

    def __init__(self, **kwargs):
        """
        Initialize FMPEnv.
        1. Setup render variables
        2. Setup FMP variables
        3. Setup Petting-Zoo environment variables (same as reset)
        """

        # setup render variables
        if "mode" not in kwargs:
            raise Exception("Need a mode to identify")

        elif kwargs["mode"] == "sumo_config":
            # render_env 설정 먼저 확인
            if "render_env" in kwargs:
                self.render_env = kwargs["render_env"]
                del kwargs["render_env"]
            else:
                self.render_env = False
            
            # render_env가 True일 때만 SUMO_GUI_PATH 필수
            if self.render_env:
                if "SUMO_GUI_PATH" in os.environ:
                    self.sumo_gui_path = os.environ["SUMO_GUI_PATH"]
                else:
                    raise Exception("Need 'SUMO_GUI_PATH' in the local environment when render_env=True")
            else:
                self.sumo_gui_path = None

            if "sumo_config_path" in kwargs:
                self.sumo_config_path = kwargs["sumo_config_path"]
                del kwargs["sumo_config_path"]
            else:
                raise Exception("Need 'sumo_config_path' argument to initialize")

        elif kwargs["mode"] == "numerical":
            if "render_env" in kwargs:
                raise Exception("Only support render for 'sumo_config' mode")

        else:
            raise Exception("Need a valid mode")

        if "verbose" in kwargs:
            self.verbose = kwargs["verbose"]
            del kwargs["verbose"]
        else:
            self.verbose = False

        # setup VRPTW variables
        self._vrptw = VRPTW(**kwargs)
        self.sumo = (
            SumoRender(
                self.sumo_gui_path,
                self.sumo_config_path,
                self.vrptw.edge_dict,
                self.vrptw.edge_length_dict,
                self.vrptw.truck_name_idx_mapping,
                self.vrptw.edges,
                self.vrptw.vertices,
                self.vrptw.vertex_dict,
                self.vrptw.n_delivery_truck,
            )
            if hasattr(self, "render_env") and self.render_env is True
            else None
        )
        self.travel_info = {i: None for i in range(self.vrptw.n_delivery_truck)}

        # setup Petting-Zoo environment variables
        self.possible_agents = [f"truck_{i}" for i in range(5)]  # 5대 트럭
        self.agent_name_idx_mapping = {f"truck_{i}": i for i in range(5)}

        self._action_spaces = {
            agent: VRPTWActionSpace(52) for agent in self.possible_agents  # 0:허브, 1-50:배송지, 51:대기
        }
        self._observation_spaces = {
            agent: VRPTWActionSpace(52) for agent in self.possible_agents
        }

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.possible_agents}

        self.upper_rewards = {agent: 0.0 for agent in self.agents}
        self.lower_reward_demand = 0  # network specific, no agent info
        self.lower_reward_cs = 0
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

        self.dones = {agent: False for agent in self.agents}
        self.infos = Metrics()

        self.states = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}

        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return FMPActionSpace(3)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return FMPActionSpace(3)

    def observe(self, agent):
        return self.observations[agent]

    def action_space_lower_demand(self):
        return FMPLowerDemandActionSpace(self.vrptw.n_demand)

    def action_space_lower_cs(self):
        return FMPLowerCSActionSpace(1)  # VRPTW에는 허브만 있음

    def reset(self):
        """
        Reset Petting-Zoo environment variables.
        """
        for i, truck in enumerate(self.vrptw.delivery_trucks):
            self.vrptw.delivery_trucks[i].location = self.vrptw.hub_location
            self.vrptw.delivery_trucks[i].current_time = 0
            self.vrptw.delivery_trucks[i].cargo_count = 0
            self.vrptw.delivery_trucks[i].status = 0  # 0: 허브에서 대기
        
        # 모든 배송지를 미완료 상태로 초기화
        for i, demand in enumerate(self.vrptw.demands):
            self.vrptw.demands[i].is_completed = False

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.upper_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.lower_reward_demand = 0
        self.lower_reward_cs = 0
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}

        self.dones = {agent: False for agent in self.possible_agents}
        self.infos = Metrics()

        self.states = {
            agent: self._get_vrptw_state(i)
            for i, agent in enumerate(self.possible_agents)
        }
        self.observations = {agent: self.states[agent] for agent in self.agents}

        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.travel_info = {i: None for i in range(self.vrptw.n_delivery_truck)}

        return self.observations

    def _get_vrptw_state(self, truck_idx):
        """VRPTW용 상태 벡터 생성 (고정 길이)"""
        truck = self.vrptw.delivery_trucks[truck_idx]
        
        # 기본 상태 정보 (3차원)
        state = [
            truck.current_time / 180.0,  # 정규화된 현재 시간
            truck.location / 50.0,       # 정규화된 위치
            truck.cargo_count / truck.capacity,     # 정규화된 적재량
        ]
        
        # 고정된 상태 길이 보장을 위해 demands 수를 일정하게 유지
        max_demands = 50  # 최대 배송지 수로 고정
        
        # 배송 완료 상태 (50개로 패딩)
        delivery_status = []
        for i in range(max_demands):
            if i < len(self.vrptw.demands):
                delivery_status.append(1.0 if self.vrptw.demands[i].is_completed else 0.0)
            else:
                delivery_status.append(0.0)  # 패딩
        state.extend(delivery_status)
        
        # 시간창 정보 (50개 × 2 = 100차원으로 패딩)
        for i in range(max_demands):
            if i < len(self.vrptw.demands):
                demand = self.vrptw.demands[i]
                state.extend([
                    demand.earliest_time / 180.0,  # 정규화된 시작 시간
                    demand.latest_time / 180.0,    # 정규화된 종료 시간
                ])
            else:
                state.extend([0.0, 0.0])  # 패딩
        
        # 최종 상태 벡터 길이: 3 + 50 + 100 = 153차원으로 고정
        return state

    def _was_done_step(self, action):
        if action is not None:
            raise ValueError("when an agent is done, the only valid action is None")

        # removes done agent
        agent = self.agent_selection
        agent_idx = self.agent_name_idx_mapping[agent]
        assert self.dones[
            agent
        ], "an agent that was not done as attempted to be removed"
        del self.dones[agent]
        del self.upper_rewards[agent]
        del self._cumulative_rewards[agent]
        self.agents.remove(agent)

        # finds next done agent or loads next live agent (Stored in _skip_agent_selection)
        _dones_order = [agent for agent in self.agents if self.dones[agent]]
        if _dones_order:
            if getattr(self, "_skip_agent_selection", None) is None:
                self._skip_agent_selection = self.agent_selection
            self.agent_selection = _dones_order[0]
        else:
            if getattr(self, "_skip_agent_selection", None) is not None:
                self.agent_selection = self._skip_agent_selection
            self._skip_agent_selection = None
        self._clear_rewards()

        if self.fmp.electric_vehicles[agent_idx].status > 2 * self.fmp.n_demand:
            if (
                self.fmp.electric_vehicles[agent_idx].status
                > 2 * self.fmp.n_demand + self.fmp.n_charging_station
            ):
                cs_idx = (
                    self.fmp.electric_vehicles[agent_idx].status
                    - 2 * self.fmp.n_demand
                    - self.fmp.n_charging_station
                    - 1
                )
            else:
                cs_idx = (
                    self.fmp.electric_vehicles[agent_idx].status
                    - 2 * self.fmp.n_demand
                    - 1
                )

            if agent in self.fmp.charging_stations[cs_idx].charging_vehicle:
                self.fmp.charging_stations[cs_idx].charging_vehicle.remove(agent)

    def step(self, action):
        """
        Step takes an action for the current agent.
        VRPTW uses single-level actions: 0=hub, 1-50=delivery locations, 51=wait
        """

        # VRPTW는 단일 액션만 사용
        vrptw_action = action

        agent = self.agent_selection
        agent_idx = self.agent_name_idx_mapping[agent]

        if self.dones[agent]:
            self._was_done_step(None)
            self.observations[agent] = None
            self.agent_selection = (
                self._agent_selector.next()
                if self._agent_selector.agent_order
                else None
            )
            return self.observations, self.upper_rewards, self.dones, self.infos

        # VRPTW 액션 처리
        self.upper_rewards[agent] = 0
        truck = self.vrptw.delivery_trucks[agent_idx]
        
        if vrptw_action == 0:  # 허브 복귀
            self._handle_hub_action(agent_idx)
        elif 1 <= vrptw_action <= 50:  # 배송지 이동
            delivery_idx = vrptw_action - 1
            self._handle_delivery_action(agent_idx, delivery_idx)
        elif vrptw_action == 51:  # 대기
            self._handle_wait_action(agent_idx)
        
        # 상태 업데이트
        self.states[agent] = self._get_vrptw_state(agent_idx)
        self.observations[agent] = self.states[agent]
        
        # 완료 조건 체크
        all_delivered = all(demand.is_completed for demand in self.vrptw.demands)
        time_exceeded = truck.current_time > 180
        self.dones[agent] = all_delivered or time_exceeded

        self._cumulative_rewards[agent] += self.upper_rewards[agent]

        if self.verbose:
            print(f"Agent {agent}: Action={vrptw_action}, Reward={self.upper_rewards[agent]}")

        # 다음 에이전트로
        self.agent_selection = self._agent_selector.next()

        return self.observations, self.upper_rewards, self.dones, self.infos

    def _handle_hub_action(self, truck_idx):
        """허브 복귀 액션 처리"""
        truck = self.vrptw.delivery_trucks[truck_idx]
        truck.location = self.vrptw.hub_location
        truck.cargo_count = min(truck.capacity, len([d for d in self.vrptw.demands if not d.is_completed]))
        truck.current_time += 5  # 허브에서 적재 시간
        self.upper_rewards[f"truck_{truck_idx}"] = 10  # 허브 복귀 보상

    def _handle_delivery_action(self, truck_idx, delivery_idx):
        """배송 액션 처리"""
        truck = self.vrptw.delivery_trucks[truck_idx]
        
        if delivery_idx < len(self.vrptw.demands) and not self.vrptw.demands[delivery_idx].is_completed:
            demand = self.vrptw.demands[delivery_idx]
            truck.location = demand.destination
            truck.current_time += 10  # 배송 시간
            truck.cargo_count = max(0, truck.cargo_count - 1)
            
            # 배송 완료 표시
            self.vrptw.demands[delivery_idx].is_completed = True
            
            # 보상 계산
            if demand.earliest_time <= truck.current_time <= demand.latest_time:
                self.upper_rewards[f"truck_{truck_idx}"] = 50  # 시간창 내 배송 보상
            else:
                self.upper_rewards[f"truck_{truck_idx}"] = -20  # 시간창 위반 페널티
        else:
            self.upper_rewards[f"truck_{truck_idx}"] = -10  # 잘못된 배송지 선택 페널티

    def _handle_wait_action(self, truck_idx):
        """대기 액션 처리"""
        truck = self.vrptw.delivery_trucks[truck_idx]
        truck.current_time += 1  # 1분 대기
        self.upper_rewards[f"truck_{truck_idx}"] = -1  # 대기 페널티

    def _state_move(self):
        """
        Deal with moving state:
        1. Move the state
        2. Update observation (force to change observations' action) and reward, accordingly
        """
        agent = self.agent_selection
        agent_idx = self.agent_name_idx_mapping[agent]

        # if responding
        if 0 < self.fmp.electric_vehicles[agent_idx].status <= 2 * self.fmp.n_demand:
            if self.fmp.electric_vehicles[agent_idx].status > self.fmp.n_demand:
                dmd_idx = (
                    self.fmp.electric_vehicles[agent_idx].status - self.fmp.n_demand - 1
                )
                dest_loc = self.fmp.demands[dmd_idx].destination
                if self.verbose:
                    print("Move: ", agent, " is in responding demand ", dmd_idx)

                self.fmp.electric_vehicles[
                    agent_idx
                ].location = one_step_to_destination(
                    self.fmp.vertices,
                    self.fmp.edges,
                    self.fmp.electric_vehicles[agent_idx].location,
                    dest_loc,
                )
                self.fmp.electric_vehicles[agent_idx].battery -= 1
                if self.fmp.electric_vehicles[agent_idx].location == dest_loc:
                    self.fmp.electric_vehicles[agent_idx].status = 0

            else:
                dmd_idx = self.fmp.electric_vehicles[agent_idx].status - 1
                dest_loc = self.fmp.demands[dmd_idx].departure
                if self.verbose:
                    print("Move: ", agent, " is to respond demand ", dmd_idx)

                self.fmp.electric_vehicles[
                    agent_idx
                ].location = one_step_to_destination(
                    self.fmp.vertices,
                    self.fmp.edges,
                    self.fmp.electric_vehicles[agent_idx].location,
                    dest_loc,
                )
                self.fmp.electric_vehicles[agent_idx].battery -= 1
                if self.fmp.electric_vehicles[agent_idx].location == dest_loc:
                    self.fmp.electric_vehicles[agent_idx].status += self.fmp.n_demand

        # if charging
        elif self.fmp.electric_vehicles[agent_idx].status > 2 * self.fmp.n_demand:
            if (
                self.fmp.electric_vehicles[agent_idx].status
                > 2 * self.fmp.n_demand + self.fmp.n_charging_station
            ):
                cs_idx = (
                    self.fmp.electric_vehicles[agent_idx].status
                    - 2 * self.fmp.n_demand
                    - self.fmp.n_charging_station
                    - 1
                )
                if self.verbose:
                    print(
                        "Move: ", agent, " is in charging at ", cs_idx,
                        " with position: ", self.fmp.charging_stations[cs_idx].charging_vehicle.index(agent), 
                        ". Total vehicles in charging station: ", self.fmp.charging_stations[cs_idx].charging_vehicle
                    )

                cs_edge = next(self.fmp.edges[i] for i in range(len(self.fmp.edges)) if self.fmp.edges[i].end == self.fmp.charging_stations[cs_idx].location)
                cs_edge_id = next(edge_id for edge_id, edge_idx in self.fmp.edge_dict.items() if self.fmp.edges[edge_idx] == cs_edge)
                cs_lane_position = self.fmp.edge_length_dict[cs_edge_id]
                print("CHECKKK: ", cs_lane_position)

                vehicle_stopped = (
                    self.sumo.retrieve_stop_status()[agent_idx] if self.sumo else None
                )
                
                if (
                    self.fmp.charging_stations[cs_idx].charging_vehicle.index(agent)
                    < self.fmp.charging_stations[cs_idx].n_slot
                    or
                    (vehicle_stopped is not None and abs(vehicle_stopped - cs_lane_position) < 5)
                ):
                    self.fmp.electric_vehicles[agent_idx].battery = min(
                        self.fmp.electric_vehicles[agent_idx].battery
                        + self.fmp.charging_stations[cs_idx].charging_speed,
                        self.fmp.electric_vehicles[agent_idx].capacity,
                    )
                else:
                    self.infos.charge_waiting_time += 1

                if (
                    self.fmp.electric_vehicles[agent_idx].battery
                    >= self.fmp.electric_vehicles[
                        self.agent_name_idx_mapping[agent]
                    ].capacity
                ):
                    self.fmp.electric_vehicles[agent_idx].status = 0
                    self.fmp.charging_stations[cs_idx].charging_vehicle.remove(agent)
                    print("Charging finished for vehicle; ", agent, self.fmp.charging_stations[cs_idx].charging_vehicle)
            else:
                cs_idx = (
                    self.fmp.electric_vehicles[agent_idx].status
                    - 2 * self.fmp.n_demand
                    - 1
                )
                dest_loc = self.fmp.charging_stations[cs_idx].location
                if self.verbose:
                    print("Move: ", agent, "is to go to charge at ", cs_idx)
                self.fmp.electric_vehicles[
                    agent_idx
                ].location = one_step_to_destination(
                    self.fmp.vertices,
                    self.fmp.edges,
                    self.fmp.electric_vehicles[agent_idx].location,
                    dest_loc,
                )
                self.fmp.electric_vehicles[agent_idx].battery -= 1

                if self.fmp.electric_vehicles[agent_idx].location == dest_loc:
                    self.fmp.electric_vehicles[
                        agent_idx
                    ].status += self.fmp.n_charging_station
                    self.fmp.charging_stations[cs_idx].charging_vehicle.append(agent)

        # not charging and not loading should not be moving
        else:
            raise ValueError("Agent that not responding or charging should not move")

        self.states[agent] = 3
        self.observations[agent] = 3
        self.upper_rewards[agent] = 0

    def _state_transition(self, upper_action, lower_action):
        """
        Transit the state of current agent according to the action and update observation:
        1. Update states (if "move" action for a state need to make decision, then no change)
        2. Update responded if responding a new demand
        3. Update observation and reward, accordingly
        """
        agent = self.agent_selection
        agent_idx = self.agent_name_idx_mapping[agent]
        is_valid = 1

        if upper_action == 2:
            if self.verbose:
                print("Trans: ", agent, "is taking moving action")

        # action to charge
        elif upper_action == 1:
            if self.verbose:
                print("Trans: ", agent, "is to go to charge at ", lower_action)

            self.fmp.electric_vehicles[agent_idx].location = one_step_to_destination(
                self.fmp.vertices,
                self.fmp.edges,
                self.fmp.electric_vehicles[agent_idx].location,
                self.fmp.charging_stations[lower_action].location,
            )
            self.fmp.electric_vehicles[agent_idx].battery -= 1
            self.fmp.electric_vehicles[agent_idx].status = (
                2 * self.fmp.n_demand + lower_action + 1
            )

        # action to load
        elif upper_action == 0:
            if self.verbose:
                print(
                    "Trans: ",
                    agent,
                    " is to respond demand ",
                    lower_action,
                )

            self.fmp.electric_vehicles[agent_idx].location = one_step_to_destination(
                self.fmp.vertices,
                self.fmp.edges,
                self.fmp.electric_vehicles[agent_idx].location,
                self.fmp.demands[lower_action].departure,
            )
            self.fmp.electric_vehicles[agent_idx].battery -= 1
            self.fmp.electric_vehicles[agent_idx].status = 1 + lower_action
            is_valid = (
                1
                if lower_action
                not in set(
                    chain.from_iterable(
                        [ev.responded for ev in self.fmp.electric_vehicles]
                    )
                )
                else 0
            )
            self.fmp.electric_vehicles[agent_idx].responded.append(lower_action)

        self.states[agent] = get_safe_indicator(
            self.fmp.vertices,
            self.fmp.edges,
            self.fmp.demands,
            self.fmp.charging_stations,
            self.fmp.electric_vehicles[agent_idx].location,
            self.fmp.electric_vehicles[agent_idx].battery,
        )
        self.observations[agent] = self.states[agent]
        self._calculate_upper_reward(agent, agent_idx, upper_action, lower_action)
        # self._calculate_lower_reward(self.fmp.electric_vehicles[agent_idx].location, is_valid, upper_action, lower_action)

    def _calculate_upper_reward(self, agent, agent_idx, upper_action, lower_action):
        self.upper_rewards[agent] = 0
        if self.states[agent] == 0:
            if upper_action == 0:
                self.upper_rewards[agent] = -100
            elif upper_action == 1:
                self.upper_rewards[agent] = 100
        elif self.states[agent] == 1:
            if upper_action == 0:
                next_state = get_safe_indicator(
                    self.fmp.vertices,
                    self.fmp.edges,
                    self.fmp.demands,
                    self.fmp.charging_stations,
                    self.fmp.demands[lower_action].destination,
                    self.fmp.electric_vehicles[agent_idx].battery
                    - get_dist_to_finish_demands(
                        self.fmp.vertices,
                        self.fmp.edges,
                        self.fmp.demands,
                        self.fmp.electric_vehicles[agent_idx].location,
                    )[lower_action],
                )
                self.upper_rewards[agent] = -100 if next_state == 0 else 50
            elif upper_action == 1:
                self.upper_rewards[agent] = 20
        elif self.states[agent] == 2:
            if upper_action == 0:
                self.upper_rewards[agent] = 50
            elif upper_action == 1:
                self.upper_rewards[agent] = -20

    def last(self, observe=True):
        agent = self.agent_selection
        agent_idx = self.agent_name_idx_mapping[agent]
        if agent is None:
            return None, 0, True, {}
        observation = self.observe(agent) if observe else None

        # VRPTW는 단일 레벨 액션이므로 상위 결과만 반환
        return (
            observation,
            self.upper_rewards[agent],
            self.dones[agent],
            self.infos,
        )

    def _generate_cs_vector(self):
        # VRPTW에서는 허브 정보만 반환 (충전소 없음)
        return [1]  # 허브는 항상 사용 가능

    def _generate_demand_vector(self):
        # 완료된 배송지 목록
        completed_demands = [1 if demand.is_completed else 0 for demand in self.vrptw.demands]
        return completed_demands

    def render(self, mode="human"):
        if self.sumo_gui_path is None:
            raise EnvironmentError("Need sumo-gui path to render")
        elif self.sumo is not None:
            self.sumo.render()

    def close(self):
        if hasattr(self, "sumo") and self.sumo is not None:
            self.sumo.close()
