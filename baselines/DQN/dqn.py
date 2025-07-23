import random
import numpy as np

import torch


class ReplayBuffer(object):
    def __init__(self, max_len=10_000):
        self.max_len = max_len
        self.memory = list()

    def push(self, transition):
        if len(self.memory) >= self.max_len:
            self.memory.pop(0)

        self.memory.append(transition)

    def sample(self, batch_size):
        # numpy 배열 때문에 set 사용 불가, 직접 랜덤 샘플링
        available_memory = self.memory[:-1] if len(self.memory) > 1 else self.memory
        return (
            random.sample(available_memory, batch_size)
            if len(available_memory) >= batch_size
            else available_memory
        )

    def __repr__(self):
        return str(self.memory)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, item):
        return self.memory[item]


def run_target_update(q_principal, q_target):
    for v, v_ in zip(q_principal.model.parameters(), q_target.model.parameters()):
        v_.data.copy_(v.data)


def to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)


# TODO: extract common functions in the both upper and lower level networks


class QNetwork(object):
    def __init__(self, observation_size, n_action, lr):
        self.observation_size = observation_size
        self.n_action = n_action
        self.lr = lr
        # VRPTW를 위한 더 깊은 네트워크
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.observation_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.n_action),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def compute_q(self, states, actions):
        # numpy 배열이 섞여있는 경우 처리
        processed_states = []
        for s in states:
            if isinstance(s, np.ndarray):
                processed_states.append(s.tolist())
            else:
                processed_states.append(s)
        
        # 모든 상태의 길이가 같은지 확인하고 패딩
        if processed_states:
            max_len = max(len(s) for s in processed_states)
            for i, s in enumerate(processed_states):
                if len(s) < max_len:
                    # 짧은 상태는 0으로 패딩
                    processed_states[i] = s + [0.0] * (max_len - len(s))
        
        states = torch.FloatTensor(processed_states)
        q_preds = self.model(states)
        action_onehot = to_one_hot(actions, self.n_action)
        q_preds_selected = torch.sum(q_preds * action_onehot, axis=-1)
        return q_preds_selected

    def compute_max_q(self, states):
        # numpy 배열을 일반 리스트로 변환 후 텐서 생성
        if isinstance(states[0], np.ndarray):
            states = [s.tolist() for s in states]
        states = torch.FloatTensor(states)
        q_values = self.model(states).cpu().data.numpy()
        q_pred_greedy = np.max(q_values, 1)
        return q_pred_greedy

    def compute_argmax_q(self, state):
        # numpy 배열 처리
        if isinstance(state, np.ndarray):
            state = state.tolist()
        state = torch.FloatTensor([state])
        qvalue = self.model(state).cpu().data.numpy()
        greedy_action = np.argmax(qvalue.flatten())
        return greedy_action
    
    def compute_q_values(self, state):
        """단일 상태에 대한 모든 액션의 Q값 반환"""
        # numpy 배열 처리
        if isinstance(state, np.ndarray):
            state = state.tolist()
        state = torch.FloatTensor([state])
        with torch.no_grad():
            q_values = self.model(state).cpu().data.numpy().flatten()
        return q_values

    def train(self, states, actions, targets):
        # numpy 배열이 섞여있는 경우 처리
        processed_states = []
        for s in states:
            if isinstance(s, np.ndarray):
                processed_states.append(s.tolist())
            else:
                processed_states.append(s)
        
        states = torch.FloatTensor(processed_states)
        actions = torch.LongTensor(actions)
        targets = torch.FloatTensor(targets).squeeze()
        
        q_pred_selected = self.compute_q(states, actions)
        loss = torch.mean((q_pred_selected - targets) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().data.item()


class LowerQNetwork_Demand(object):
    def __init__(self, observation_size, n_action, lr):
        self.observation_size = observation_size + 1  # plus one for location area
        self.n_action = n_action
        self.lr = lr
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.observation_size, self.observation_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.observation_size, self.observation_size * 3),
            torch.nn.ReLU(),
            torch.nn.Linear(self.observation_size * 3, self.observation_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.observation_size * 2, self.n_action),
            torch.nn.ReLU(),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def compute_q(self, states, actions):
        states = torch.FloatTensor(states)
        q_preds = self.model(states)
        action_onehot = to_one_hot(actions, self.n_action)
        q_preds_selected = torch.sum(q_preds * action_onehot, axis=-1)
        return q_preds_selected

    def compute_max_q(self, states):
        states = torch.FloatTensor(states)
        q_values = self.model(states).cpu().data.numpy()
        q_pred_greedy = np.max(q_values, 1)
        return q_pred_greedy

    def compute_argmax_q(self, state):
        state = torch.FloatTensor(state)
        qvalue = self.model(state).cpu().data.numpy()
        greedy_action = np.argmax(qvalue.flatten())
        return greedy_action

    def train(self, states, actions, targets):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        targets = torch.FloatTensor(targets)
        q_pred_selected = self.compute_q(states, actions)
        loss = torch.mean((q_pred_selected - targets) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().data.item()


class LowerQNetwork_ChargingStation(object):
    def __init__(self, observation_size, n_action, lr):
        self.observation_size = observation_size + 1  # plus one for location area
        self.n_action = n_action
        self.lr = lr
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.observation_size, self.observation_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.observation_size, self.observation_size * 3),
            torch.nn.ReLU(),
            torch.nn.Linear(self.observation_size * 3, self.observation_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.observation_size * 2, self.n_action),
            torch.nn.ReLU(),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def compute_q(self, states, actions):
        states = torch.FloatTensor(states)
        q_preds = self.model(states)
        action_onehot = to_one_hot(actions, self.n_action)
        q_preds_selected = torch.sum(q_preds * action_onehot, axis=-1)
        return q_preds_selected

    def compute_max_q(self, states):
        states = torch.FloatTensor(states)
        q_values = self.model(states).cpu().data.numpy()
        q_pred_greedy = np.max(q_values, 1)
        return q_pred_greedy

    def compute_argmax_q(self, state):
        state = torch.FloatTensor(state)
        qvalue = self.model(state).cpu().data.numpy()
        greedy_action = np.argmax(qvalue.flatten())
        return greedy_action

    def train(self, states, actions, targets):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        targets = torch.FloatTensor(targets)
        q_pred_selected = self.compute_q(states, actions)
        loss = torch.mean((q_pred_selected - targets) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().data.item()
