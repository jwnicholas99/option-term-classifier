import networkx as nx
import pickle
import gzip
import matplotlib.pyplot as plt
from envs import MonteRAMState


class LocalGraphPartitioning:
    def __init__(
        self,
        trajectories=[("rgb_screens", "ram_states", "actions", "rewards", "dones")],
        discretize_factor=4,
    ):
        self.trajectories = trajectories
        self.discretize_factor = discretize_factor

    def get_run_transitions(self, state: MonteRAMState):
        """
        RUN REGIONS::

        self.UPPER_LEVEL: [(0x00, 0x05), (0x05, 0x33), (0x43, 0x4d), (0x4d, 0x55),
                                   (0x66, 0x95), (0x95, 0x99)],
        self.MIDDLE_LEVEL_1: [(0x09, 0x15), (0x15, 0x20), (0x38, 0x4d), (0x39, 0x4d),
                              (0x4d, 0x61), (0x4e, 0x61), (0x7b, 0x85), (0x85, 0x91)],
        # Running left on the treadmill changes x position by 2 every other step, so we need
        # to overlap the platforms so the left ends of the treadmill sections span 2px each.
        self.LOWER_LEVEL_1: [(0x15, 0x85)],
        """
        return []

    def get_jump_transitions(self, state: MonteRAMState):
        #needs to handle vertical and horizontal jumps
        #maybe only hard code all the "safe" jumps for now
        return []

    def get_climb_transitions(self, state: MonteRAMState):
        """
        CLIMB REGIONS::

        //can start climbing within 4 of actual ladder position

        # Ladder x_centers
        LEFT_LADDER = 0x14
        CENTER_LADDER = 0x4d
        RIGHT_LADDER = 0x85

        # Rope regions by screen: x_center, (y_min, y_max)
        ROPE_1 = 0x6d, (0xae, 0xd2)  # true y_max is 0xd4, but jumping above 0xd2 kills you

        regions.append((self.CENTER_LADDER, (self.MIDDLE_LEVEL_1, self.UPPER_LEVEL)))
        regions.append((self.LEFT_LADDER, (self.LOWER_LEVEL_1, self.MIDDLE_LEVEL_1)))
        regions.append((self.RIGHT_LADDER, (self.LOWER_LEVEL_1, self.MIDDLE_LEVEL_1)))
        """
        return []

    def construct_true_graph(self) -> nx.Graph:
        G = nx.Graph()

        num_positions = 256 // self.discretize_factor
        G.add_nodes_from(
            MonteRAMState(x, y, has_key, door_left_locked, door_right_locked, skull_x)
            for x in range(num_positions)
            for y in range(num_positions)
            for skull_x in range(num_positions)
            for has_key in [False, True]
            for door_left_locked in [False, True]
            for door_right_locked in [False, True]
        )

        #TODO add edges based on know transitions

        # Find x range and y positions for all platforms : extract from Cam's code
        # need to find jump-height
        #edges for:
        # - adjacent x along platforms and skull moves left or right
        # - adjacent y along ladders/rope and skull moves left or right
        # - on platforms or ladders/rope skull moves left or right
        # - above platforms within the jump height adjacent y states and skull moves left or right
        # - need to be careful around the key

        #maybe there is a better way to go this
        transitions = []
        for door_left_locked in [False, True]:
            for door_right_locked in [False, True]:
                for has_key in [False, True]:
                    for skull_x in range(num_positions):
                        for x in range(num_positions):
                            for y in range(num_positions):
                                state = MonteRAMState(x, y, has_key, door_left_locked, door_right_locked, skull_x)

                                transitions += self.get_run_transitions(state)
                                transitions += self.get_jump_transitions(state)
        G.add_edges_from(transitions)

        return G


class TrueGraph(nx.Graph):
    def __init__(self, discretize_factor=4):
        super(TrueGraph, self).__init__()

        self.discretize_factor = discretize_factor

    def discretize_state(self, state: MonteRAMState) -> MonteRAMState:
        x, y, has_key, door_left_locked, door_right_locked, skull_x = state
        x //= self.discretize_factor
        y //= self.discretize_factor
        skull_x //= self.discretize_factor
        return MonteRAMState(
            x, y, has_key, door_left_locked, door_right_locked, skull_x
        )

    def add_trajectory(self, trajectory):
        _, rams, _, _, dones = trajectory  # (rgb_screens, ram_states, actions, rewards, dones)
        self.add_nodes_from(
            self.discretize_state(ram)
            for ram, done in zip(rams, dones)
            if not done
        )
        self.add_edges_from(
            (self.discretize_state(ram), self.discretize_state(next_ram))
            for ram, next_ram, done in zip(rams, rams[1:], dones)
            if not done
        )

    def add_trajectories_from(self, trajectories):
        for trajectory in trajectories:
            self.add_trajectory(trajectory)

    def draw(self):
        fig, ax = plt.subplots()
        nx.draw_spectral(self, ax=ax, node_size=0.1, node_color=(1,0,0), edge_color=(0,0,0))
        ax.set_title("True Graph")
        plt.savefig("true_graph.png")


if __name__ == "__main__":
    true_graph = TrueGraph(16)

    with gzip.open("monte_rnd_full_trajectories.pkl.gz", "rb") as f:
        try:
            for i in range(1):
                trajectories = pickle.load(f)
                true_graph.add_trajectories_from(trajectories)
        except EOFError:
            pass

    true_graph.draw()
