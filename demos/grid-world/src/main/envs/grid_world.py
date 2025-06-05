from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from typing import Any, List, Tuple

import src.main.configs.global_configs as configs
from src.main.utility.enum_types import Actions

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(
            self,
            render_mode=configs.RENDER_MODE,
            size=configs.GRID_SIZE,
            is_use_fixed_start_and_goal=configs.IS_USE_FIXED_START_AND_PIT_POSITIONS,
            n_pits=configs.N_GRID_PITS
    ):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self._n_pits = n_pits
        if is_use_fixed_start_and_goal:
            self._agent_start_position = np.array([0, 0])
            self._goal_position = np.array([size - 1, size - 1])
            self._pit_positions = [np.array([0, int(size/2)]) ,
                                   np.array([int(size/2) + 1, int(size / 2) + 1]),
                                   np.array([int(size - 1), int(size/2)])][:self._n_pits]

        self._agent_location = None
        self._target_location = None

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self._agent_start_position

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._target_location = self._goal_position

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Steps through the environment
        :param action: Action taken by the agent
        """
        self._agent_location = self._clipState(self._agent_location, action)
        if self.render_mode == "human":
            self._reportCurrentGridPosition(self._agent_location)

        # An episode is done iff the agent has reached the target
        is_fallen_in_pit = self._getIsFallenIntoPitFlag()

        is_achieved_goal = np.array_equal(self._agent_location, self._target_location)
        if is_achieved_goal:
            reward = 10
            terminated = True
        elif is_fallen_in_pit:
            reward = -10
            terminated = True
        else:
            reward = 0
            terminated = False


        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _getIsFallenIntoPitFlag(self) -> bool | None:
        """
        Gets flag to signal the falling into any of the pits
        :return: Flag to signal the falling into any of the pits
        """
        match self._n_pits:
            case 1:
                np.array_equal(self._agent_location, self._pit_positions[0])
            case 2:
                return (np.array_equal(self._agent_location, self._pit_positions[0]) or
                        np.array_equal(self._agent_location, self._pit_positions[1]))
            case 3:
                return (np.array_equal(self._agent_location, self._pit_positions[0]) or
                       np.array_equal(self._agent_location, self._pit_positions[1]) or
                        np.array_equal(self._agent_location, self._pit_positions[2]))
            case _:
                raise Exception("Invalid number of pits specified!!")


    def render(self):
        # if self.render_mode == "rgb_array":
        return self._render_frame()

    def _clipState(
            self,
            current_location: np.ndarray,
            action: int
    ) -> np.ndarray:
        """
        Clips the state to be within the grid boundary
        :param current_location:Current location of the agent
        :param action: Action taken
        :return: new state after clipping
        """
        row, col = int(current_location[0]), int(current_location[1])

        match action:
            case 0:  # Up
                col = min(col + 1, self.size - 1)
            case 1:  # Left
                row = max(row - 1, 0)
            case 2:  # Down
                col = max(col - 1, 0)
            case 3:  # Right
                row = min(row + 1, self.size - 1)
            case _:
                raise Exception("Clipping failed error")
        new_location = np.array([row, col])
        return new_location

    def _render_frame(self):
        pygame.display.set_caption("Simple Gridworld")

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(configs.WHITE)
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target (green)
        pygame.draw.rect(
            canvas,
            configs.GREEN,
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent (blue)
        pygame.draw.circle(
            canvas,
            configs.BLUE,
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Define the pit (red)
        for idx in range(len(self._pit_positions)):
            pygame.draw.rect(
                canvas,
                "red",
                pygame.Rect(
                    pix_square_size * self._pit_positions[idx],
                    (pix_square_size, pix_square_size),
                ),
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

        return None

    def _reportCurrentGridPosition(self, current_position: np.ndarray):
        """
        Report current grid position
        :param current_position: Current grid position
        :return: None
        """
        font = pygame.font.Font(None, 40)
        current_position_str = f"(Current agent position: ({current_position[0]},{current_position[1]}))"
        pygame.display.set_caption(current_position_str)
        print(current_position_str)
        # text = font.render(current_position_str, True, configs.WHITE)
        # self.window.blit(text, [0, 0])



    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    @property
    def render_mode(self):
        """
        Getter for render mode.
        :return: render mode
        """
        return self._render_mode

    @render_mode.setter
    def render_mode(self, value: str):
        """
        Setter for render mode.
        :param value: render mode
        """
        self._render_mode = value
