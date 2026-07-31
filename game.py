import random
from enum import Enum
from collections import namedtuple
import numpy as np
import pygame

from config import (
    BLOCK_SIZE,
    WIDTH,
    HEIGHT,
    REWARD_EAT,
    REWARD_DIE,
    STEP_PENALTY,
    DIST_COEF_PER_CELL,
    CLOSE_BONUS_DIST,
    CLOSE_BONUS,
    STUCK_MULT,
    MAX_STEPS_WITHOUT_FOOD,
)

Point = namedtuple('Point', 'x y')

class Direction(Enum):
    RIGHT = 1
    LEFT  = 2
    UP    = 3
    DOWN  = 4

class SnakeGame:
    def __init__(
        self,
        w=WIDTH,
        h=HEIGHT,
        render=False,
        title='Snake AI',
        invincible: bool = False,
        render_fps: int = 90,
    ):
        self.w = w
        self.h = h
        self.render_enabled = render
        self.invincible = invincible
        self.render_fps = render_fps
        if self.render_enabled:
            pygame.display.set_caption(title)
            self.display = pygame.display.set_mode((self.w, self.h))
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.font = pygame.font.Font(None, 24)
        else:
            self.display = None
            self.clock = None
            self.font = None
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y),
        ]
        self.score = 0
        self.food = None

        self._last_dist_cells = None
        self._steps_since_food = 0

        self._place_food()
        return self.get_state()

    def _place_food(self):
        x = random.randrange(0, (self.w // BLOCK_SIZE)) * BLOCK_SIZE
        y = random.randrange(0, (self.h // BLOCK_SIZE)) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
            return

        hx, hy = self.head.x // BLOCK_SIZE, self.head.y // BLOCK_SIZE
        fx, fy = self.food.x // BLOCK_SIZE, self.food.y // BLOCK_SIZE
        self._last_dist_cells = abs(hx - fx) + abs(hy - fy)
        self._steps_since_food = 0

    def step(self, action: int):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if action == 0:
            new_dir = clock_wise[(idx - 1) % 4]
        elif action == 2:
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[idx]
        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction == Direction.LEFT: x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN: y += BLOCK_SIZE
        elif self.direction == Direction.UP: y -= BLOCK_SIZE
        new_head = Point(x, y)

        if self._is_collision(new_head):
            return REWARD_DIE, True, self.score

        self.snake.insert(0, new_head)
        self.head = new_head

        reward = STEP_PENALTY

        hx, hy = self.head.x // BLOCK_SIZE, self.head.y // BLOCK_SIZE
        fx, fy = self.food.x // BLOCK_SIZE, self.food.y // BLOCK_SIZE
        curr_dist_cells = abs(hx - fx) + abs(hy - fy)
        if self._last_dist_cells is not None:
            delta = self._last_dist_cells - curr_dist_cells
            reward += DIST_COEF_PER_CELL * float(delta)
        self._last_dist_cells = curr_dist_cells

        if curr_dist_cells <= CLOSE_BONUS_DIST:
            reward += CLOSE_BONUS

        self._steps_since_food += 1
        done = False

        if self.head == self.food:
            self.score += 1
            reward += REWARD_EAT
            self._place_food()
        else:
            self.snake.pop()

        if self._steps_since_food > MAX_STEPS_WITHOUT_FOOD:
            return reward - 5.0, True, self.score

        return reward, done, self.score

    def _is_collision(self, pt: Point = None) -> bool:
        if getattr(self, "invincible", False):
            return False
        if pt is None:
            pt = self.head
        if pt.x < 0 or pt.x >= self.w or pt.y < 0 or pt.y >= self.h:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _cast_ray(self, point: Point, direction: Point) -> np.ndarray:
        start_x, start_y = point.x, point.y
        dir_x, dir_y = direction.x, direction.y

        max_steps = (self.w + self.h) // BLOCK_SIZE

        current_x = start_x
        current_y = start_y

        val_wall = 0.0
        val_body = 0.0
        val_food = 0.0

        found_body = False
        found_food = False

        for step in range(1, max_steps + 1):
            current_x += dir_x
            current_y += dir_y

            if current_x < 0 or current_x >= self.w or current_y < 0 or current_y >= self.h:
                val_wall = 1.0 / step
                break

            if not found_body and Point(current_x, current_y) in self.snake:
                val_body = 1.0 / step
                found_body = True

            if not found_food and Point(current_x, current_y) == self.food:
                val_food = 1.0 / step
                found_food = True

        return np.array([val_wall, val_body, val_food], dtype=np.float32)

    def get_state(self):
        head = self.head

        directions = [
            Point(0, -BLOCK_SIZE),
            Point(BLOCK_SIZE, -BLOCK_SIZE),
            Point(BLOCK_SIZE, 0),
            Point(BLOCK_SIZE, BLOCK_SIZE),
            Point(0, BLOCK_SIZE),
            Point(-BLOCK_SIZE, BLOCK_SIZE),
            Point(-BLOCK_SIZE, 0),
            Point(-BLOCK_SIZE, -BLOCK_SIZE)
        ]

        state_list = []

        for d in directions:
            ray_info = self._cast_ray(head, d)
            state_list.extend(ray_info)

        dir_l = float(self.direction == Direction.LEFT)
        dir_r = float(self.direction == Direction.RIGHT)
        dir_u = float(self.direction == Direction.UP)
        dir_d = float(self.direction == Direction.DOWN)

        state_list.extend([dir_l, dir_r, dir_u, dir_d])

        return np.array(state_list, dtype=np.float32)

    def render(self, info_str: str = ""):
        if not self.render_enabled:
            return
        self.display.fill((0, 0, 0))
        for pt in self.snake:
            pygame.draw.rect(self.display, (255, 165, 0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, (255, 140, 0), pygame.Rect(pt.x + 4, pt.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8))
        pygame.draw.rect(self.display, (200, 0, 0), pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        if self.font:
            text = self.font.render(f"Score: {self.score}  {info_str}", True, (255, 255, 255))
            self.display.blit(text, [5, 5])
        pygame.display.flip()
