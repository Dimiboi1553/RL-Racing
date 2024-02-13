import pygame
from pygame.locals import *
import math
import gym
from gym import spaces
import numpy as np
import time


#------------GLOBAL-CONSTS---------#
width, height = 800, 600
speed = 4
rotation_speed = 4

Map = [
    [0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
    [0,1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
    [0,1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2,0],
    [0,0, 0, 2, 2, 0, 2, 2, 2, 1, 0, 2, 0, 2,0],
    [0,1, 0, 2, 2, 0, 2, 2, 2, 1, 0, 2, 0, 2,0],
    [0,1, 0, 2, 2, 2, 2, 0, 2, 1, 0, 2, 0, 2,0],
    [0,1, 0, 2, 2, 2, 2, 0, 2, 1, 0, 2, 0, 2,0],
    [0,1, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 0, 2,0],
    [0,1, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 0, 2,0],
    [0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,0],
    [0,3, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2,0],
    [0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
    [0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
]

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GOLD = (218, 165, 32)
GRAY = (220, 220, 220)


cellSize = min(width // len(Map[0]), height // len(Map))
FrictionConst = 0.85
#--------------END-OF-GLOBAL-CONSTS-----#

class Car(gym.Env):
    def __init__(self):
        self.original_image = pygame.image.load("Car.png").convert_alpha()
        self.playerX = 75
        self.playerY = 75
        self.angle = 90

        new_width = 20
        new_height = 30
        self.original_image = pygame.transform.scale(self.original_image, (new_width, new_height))

        self.Acceleration = 1
        self.player_car_width = 20
        self.player_car_height = 30

        self.velocity = pygame.Vector2(0, 0)

        self.player_car_rect = pygame.Rect(
            (self.playerX, self.playerY, self.player_car_width, self.player_car_height)
        )

class RacingEnv(gym.Env):
    def __init__(self,render_mode=False):
        #Just general pygame stuff
        pygame.init()
        pygame.display.set_caption("AI learns to drive")
        self.screen = pygame.display.set_mode((width, height))   

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(43,), dtype=np.float32)

        #Total time steps if you want a hard limit on the steps
        self.max_steps = 1_000_000
        self.Player = Car()
        self.render_mode = render_mode
        self.ray_distances = []
        self.CheckpointsCoords = []
        self.done = False

        self.Map = [
            [0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
            [0,1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
            [0,1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2,0],
            [0,0, 0, 2, 1, 0, 1, 1, 1, 1, 0, 2, 0, 2,0],
            [0,1, 0, 2, 1, 0, 2, 2, 2, 1, 0, 2, 0, 2,0],
            [0,1, 0, 2, 1, 1, 2, 0, 2, 1, 0, 2, 0, 2,0],
            [0,1, 0, 2, 2, 2, 2, 0, 2, 1, 1, 2, 0, 2,0],
            [0,1, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 0, 2,0],
            [0,1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 2,0],
            [0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,0],
            [0,3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,0],
            [0,0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2,0],
            [0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
        ]

    def step(self, action):
        FrictionConst = 0.85
        reward = 0.0001

        dt = 0.5 #Time step for simulation 

        if action == 0:  # Move forward
            radians = math.radians(self.Player.angle - 90)
            self.Player.velocity.x += math.cos(radians) * self.Player.Acceleration
            self.Player.velocity.y += math.sin(radians) * self.Player.Acceleration
        elif action == 1:  # Turn left
            self.Player.angle -= rotation_speed * dt
            # if self.Player.angle <= 0:
            #     self.Player.angle += 360
        elif action == 2:  # Turn right
            self.Player.angle += rotation_speed * dt
            # if self.Player.angle >= 360:
            #     self.Player.angle -= 360
        elif action == 3:  # Brake
            FrictionConst -= 0.005


        #To maximise the speed of the AI its probably a good Idea to reward it based on speed but this implementation is faulty
        #speed = self.Player.velocity.length()
        #reward += speed
            
        #Apply friction
        self.Player.velocity.x *= FrictionConst
        self.Player.velocity.y *= FrictionConst
    
        #Update position using velocity
        self.Player.playerX += self.Player.velocity.x * dt
        self.Player.playerY += self.Player.velocity.y * dt

        #Update rect position
        self.Player.player_car_rect.x = self.Player.playerX - self.Player.player_car_width // 2
        self.Player.player_car_rect.y = self.Player.playerY
        
        #Check for collisions with walls
        for row_index, row in enumerate(self.Map):
            for col_index, cell in enumerate(row):
                cube_rect = pygame.Rect(col_index * cellSize, row_index * cellSize, cellSize, cellSize)
                if cell == 0:
                    if self.Player.player_car_rect.colliderect(cube_rect):
                        reward -= 10
                    
                        self.Player.velocity.x *= -1
                        self.Player.velocity.y *= -1

                        self.done = True
                        observation = self._get_observation()
                        return observation, reward, self.done, {}
                        
                if cell == 2:
                    #Collided with a checkpoint so reward the AI for doing well
                    if self.Player.player_car_rect.colliderect(cube_rect):
                        reward += 10
                        self.Map[row_index][col_index] = 1
                        self.CheckpointsCoords.append([row_index,col_index])

                if cell == 3:
                    # Collided with the End point so huge reward
                    #Endpoints!(Worth 100 points + what ever the function y=10000/self.current_timesteps encouraging it to get there in the least amount of time possible)
                    if self.Player.player_car_rect.colliderect(cube_rect):
                        reward += 1000
                        reward = 10000/self.current_step
                        self.done = True
                        observation = self._get_observation()
                        return observation, reward, self.done, {}


        rotated_surface = pygame.transform.rotate(self.Player.original_image, -self.Player.angle)
        rotated_rect = rotated_surface.get_rect(center=self.Player.player_car_rect.center)

        self.current_step += 1

        if self.render_mode == True:
            self.render()

        if self.current_step >= self.max_steps:
            self.done = True

        observation = self._get_observation()

        return observation, reward, self.done, {}

    def reset(self):
        self.Map = [
            [0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
            [0,1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
            [0,1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2,0],
            [0,0, 0, 2, 1, 0, 1, 1, 1, 1, 0, 2, 0, 2,0],
            [0,0, 0, 2, 1, 0, 2, 2, 2, 1, 0, 2, 0, 2,0],
            [0,0, 0, 2, 1, 1, 2, 0, 2, 1, 0, 2, 0, 2,0],
            [0,0, 0, 2, 2, 2, 2, 0, 2, 1, 0, 2, 0, 2,0],
            [0,0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 0, 2,0],
            [0,0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 2,0],
            [0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,0],
            [0,3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,0],
            [0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
        ]

        self.Player = Car()
        self.current_step = 0
        self.done = False
        return self._get_observation()

    def render(self,):
        self.screen.fill(BLACK)

        # Draw the map
        for row_index, row in enumerate(self.Map):
            for col_index, cell in enumerate(row):
                cube_rect = pygame.Rect(col_index * cellSize, row_index * cellSize, cellSize, cellSize)
                if cell == 0:
                    pygame.draw.rect(
                        self.screen, GRAY, (col_index * cellSize, row_index * cellSize, cellSize, cellSize)
                    )#Draw blocks

                if cell == 2:
                    pygame.draw.rect(
                        self.screen, BLUE, (col_index * cellSize, row_index * cellSize, cellSize, cellSize)
                    )  #Draw blocks
                if cell == 3:
                    pygame.draw.rect(
                        self.screen, GOLD, (col_index * cellSize, row_index * cellSize, cellSize, cellSize)
                    )#Draw Golden Blocks

        #Just to render the rays
        ray_distances_wall= self.calculate_ray_distances()[0]

        for i in range(len(ray_distances_wall)):
            angle_offset = (i - (len(ray_distances_wall) - 1) / 2) * math.pi / 16
            ray_angle = math.radians(self.Player.angle - 90) + angle_offset
            ray_start = (self.Player.playerX , self.Player.playerY  + 15)
            ray_end = (
                self.Player.playerX + ray_distances_wall[i][0] * math.cos(ray_angle),
                self.Player.playerY + 15 + ray_distances_wall[i][0] * math.sin(ray_angle),
            )

            pygame.draw.line(self.screen, RED, ray_start, ray_end, 2)

            square_x = int(ray_end[0])
            square_y = int(ray_end[1])

            pygame.draw.rect(self.screen, RED, (square_x - 5, square_y - 5, 10, 10))

        rotated_surface = pygame.transform.rotate(self.Player.original_image, -self.Player.angle)
        rotated_rect = rotated_surface.get_rect(center=self.Player.player_car_rect.center)
        self.screen.blit(rotated_surface, rotated_rect.topleft)

        pygame.display.update()

        if self.done:
            time.sleep(0.1)

    def close(self):
        pygame.quit()

    def calculate_ray_distances(self):
        #Ray Casting algorithm
        num_rays = 20
        max_ray_length = 1000

        ray_distances = []
        ray_distance_to_checkpoint = []

        for i in range(num_rays):
            angle_offset = (i - (num_rays - 1) / 2) * math.pi / 16
            ray_angle = math.radians(self.Player.angle - 90) + angle_offset

            ray_x = self.Player.playerX 
            ray_y = self.Player.playerY + 15

            dx = math.cos(ray_angle)
            dy = math.sin(ray_angle)

            step_size = 1

            for _ in range(int(max_ray_length / step_size)):
                ray_x += step_size * dx
                ray_y += step_size * dy

                map_x = int(ray_x / cellSize)
                map_y = int(ray_y / cellSize)

                if 0 <= map_x < len(Map[0]) and 0 <= map_y < len(Map):
                    if Map[map_y][map_x] == 2 or Map[map_y][map_x] == 3:
                        #Calculate the distance to checkpoint
                        distance = math.sqrt((ray_x - self.Player.playerX) ** 2 + (ray_y - self.Player.playerY) ** 2)
                        ray_distance_to_checkpoint.append([distance, ray_angle])
                    elif Map[map_y][map_x] == 0:  
                        break

            #Calculate the distance to wall
            distance = math.sqrt((ray_x - self.Player.playerX) ** 2 + (ray_y - self.Player.playerY) ** 2)
            ray_distances.append([distance, ray_angle])

        return ray_distances, ray_distance_to_checkpoint

    def _get_observation(self):

        def normalize_angle(angle):
            return angle/360 #(angle + math.pi) % (2 * math.pi) - math.pi
        
        #Init list for smallest Distance to checkpoint + angle + Player angle doesnt change in the min logic

        Min = [9999,0]
        #Calculate distances for rays and associated angle
        ray_distances_wall, RayDistanceToCheckPoint = self.calculate_ray_distances()

        #Normalise the distances and angles as it helps the AI alot. Without this the AI doesnt understand much
        Wall_ray = [distances / 1000 for distances,_ in ray_distances_wall]
        Wall_Angles = [normalize_angle(angles) for _,angles in ray_distances_wall]

        #Find the smallest Distance and the angle for this distance
        for i in range(len(RayDistanceToCheckPoint)):
            if RayDistanceToCheckPoint[i][0] < Min[0]:
                Min[0] = RayDistanceToCheckPoint[i][0] / 1000
                Min[1] = normalize_angle(RayDistanceToCheckPoint[i][1])

        #Build the observation for the AI
        #1) Raycasts distances 2)The angle of the raycast 3)Minimum distance to a checkpoint(raycast) 4)Angle for min ray cast 5) The current angle of the car
        observation = Wall_ray + Wall_Angles + Min + [self.Player.angle/360]
                                  
        return observation



# for i in range(10):
#     env = RacingEnv()
#     observation = env.reset()
#     done = False
#     Score = 0
#     while not done:
#         action = env.action_space.sample() #mlp take decision
#         observation, reward, done, _ = env.step(action)
#         env.render()
#         Score += reward
#     print(reward)


# env.close()