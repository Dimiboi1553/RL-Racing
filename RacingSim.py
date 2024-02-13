import pygame
from pygame.locals import *
import math

# Initialize Pygame
pygame.init()

width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("2D Racing Sim")

clock = pygame.time.Clock()

speed = 4
rotation_speed = 4
Score = 0

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
BLUE = (0,0, 255)
GOLD = (218,165,32)
GRAY = (220,220,220)

Map = [[1,1,2,1,0,0,0,0,0,0,0,0,0],
       [1,0,1,1,0,0,0,0,0,0,2,1,2],
       [1,0,1,1,0,1,1,1,1,0,1,0,1],
       [1,0,2,1,0,1,1,1,1,0,2,0,1],
       [1,0,1,1,0,2,0,1,1,0,1,0,2],
       [1,0,2,1,2,1,0,2,0,0,1,0,1],
       [1,0,0,0,0,0,0,1,1,0,1,0,1],
       [1,0,0,0,0,0,0,2,1,1,2,0,1],
       [0,0,0,0,0,0,0,0,0,0,0,0,2],
       [3,1,1,2,1,1,1,1,2,1,1,1,1],]

cellSize = min(width // len(Map[0]), height // len(Map))
FrictionConst = 0.85

class Car:
    def __init__(self):

        self.original_image = pygame.image.load("Car.png").convert_alpha()
        self.playerX = 25
        self.playerY = 25
        self.angle = 90

        new_width = 20
        new_height = 30
        self.original_image = pygame.transform.scale(self.original_image, (new_width, new_height))
        
        self.Acceleration = 0.25
        self.player_car_width = 20
        self.player_car_height = 30

        self.velocity = pygame.Vector2(0, 0)

        self.player_car_rect = pygame.Rect(
            (self.playerX - self.player_car_width // 2, self.playerY, self.player_car_width, self.player_car_height)
        )

Player = Car()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    keys = pygame.key.get_pressed()

    if keys[K_w]:
        radians = math.radians(Player.angle - 90)
        Player.velocity.x += math.cos(radians) * Player.Acceleration
        Player.velocity.y += math.sin(radians) * Player.Acceleration

    if keys[K_a]:
        #Turn Left 
        Player.angle -= rotation_speed

    if keys[K_d]:
        #Turn Right 
        Player.angle += rotation_speed

    if keys[K_LSHIFT]:
        #Braking Mechanic
        FrictionConst -= 0.005

    if not keys[K_LSHIFT]:
        FrictionConst = 0.96   

    speed = Player.velocity.length()

    Player.velocity.x *= FrictionConst
    Player.velocity.y *= FrictionConst

    # Update position using velocity
    Player.playerX += Player.velocity.x
    Player.playerY += Player.velocity.y

    # Update rect position
    Player.player_car_rect.x = Player.playerX - Player.player_car_width // 2
    Player.player_car_rect.y = Player.playerY

    # Draw on the screen
    screen.fill(BLACK)

    # Draw the map
    for row_index, row in enumerate(Map):
        for col_index, cell in enumerate(row):
            if cell == 0:
                Cube = Rect(col_index * cellSize, row_index * cellSize, cellSize, cellSize)
                if Player.player_car_rect.colliderect(Cube):
                    print("Collision")
                    Player.velocity.x *= -1
                    Player.velocity.y *= -1
                    Score -= 10
                pygame.draw.rect(
                    screen, GRAY, (col_index * cellSize, row_index * cellSize, cellSize, cellSize)
                )  # Draw blocks

            if cell == 2:
                Cube = Rect(col_index * cellSize, row_index * cellSize, cellSize, cellSize)
                if Player.player_car_rect.colliderect(Cube):
                    print("Collision Checkpoint")
                    Score += 10
                    Map[row_index][col_index] = 1
                pygame.draw.rect(
                    screen, BLUE, (col_index * cellSize, row_index * cellSize, cellSize, cellSize)
                )  # Draw blocks

            if cell == 3:
                Cube = Rect(col_index * cellSize, row_index * cellSize, cellSize, cellSize)
                if Player.player_car_rect.colliderect(Cube):
                    print("Collision Checkpoint")
                    Score += 100
                pygame.draw.rect(
                    screen,GOLD, (col_index * cellSize, row_index * cellSize, cellSize, cellSize)
                )  # Draw blocks

    # Draw car
    rotated_surface = pygame.transform.rotate(Player.original_image, -Player.angle)
    rotated_rect = rotated_surface.get_rect(center=Player.player_car_rect.center)
    screen.blit(rotated_surface, rotated_rect.topleft)

    pygame.display.flip()

    clock.tick(60)

for i in range(10):
    env = RacingEnv()
    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        env.render()
    env.close()