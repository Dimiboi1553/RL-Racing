# RL-Driving

Overview:
This is a driving environent for deep leanrning algorithms(RL) to learn how to drive. 


## Pygame Environment
This Environment is made in Pygame so you can see the AI's actions as it is training.
The Track itself is mazelike starting very easy and becoming harder as it progresses requiring more precision from the AI, resulting in a unique test to see the capabilities of these algorithms

## Stable Baselines:
The RL algorithms are provided by Stable Baselines3,meaning that the algorithm is flexible as is can be changed easily and quickly.

## Pretrained AI:
I have also attached trained AIs(Which has been trained on 500k timesteps) to show what this environment is capable of producing.
The well trained model is V1_3(Is very consistant but almost never finishes the course)
The other stable AI I would consider using is V2_2 as it does finish the course but is inconsistant.

## Environment flexibility:
The map is customisable as it is a 2D array of 0: Walls(worth -10 points if collided with), 1: Road, 2:Checkpoints(worth 10 points each for training), 3:Golden Blocks(worth 100 points and ending the episode).
Meaning you can change the map to be any size and shape and the program will adjust for it automatically.

## New features!:
We've upgraded the observation toolkit for the AI with three new metrics, enhancing its perception of the envrionement. Initially, the system provided the AI with just two metrics:
1) the Raycast distances and 2) the Minimum Ray distance. Now, we've expanded the metrics to include:
- The original Raycast distances and Minimum Ray distance.
- The angle of each Raycast.
- The angle of the Minimum Ray distance.
- The current angle of the Car.
   
These metrics helps the AI train faster and better achieving better results with less training.
These new additions made the AI's improve by 5x!

## Next Features to be implemented:
Drifting!
 
 
