import os
import sys
import math
import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def maze():
    # Maze
    maze = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 'r', 0, 0, 0, 0, 0, 0, 1],
        [1, 'h', 1, 1, 1, 1, 1, 0, 1],
        [1, 'h', 1, 1, 1, 1, 1, 0, 1],
        [1, 'h', 1, 1, 1, 1, 1, 0, 1],
        [1, 'h', 1, 1, 1, 1, 1, 0, 1],
        [1, 'h', 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    # We draw a black white image
    maze_image = 255 * np.ones(maze.shape + (3,))
    x, y = np.where(maze == '1')
    maze_image[x, y, :] = 0.
    x, y = np.where(maze == "r")
    maze_image[x, y, 0::2] = 0.
    x, y = np.where(maze == "h")
    maze_image[x, y, 1:] = 0.
    return maze_image


# Plot
LOWER_CONTEXT_BOUNDS = np.array([-9, -9, 0.5])
UPPER_CONTEXT_BOUNDS = np.array([9, 9, 5.])
FONT_SIZE = 8
TICK_SIZE = 6

fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
ax.imshow(maze(), extent=(LOWER_CONTEXT_BOUNDS[0], UPPER_CONTEXT_BOUNDS[0], LOWER_CONTEXT_BOUNDS[1],
                        UPPER_CONTEXT_BOUNDS[1]), origin="lower")
ax.set_xlim(LOWER_CONTEXT_BOUNDS[0], UPPER_CONTEXT_BOUNDS[0])
ax.set_ylim(LOWER_CONTEXT_BOUNDS[1], UPPER_CONTEXT_BOUNDS[1])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params('both', length=0, width=0, which='major')
ax.set_rasterized(True)
# no outer box
plt.axis('off')

plt.savefig("safety_maze.png", bbox_inches='tight', dpi=500)