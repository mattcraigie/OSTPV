import torch
from torch.fft import fft2, ifft2, fftshift
import numpy as np
import matplotlib.pyplot as plt
import os


def make_triangle_grid(size, a, b, num_triangles):
    grid = np.zeros((size, size), dtype=int)

    for i in range(num_triangles):
        # Choose a random position for the first point
        x1, y1 = np.random.randint(0, size), np.random.randint(0, size)

        # Choose a random direction for the second point (represented by a unit vector)
        direction = np.random.randn(2)
        direction /= np.linalg.norm(direction)

        # Calculate the position of the second point based on the random direction and distance 'a'
        x2, y2 = x1 + int(a * direction[0]), y1 + int(a * direction[1])

        # Calculate the direction perpendicular to the second point's direction (rotated 90 degrees)
        perpendicular_direction = np.array([-direction[1], direction[0]])

        # Calculate the position of the third point based on the perpendicular direction and distance 'b'
        x3, y3 = x1 + int(b * perpendicular_direction[0]), y1 + int(b * perpendicular_direction[1])

        # Wrap the points around the grid boundaries if they go beyond
        x1 %= size
        y1 %= size
        x2 %= size
        y2 %= size
        x3 %= size
        y3 %= size

        # Add the points to the grid
        grid[x1, y1] += 1
        grid[x2, y2] += 1
        grid[x3, y3] += 1

    return grid


def make_2d_mocks(num_mocks, size, a, b, num_triangles, save_path=None):

    all_mocks = torch.zeros(num_mocks, size, size)

    print("Making {} mocks".format(num_mocks))
    for i in range(num_mocks):
        if i % 100 == 0:
            print(i)
        resulting_grid = make_triangle_grid(size, a, b, num_triangles)
        all_mocks[i] = torch.from_numpy(resulting_grid)

    if save_path is None:
        save_path = 'mocks_2d.pt'

    torch.save(all_mocks, save_path)


def random_unit_vector():
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)
    return vec


def get_random_orthog_vecs():

    # Align i with the random unit vector
    i = random_unit_vector()

    # Calculate j and k based on i
    j = np.cross(i, random_unit_vector())  # pick a random direction and cross off that to get the first perp vec
    j /= np.linalg.norm(j)

    k = np.cross(i, j)

    return i, j, k


def add_tetra_to_grid(size, a, b, c, num_tetras):
    # a, b and c are the sizes of the tetra legs
    grid = np.zeros((size, size, size), dtype=int)
    for i in range(num_tetras):
        # Choose a random position for the first point
        x1, y1, z1 = np.random.randint(0, size), np.random.randint(0, size), np.random.randint(0, size)
        point_1 = np.array([x1, y1, z1])

        direction_2, direction_3, direction_4 = get_random_orthog_vecs()

        # Calculate the position of the points
        point_2 = point_1 + (a * direction_2).astype(int)
        point_3 = point_1 + (b * direction_3).astype(int)
        point_4 = point_1 + (c * direction_4).astype(int)

        # Add the points to the grid, wrapping if they go over
        for p in [point_1, point_2, point_3, point_4]:
            p = p % size
            grid[p[0], p[1], p[2]] += 1

    return grid


def make_3d_mocks(num_mocks, size, a, b, c, num_tetras, save_path=None):
    all_mocks = np.zeros((num_mocks, size, size, size))

    print("Making {} mocks".format(num_mocks))
    for i in range(num_mocks):
        if i % 100 == 0:
            print(i)
        resulting_grid = add_tetra_to_grid(size, a, b, c, num_tetras)
        all_mocks[i] = resulting_grid

    if save_path is None:
        save_path = 'mocks_3d.npy'

    np.save(save_path, all_mocks)