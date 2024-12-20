import matplotlib.pyplot as plt
import pandas as pd

def plot_fn(timestamps, x, filetag, y_min=-10, y_max=10, color='blue'):
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, x, label='X Coordinate', alpha=0.8, color=color)
    # plt.title('X Coordinate')
    plt.xlabel('Timestamp')
    plt.ylabel('X')
    plt.ylim(y_min, y_max)  # Set consistent y-axis limits
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{filetag}.png')

    # # Plotting Y Coordinate
    # plt.figure(figsize=(10, 6))
    # plt.plot(timestamps, y, label='Y Coordinate', alpha=0.8, color='orange')
    # plt.title('Y Coordinate over Time')
    # plt.xlabel('Timestamp')
    # plt.ylabel('Y')
    # plt.ylim(y_min, y_max)  # Set consistent y-axis limits
    # plt.grid(True)
    # plt.legend()
    # plt.savefig(f'{filetag}_Y_Coordinate_over_Time.png')

    # # Plotting Z Coordinate
    # plt.figure(figsize=(10, 6))
    # plt.plot(timestamps, z, label='Z Coordinate', alpha=0.8, color='green')
    # plt.title('Z Coordinate over Time')
    # plt.xlabel('Timestamp')
    # plt.ylabel('Z')
    # plt.ylim(y_min, y_max)  # Set consistent y-axis limits
    # plt.grid(True)
    # plt.legend()
    # plt.savefig(f'{filetag}_Z_Coordinate_over_Time.png')