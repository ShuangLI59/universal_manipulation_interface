import matplotlib.pyplot as plt
import pandas as pd


def polt_fn(timestamps, x, y, z, filetag):
    # Plotting X Coordinate
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, x, label='X Coordinate', alpha=0.8)
    plt.title('X Coordinate over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('X')
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(f'{filetag}_X_Coordinate_over_Time.png')


    # Plotting Y Coordinate
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, y, label='Y Coordinate', alpha=0.8, color='orange')
    plt.title('Y Coordinate over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(f'{filetag}_Y_Coordinate_over_Time.png')

    # Plotting Z Coordinate
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, z, label='Z Coordinate', alpha=0.8, color='green')
    plt.title('Z Coordinate over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Z')
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(f'{filetag}_Z_Coordinate_over_Time.png')


