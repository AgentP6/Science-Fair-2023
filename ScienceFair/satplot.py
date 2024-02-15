import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_satellite_orbits(num_satellites, altitude_km):
    """
    Plot simulated satellite orbits and actual positions around the Earth.
    :param num_satellites: Number of satellites (and orbits) to plot.
    :param altitude_km: Altitude above Earth's surface in kilometers for the orbits.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Earth as a sphere
    earth_radius_km = 6371  # Earth's radius in kilometers
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]  # Meshgrid for sphere
    x_earth = earth_radius_km * np.cos(u) * np.sin(v)
    y_earth = earth_radius_km * np.sin(u) * np.sin(v)
    z_earth = earth_radius_km * np.cos(v)
    ax.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.3)
    
    # Generate and plot orbits
    for _ in range(num_satellites):
        # Randomly generate longitude of ascending node (Ω) and inclination (i) for the orbit
        omega = np.random.uniform(0, 2 * np.pi)
        inclination = np.random.uniform(0, np.pi)
        
        # Generate points along the orbit
        theta = np.linspace(0, 2 * np.pi, 100)  # Parameter for orbit
        r = earth_radius_km + altitude_km  # Radius from Earth's center to orbit
        x_orbit = r * np.cos(theta) * np.cos(omega) - r * np.sin(theta) * np.cos(inclination) * np.sin(omega)
        y_orbit = r * np.cos(theta) * np.sin(omega) + r * np.sin(theta) * np.cos(inclination) * np.cos(omega)
        z_orbit = r * np.sin(theta) * np.sin(inclination)
        
        # Plot the orbit
        ax.plot(x_orbit, y_orbit, z_orbit, color='red', linewidth=0.5)
        
        # Mark the actual satellite position with a larger, black dot
        # Choose a random point along the orbit as the "actual" position
        actual_pos_index = np.random.randint(0, 100)
        ax.scatter(x_orbit[actual_pos_index], y_orbit[actual_pos_index], z_orbit[actual_pos_index], color='black', s=20)

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Simplified Satellite Orbits and Positions Around Earth')
    plt.show()

# Example usage
num_satellites = 100  # Number of satellites to visualize
altitude_km = 400  # Altitude of satellites in kilometers above Earth's surface
plot_satellite_orbits(num_satellites, altitude_km)
