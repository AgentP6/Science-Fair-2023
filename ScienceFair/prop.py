import pandas as pd
from datetime import datetime, timedelta
from sgp4.api import Satrec, WGS72, jday
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to convert NORAD data row to TLE format
def create_tle_from_norad(row):
    epoch = datetime.strptime(row['EPOCH'], "%Y-%m-%dT%H:%M:%S.%f")
    year = epoch.strftime("%y")
    day_of_year = epoch.timetuple().tm_yday + (epoch.hour / 24.0) + (epoch.minute / 1440.0) + (epoch.second / 86400.0)
    bstar = row['BSTAR']
    
    line1 = f"1 {row['NORAD_CAT_ID']:05d}U 00000A   {year}{day_of_year:012.8f}  .00000000  00000-0  {bstar:8.4e} 0  9999"
    line2 = f"2 {row['NORAD_CAT_ID']:05d} {row['INCLINATION']:8.4f} {row['RA_OF_ASC_NODE']:8.4f} {row['ECCENTRICITY']:7.4f} {row['ARG_OF_PERICENTER']:8.4f} {row['MEAN_ANOMALY']:8.4f} {row['MEAN_MOTION']:11.8f}{row['REV_AT_EPOCH']:5d}"
    return line1, line2

def propagate_satellites(tles, start_datetime, end_datetime, step_seconds=60):
    positions = {}
    for line1, line2 in tles:
        satellite = Satrec.twoline2rv(line1, line2)
        jd, fr = jday(start_datetime.year, start_datetime.month, start_datetime.day,
                      start_datetime.hour, start_datetime.minute, start_datetime.second)
        e, r, v = satellite.sgp4(jd, fr)  # Initial propagation to check for errors
        if e != 0:
            print(f"Error propagating {satellite.satnum}: error code {e}")
            continue  # Skip satellites that fail to propagate

        positions[satellite.satnum] = [r]  # Store the initial position
        
        # For a detailed propagation over time, additional logic would be needed here

    return positions


# Function to detect collisions
def detect_collisions(positions, threshold_km=1.0):
    collisions = []
    sat_ids = list(positions.keys())
    for i in range(len(sat_ids)):
        for j in range(i+1, len(sat_ids)):
            for pos_i, pos_j in zip(positions[sat_ids[i]], positions[sat_ids[j]]):
                distance = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
                if distance <= threshold_km:
                    collisions.append((sat_ids[i], sat_ids[j], distance))
                    break  # Assuming only the first collision is of interest
    return collisions

# Visualization of current satellite positions
def visualize_current_positions(positions):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth as a sphere
    earth_radius_km = 6371  # Earth's radius in kilometers
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]  # Create a meshgrid for the sphere
    x_earth = earth_radius_km * np.cos(u) * np.sin(v)
    y_earth = earth_radius_km * np.sin(u) * np.sin(v)
    z_earth = earth_radius_km * np.cos(v)
    ax.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.3, zorder=1)

    # Plot each satellite's current position
    for sat_id, pos_list in positions.items():
        if pos_list:  # Check if there are positions available
            # Assuming each 'pos_list' contains tuples of (x, y, z)
            x, y, z = pos_list[0]  # Get the first position tuple
            ax.scatter(x, y, z, label=f"Sat {sat_id}", s=20, zorder=2)

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Current Satellite Positions in 3D Space')
    plt.legend(loc='upper left', fontsize='small')
    plt.show()


# Main execution
if __name__ == "__main__":
    # Update the CSV file path
    csv_file_path = "TestData.csv"
    norad_data = pd.read_csv(csv_file_path)

    tles = [create_tle_from_norad(row) for _, row in norad_data.iterrows()]
    print(tles)
    
    # Define your propagation time window
    start_datetime = datetime.utcnow()  # For example, start now
    end_datetime = start_datetime + timedelta(hours=24)  # Propagate for 24 hours
    
    positions = propagate_satellites(tles, start_datetime, end_datetime)
    collisions = detect_collisions(positions)
    # Call the visualization function with the positions
    visualize_current_positions(positions)
    print(len(positions))
    print(positions)
    print(len(collisions))
    # Output collisions
    for collision in collisions:
        print(f"Collision detected between satellites {collision[0]} and {collision[1]} with minimum distance {collision[2]:.2f} km")
