import sgp4
import pandas as pd
from sgp4.api import Satrec
from sgp4.earth_gravity import wgs72


# Load TLE data from CSV
df = pd.read_csv("NORADdata.csv")

# Function to propagate orbit and return position and velocity
def propagate_orbit(tle_data, desired_time):
    tle_lines = tle_data.splitlines()  # Split into lines for clarity
    tle_string = "\n".join(tle_lines)  # Join lines into a string
    satellite = Satrec.twoline2rv(tle_string, "wgs72")  # Pass the string to twoline2rv
    return position, velocity  # Ensure the return statement is present
# Create a list to store the data for each object
orbit_data = []

# Iterate through each row in the CSV
for index, row in df.iterrows():
    tle_line1 = f"1 {row['NORAD_CAT_ID']}U {row['CLASSIFICATION_TYPE']}"
    tle_line2 = f"2 {row['NORAD_CAT_ID']} {row['EPOCH']} {row['MEAN_MOTION_DOT']} {row['MEAN_MOTION_DDOT']} {row['BSTAR']} {row['MEAN_MOTION']}"

    # Desired time (replace with your desired UTC time for propagation)
    desired_time = 2024 * 365.25 + 21  # Example: January 21, 2024

    # Propagate orbit and get position and velocity
    position, velocity = propagate_orbit(tle_line1 + "\n" + tle_line2, desired_time)

    # Collect data for this object
    object_data = {
        "object_name": row['OBJECT_NAME'],
        "position": position.tolist(),  # Convert NumPy array to list
        "velocity": velocity.tolist()
    }
    orbit_data.append(object_data)

# Print the collected data in a structured format
print(orbit_data)
