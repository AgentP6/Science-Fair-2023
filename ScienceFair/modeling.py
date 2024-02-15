import pandas as pd
import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.constants import J2_earth, R_earth
from poliastro.core.perturbations import J2_perturbation
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants for the exponential atmosphere model
rho0 = 1.225 * 1e-9  # kg/m^3 to kg/km^3
h0 = 0  # Sea level
H = 8.5  # Scale height in km

# Function to plot Earth as a sphere
def plot_earth(ax):
    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
    x = Earth.R / 1000 * np.cos(u) * np.sin(v)
    y = Earth.R / 1000 * np.sin(u) * np.sin(v)
    z = Earth.R / 1000 * np.cos(v)
    ax.plot_surface(x, y, z, color='b', alpha=0.3)

# Function to estimate atmospheric density using an exponential model
def exponential_atmosphere_density(altitude_km):
    rho = rho0 * np.exp(-(altitude_km - h0) / H)
    return rho

# Function to calculate the acceleration due to atmospheric drag
def atmospheric_drag_acceleration(r, v, bstar, rho):
    Cd = 2.2  # Drag coefficient
    A = 1.0  # Cross-sectional area in m^2
    m = 1000  # Mass in kg
    A_km2 = A * 1e-6  # Convert area to km^2
    v_mag = np.linalg.norm(v)
    v_rel = v / v_mag if v_mag > 0 else v  # Relative velocity
    B = bstar / (rho0 * A * Cd)
    a_drag = -0.5 * Cd * A_km2 * rho * v_mag * v_rel / B
    return a_drag

# Function to propagate the orbit considering J2 and atmospheric drag
def to_propagate(t0, state, bstar):
    r = state[:3]
    v = state[3:]
    k = Earth.k.to(u.km**3 / u.s**2).value
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    a_twobody = -k * r / r_norm**3
    a_J2 = J2_perturbation(r, v, k, J2_earth.value, R_earth.to(u.km).value)
    altitude_km = r_norm - Earth.R.to(u.km).value
    rho = exponential_atmosphere_density(altitude_km)
    a_drag = atmospheric_drag_acceleration(r, v, bstar, rho)
    a_total = a_twobody + a_J2 + a_drag
    return np.hstack((v, a_total))

# Read orbital data from CSV file
data = pd.read_csv('TestData.csv')

# Set up the plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot Earth
plot_earth(ax)


# Loop through the DataFrame and propagate each orbit
for index, row in data.iterrows():
    # Extract the orbital elements and BSTAR value
    bstar = row['BSTAR']
    mean_motion = row['MEAN_MOTION']  # in revolutions per day
    eccentricity = row['ECCENTRICITY']
    inclination = row['INCLINATION']  # in degrees
    ra_of_asc_node = row['RA_OF_ASC_NODE']  # in degrees
    arg_of_pericenter = row['ARG_OF_PERICENTER']  # in degrees
    mean_anomaly = row['MEAN_ANOMALY']  # in degrees
    epoch = Time(row['EPOCH'])
    
    # Debug: Print extracted orbital elements
    print(f"Orbit {index}: Mean motion={mean_motion} rev/day, Eccentricity={eccentricity}, Inclination={inclination}")

       # Convert Mean Motion to radians per second
    mean_motion_rad_s = mean_motion * 2 * np.pi / 86400  # Convert to radians per second

    # Calculate the semi-major axis in meters
    mu = Earth.k.to(u.m**3 / u.s**2).value  # Gravitational parameter in m^3/s^2
    a = (mu / mean_motion_rad_s**2)**(1/3) * u.m  # Semi-major axis in meters

    # Create an Orbit instance
    orbit = Orbit.from_classical(
        Earth,
        a.to(u.km),  # Convert a to kilometers, as poliastro expects
        eccentricity * u.one,
        inclination * u.deg,
        ra_of_asc_node * u.deg,
        arg_of_pericenter * u.deg,
        mean_anomaly * u.deg,
        epoch=epoch
    )

    # Debug: Print semi-major axis
    print(f"Semi-major axis for orbit {index}: {a.to(u.km)}")

    # Propagate the orbit considering J2 and atmospheric drag
    t_span = [0, orbit.period.to(u.s).value]  # Time span for one orbital period
    t_eval = np.linspace(t_span[0], t_span[1], num=500)  # Increase num for finer resolution

    # Solve the IVP
    sol = solve_ivp(
        lambda t, y: to_propagate(t, y, bstar), 
        t_span, 
        np.hstack((orbit.r.to(u.km).value, orbit.v.to(u.km / u.s).value)),  # Initial state
        t_eval=t_eval, 
        rtol=1e-8, 
        atol=1e-10
    )
        # Debug: Check if the solution was successful
# Inside the for loop
    if not sol.success:
        print(f"Propagation failed for orbit {index}. Retrying with adjusted tolerances and reduced duration...")

        # Adjust tolerances and reduce duration
        adjusted_duration = orbit.period.to(u.s).value / 10  # Reduce duration to 1/10th of a period
        t_span_adjusted = [0, adjusted_duration]
        t_eval_adjusted = np.linspace(t_span_adjusted[0], t_span_adjusted[1], num=500)

        sol_adjusted = solve_ivp(
            lambda t, y: to_propagate(t, y, bstar),
            t_span_adjusted,
            np.hstack((orbit.r.to(u.km).value, orbit.v.to(u.km / u.s).value)),
            t_eval=t_eval_adjusted,
            rtol=1e-6,  # Adjusted relative tolerance
            atol=1e-9   # Adjusted absolute tolerance
        )

        if sol_adjusted.success:
            rr_adjusted = sol_adjusted.y[:3, :]
            ax.plot(rr_adjusted[0, :], rr_adjusted[1, :], rr_adjusted[2, :], label=f'Orbit {index+1} (adjusted)')
        else:
            print(f"Adjusted propagation still failed for orbit {index}: {sol_adjusted.message}")


    # Extract the propagated positions
    rr = sol.y[:3, :]

    # Plot the orbit
    ax.plot(rr[0, :], rr[1, :], rr[2, :], label=f'Orbit {index+1}')

# Finalize the plot
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Orbital paths around Earth')
ax.legend()
plt.show()