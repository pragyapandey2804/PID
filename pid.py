import numpy as np
import matplotlib.pyplot as plt

# =============================
# Time
# =============================
dt = 0.1
t_end = 600
time = np.arange(0, t_end, dt)

# =============================
# Thermal parameters
# =============================
C = 8.0           # thermal capacitance (J/°C)
R = 2.5            # thermal resistance (°C/W)
Tamb0 = 25.0       # nominal ambient temp
k_tec = 6.0        # TEC gain (W/unit)

tau_laser = 5.0    # laser thermal lag (s)

# =============================
# PID (PI-D) gains
# =============================
Kp = 1.0
Ki = 0.02
Kd = 10.0

# =============================
# Setpoint
# =============================
T_set = 30.0

# =============================
# Initial states
# =============================
T_mount = Tamb0
T_laser = Tamb0

integral = 0.0
prev_T = T_laser

T_log = []
U_log = []
Tamb_log = []

# =============================
# Simulation loop
# =============================
for t in time:

    # Ambient drift (slow)
    Tamb = Tamb0 + 0.5 * np.sin(2*np.pi*t/600)

    # Sensor noise
    T_measured = T_laser + np.random.normal(0, 0.01)

    # Error
    error = T_set - T_measured

    # Integral (anti-windup)
    integral += error * dt
    integral = np.clip(integral, -20, 20)

    # Derivative on measurement
    dTdt = (T_laser - prev_T) / dt

    # PI-D control law
    u = Kp * error + Ki * integral - Kd * dTdt
    u = np.clip(u, -1.0, 1.0)

    # Mount thermal dynamics
    dT_mount = (k_tec * u - (T_mount - Tamb) / R) / C
    T_mount += dT_mount * dt

    # Laser thermal lag
    dT_laser = (T_mount - T_laser) / tau_laser
    T_laser += dT_laser * dt

    prev_T = T_laser

    # Logging
    T_log.append(T_laser)
    U_log.append(u)
    Tamb_log.append(Tamb)

# =============================
# Plot
# =============================
plt.figure(figsize=(12,5))
plt.plot(time, T_log, label="Laser Temperature")
plt.plot(time, Tamb_log, '--', label="Ambient")
plt.axhline(T_set, linestyle=':', label="Setpoint")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("Laser Temperature Stabilization using TEC (Improved Model)")
plt.legend()
plt.grid()
plt.show()
