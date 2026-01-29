import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Time
# -----------------------------
dt = 1          # seconds (thermal systems are slow)
t_end = 100
time = np.arange(0, t_end, dt)

# -----------------------------
# Thermal system parameters
# -----------------------------
C = 8          # thermal capacitance (J/°C)
R = 2.0           # thermal resistance (°C/W)
Tamb = 25.0       # ambient temperature (°C)
k_tec = 5.0       # TEC power gain (W per control unit)

# -----------------------------
# PID parameters (initial)
# -----------------------------
Kp = 7.0
Ki = 0.4
Kd = 2.0

# -----------------------------
# Setpoint
# -----------------------------
T_set = 30.0      # target laser temperature (°C)

# -----------------------------
# Initialization
# -----------------------------
T = Tamb
integral = 0.0
prev_error = 0.0

T_history = []
U_history = []

# -----------------------------
# Simulation loop
# -----------------------------
for t in time:
    # Thermistor measurement (with noise)
    noise = np.random.normal(0, 0.02)
    T_measured = T + noise

    # PID
    error = T_set - T_measured
    integral += error * dt
    derivative = (error - prev_error) / dt

    u = Kp * error + Ki * integral + Kd * derivative

    # Saturation (real TEC limits)
    u = np.clip(u, -1.0, 1.0)

    # Thermal dynamics
    dT = (k_tec * u - (T - Tamb) / R) / C
    T += dT * dt

    prev_error = error

    T_history.append(T)
    U_history.append(u)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(time, T_history, label="Laser Temperature")
plt.axhline(T_set, linestyle="--", label="Setpoint")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("Laser Temperature Control using PID + TEC")
plt.legend()
plt.grid()
plt.show()
