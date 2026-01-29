import numpy as np
import matplotlib.pyplot as plt

dt = 0.5
t_end = 100
time = np.arange(0, t_end, dt)

C = 8         # thermal capacitance (J/°C)
R = 2.0           # thermal resistance (°C/W)
Tamb = 25.0       # ambient temperature (°C)
k_tec = 5.0      # TEC power gain (W per control unit)

kp = 8
ki = 0.4
kd = 2.0

T_set = 30.0      # target laser temperature (°C)

T = Tamb
integral = 0.0
prev_error = 0.0

T_history = []
U_history = []

for t in time:
    noise = np.random.normal(0, 0.02)
    T_measured = T + noise

    error = T_set - T_measured
    integral += error * dt
    derivative = (error - prev_error) / dt

    u = kp * error + ki * integral + kd * derivative

    u = np.clip(u, -1.0, 1.0)

    dT = (k_tec * u - (T - Tamb) / R) / C
    T += dT * dt

    T_history.append(T)
    U_history.append(u)

    prev_error = error

    T_history.append(T)
    U_history.append(u)

plt.figure(figsize=(12, 5))
plt.plot(time, T_history, label="Laser Temperature")
plt.axhline(T_set, linestyle="--", label="Setpoint")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("Laser Temperature Control using PID + TEC")
plt.legend()
plt.grid()
plt.show()