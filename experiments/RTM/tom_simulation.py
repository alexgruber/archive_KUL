
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button

def TOM(tau, omega, Rs, Ts, t_deg):

    theta = t_deg * np.pi / 180.
    gamma = np.exp(-tau / np.cos(theta))                        # tau is the only parameter that causes non-linearity!

    Tbv = (1 - omega) * (1 - gamma) * Ts                        # Vegetation emission
    Tbs = Ts * gamma * (1 - Rs)                                 # Attenuated soil emmission
    Tbvs = (1 - omega) * (1 - gamma) * Ts * gamma * Rs          # Soil-reflected vegetation emission

    Tb = Tbv + Tbs + Tbvs

    return Tb

def draw(ax, x, y, Tb, xlabel, ylabel):
    ax.cla()
    ax.plot_surface(x, y, Tb, cmap = 'twilight')
    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=12)
    ax.set_zlabel('$T_b$ [K]', fontsize=fontsize, labelpad=12)
    ax.set_zlim(130,320)
    for axs in [ax.xaxis, ax.yaxis, ax.zaxis]:
        for tick in axs.get_major_ticks():
            tick.label.set_fontsize(fontsize)

def update_all(val):
    t_deg = st.val
    omega = som.val
    Rs = sRs.val
    Ts = sTs.val
    Tb1 = TOM(tau1, omega1, Rs, Ts, t_deg)
    Tb2 = TOM(tau2, omega, Rs, Ts2, t_deg)
    Tb3 = TOM(tau3, omega, Rs3, Ts, t_deg)
    draw(ax1, omega1, tau1, Tb1, '$\omega$ [-]', r'$\tau$ [-]')
    draw(ax2, Ts2, tau2, Tb2, '$T_s$ [-]', r'$\tau$ [-]')
    draw(ax3, Rs3, tau3, Tb3, '$R_s$ [-]', r'$\tau$ [-]')

def update_omega(val):
    t_deg = st.val
    omega = som.val
    Rs = sRs.val
    Ts = sTs.val
    Tb2 = TOM(tau2, omega, Rs, Ts2, t_deg)
    Tb3 = TOM(tau3, omega, Rs3, Ts, t_deg)
    draw(ax2, Ts2, tau2, Tb2, '$T_s$ [-]', r'$\tau$ [-]')
    draw(ax3, Rs3, tau3, Tb3, '$R_s$ [-]', r'$\tau$ [-]')

def update_ts(val):
    t_deg = st.val
    omega = som.val
    Rs = sRs.val
    Ts = sTs.val
    Tb1 = TOM(tau1, omega1, Rs, Ts, t_deg)
    Tb3 = TOM(tau3, omega, Rs3, Ts, t_deg)
    draw(ax1, omega1, tau1, Tb1, '$\omega$ [-]', r'$\tau$ [-]')
    draw(ax3, Rs3, tau3, Tb3, '$R_s$ [-]', r'$\tau$ [-]')

def update_rs(val):
    t_deg = st.val
    omega = som.val
    Rs = sRs.val
    Ts = sTs.val
    Tb1 = TOM(tau1, omega1, Rs, Ts, t_deg)
    Tb2 = TOM(tau2, omega, Rs, Ts2, t_deg)
    draw(ax1, omega1, tau1, Tb1, '$\omega$ [-]', r'$\tau$ [-]')
    draw(ax2, Ts2, tau2, Tb2, '$T_s$ [-]', r'$\tau$ [-]')

def reset_all(event):
    st.reset()
    som.reset()
    sTs.reset()
    sRs.reset()

# Realize function with initial guesses
t_deg0 = 40
Ts0 = 293
Rs0 = 0.3
omega0 = 0.2

# Create axes ranges
omega = np.linspace(0.0, 0.6, 100)
tau = np.linspace(0.0, 2.5, 100)
Ts = np.linspace(273, 313, 100)
Rs = np.linspace(0.0, 0.5, 100)

omega1, tau1 = np.meshgrid(omega, tau)
Ts2, tau2 = np.meshgrid(Ts, tau)
Rs3, tau3 = np.meshgrid(Rs, tau)

Tb1 = TOM(tau1, omega1, Rs0, Ts0, t_deg0)
Tb2 = TOM(tau2, omega0, Rs0, Ts2, t_deg0)
Tb3 = TOM(tau3, omega0, Rs3, Ts0, t_deg0)

# Create the plotting window
fig = plt.figure(figsize=(20,6))
ax1 = fig.add_subplot(1,3,1, projection='3d')
ax2 = fig.add_subplot(1,3,2, projection='3d')
ax3 = fig.add_subplot(1,3,3, projection='3d')
fontsize = 10

# Plot with initial values
draw(ax1, omega1, tau1, Tb1, '$\omega$ [-]', r'$\tau$ [-]')
draw(ax2, Ts2, tau2, Tb2, '$T_s$ [-]', r'$\tau$ [-]')
draw(ax3, Rs3, tau3, Tb3, '$R_s$ [-]', r'$\tau$ [-]')

# Add axes for sliders
axcolor = 'lightgoldenrodyellow'
plt.subplots_adjust(left=0.00, right=0.95, top=1, bottom=0.20)
axt = plt.axes([0.15, 0.125, 0.75, 0.025], facecolor=axcolor)
axom = plt.axes([0.15, 0.095, 0.75, 0.025], facecolor=axcolor)
axTs = plt.axes([0.15, 0.065, 0.75, 0.025], facecolor=axcolor)
axRs = plt.axes([0.15, 0.035, 0.75, 0.025], facecolor=axcolor)
resetax = plt.axes([0.8, 0.005, 0.1, 0.025])

# Add sliders and reset button to axes
st = Slider(axt, 'Theta', 25, 55, valinit=t_deg0)
som = Slider(axom, 'Single Scattering Albedo', 0, 0.6, valinit=omega0)
sTs = Slider(axTs, 'Skin Temperature', 273, 313, valinit=Ts0)
sRs = Slider(axRs, 'Soil Reflectivity', 0, 0.4, valinit=Rs0)
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

# Enable slider and button functionalities
st.on_changed(update_all)
som.on_changed(update_omega)
sTs.on_changed(update_ts)
sRs.on_changed(update_rs)

button.on_clicked(reset_all)

plt.show()