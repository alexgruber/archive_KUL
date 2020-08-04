
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button

def WCM(SM, LAI, A, B, C, D, t_deg):

    theta = t_deg * np.pi / 180.
    sig0s = 10 ** ((C + D * SM)/10)
    T2 = np.exp((-2 * B * LAI) / np.cos(theta))
    sig0v = A * LAI * np.cos(theta) * (1 - T2)
    sig0 = 10 * np.log10(T2 * sig0s + sig0v)

    return sig0

def draw(SM, LAI, sig0):
    ax.cla()
    ax.plot_surface(SM, LAI, sig0, cmap = 'twilight')
    ax.set_xlabel('Soil Moisture [$m^3m^{-3}$]', fontsize=fontsize, labelpad=12)
    ax.set_ylabel('Leaf Area Index [$m^2m^{-2}$]', fontsize=fontsize, labelpad=12)
    ax.set_zlabel('$\sigma^0$ [$dB$]', fontsize=fontsize, labelpad=12)
    ax.set_zlim(-20,5)
    for axs in [ax.xaxis, ax.yaxis, ax.zaxis]:
        for tick in axs.get_major_ticks():
            tick.label.set_fontsize(fontsize)

def update(val):
    A = sA.val
    B = sB.val
    C = sC.val
    D = sD.val
    t_deg = st.val
    sig0 = WCM(SM, LAI, A, B, C, D, t_deg)
    draw(SM, LAI, sig0)

def reset(event):
    sA.reset()
    sB.reset()
    sC.reset()
    sD.reset()
    st.reset()

# Create the plot in general
fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
plt.subplots_adjust(left=0.1, right=0.9, top=1, bottom=0.20)

# Realize function with initial guesses
A0 = 0.2
B0 = 0.2
C0 = -20
D0 = 50
t_deg0 = 35
SM = np.linspace(0.0, 0.6, 100)
LAI = np.linspace(0.0, 10, 100)
SM, LAI = np.meshgrid(SM, LAI)
sig0 = WCM(SM, LAI, A0, B0, C0, D0, t_deg0)

# Initial plot
fontsize = 14
draw(SM, LAI, sig0)

# Create axes for sliders
axcolor = 'lightgoldenrodyellow'
axt = plt.axes([0.1, 0.155, 0.8, 0.025], facecolor=axcolor)
axA = plt.axes([0.1, 0.125, 0.8, 0.025], facecolor=axcolor)
axB = plt.axes([0.1, 0.095, 0.8, 0.025], facecolor=axcolor)
axC = plt.axes([0.1, 0.065, 0.8, 0.025], facecolor=axcolor)
axD = plt.axes([0.1, 0.035, 0.8, 0.025], facecolor=axcolor)
resetax = plt.axes([0.8, 0.005, 0.1, 0.025])

# Add sliders and reset button to axes
st = Slider(axt, 'Theta', 25, 65, valinit=t_deg0)
sA = Slider(axA, 'A', 0, 0.4, valinit=A0)
sB = Slider(axB, 'B', 0, 0.4, valinit=B0)
sC = Slider(axC, 'C', -30, -10, valinit=C0)
sD = Slider(axD, 'D', 25, 65, valinit=D0)
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

# Enable slider and button functionalities
sA.on_changed(update)
sB.on_changed(update)
sC.on_changed(update)
sD.on_changed(update)
st.on_changed(update)
button.on_clicked(reset)

plt.show()