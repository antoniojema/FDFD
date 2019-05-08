import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import h5py as h5
import argparse
import sys
from mpl_toolkits.mplot3d import Axes3D


L = 0
U = 1

X = 0
Y = 1

c_alpha = 0.3
colors = np.array([
    [ 50, 205,  50, c_alpha],
    [138,  43, 226, c_alpha],
    [255, 255,   0, c_alpha],
    [178,  34,  34, c_alpha]
])
colors[:,:-1] /= 255


def snap(x, v):
    dist = []
    for i in v:
        dist += [abs(x-i)]
    return np.argmin(dist)


def setRanges(data, pos):
    ranges = []
    for dat_str in data:
        dat = data[dat_str]
        val = dat["value"][0]
        lims = (pos[snap(dat["limits"][L], pos)], pos[snap(dat["limits"][U], pos)])
        found = False
        for i in range(len(ranges)):
            if ranges[i][1] == val:
                found = True
                val_i = i
                break
        if not found:
            ranges.append([[[lims[L], lims[U]]], val])
        else:
            ranges[val_i][0].append([lims[L], lims[U]])
    return ranges


#   Read data   #
parser = argparse.ArgumentParser(description='Python FDFD plot')
parser.add_argument('-i', '--input', nargs=1, type=str)
args = parser.parse_args()
if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()
inputFilename = ''.join(args.input).strip()
data = h5.File(inputFilename, 'r')


#   Set fields   #
posE = np.array(data["posE"][:], dtype=complex)
posH = np.array(data["posH"][:], dtype=complex)
Ex = np.array(data["Ex"][:], dtype=complex)
Ey = np.array(data["Ey"][:], dtype=complex)
Hx = np.array(data["Hx"][:], dtype=complex)
Hy = np.array(data["Hy"][:], dtype=complex)


#   Plot modulus   #
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(posE, abs(Ex), np.zeros(len(posE)), color="blue", label="Ex")
ax.plot(posE, np.zeros(len(posE)), abs(Ey), color="blue", label="Ey")
ax.plot(posH, abs(Hx), np.zeros(len(posH)), color="orange", label="Hx")
ax.plot(posH, np.zeros(len(posH)), abs(Hy), color="orange", label="Hy")
ymax = max([max(abs(Ex)), max(abs(Ey)), max(abs(Hx)), max(abs(Hy))])
if "eps_yy" in data:
    color_i = 0
    for rang in setRanges(data["eps_yy"], posE):
        for lims in rang[0]:
            xc, yc, zc = np.indices((2, 2, 2))
            xc = xc * (lims[U] - lims[L]) + lims[L]
            yc = yc * ymax
            zc = zc * ymax
            ax.voxels(xc, yc, zc, np.array([[[True]]]), facecolors=np.array([[[colors[color_i]]]]))
        color_i = (color_i + 1) % len(colors)
elif "sigmaE" in data:
    color_i = 0
    for rang in setRanges(data["sigmaE"], posE):
        for lims in rang[0]:
            xc, yc, zc = np.indices((2, 2, 2))
            xc = xc * (lims[U] - lims[L]) + lims[L]
            yc = yc * ymax
            zc = zc * ymax
            ax.voxels(xc, yc, zc, np.array([[[True]]]), facecolors=np.array([[[colors[color_i]]]]))
        color_i = (color_i + 1) % len(colors)

ax.set_xlim(posE[0], posE[-1])
ax.set_ylim(0, 1.1 * ymax)
ax.set_zlim(0, 1.1 * ymax)
plt.show()

#   Animation   #
fig = plt.figure()
ax = fig.gca(projection="3d")
t_scale = 0.05
ymax = max([max(abs(Ex)), max(abs(Ey)), max(abs(Hx)), max(abs(Hy))])
plot_step = 20


def init():
    plot_posE = posE[::plot_step]
    zeroE = np.zeros(len(plot_posE))
    plot_Ex = (abs(Ex) * np.cos(np.angle(Ex)))[::plot_step]
    plot_Ey = (abs(Ey) * np.cos(np.angle(Ey)))[::plot_step]
    
    plot_posH = posH[::plot_step]
    zeroH = np.zeros(len(plot_posH))
    plot_Hx = (abs(Hx) * np.cos(np.angle(Hx)))[::plot_step]
    plot_Hy = (abs(Hy) * np.cos(np.angle(Hy)))[::plot_step]
    
    ax.quiver(plot_posE, zeroE, zeroE, zeroE, plot_Ex, plot_Ey, color="blue", arrow_length_ratio=0)
    ax.quiver(plot_posH, zeroH, zeroH, zeroH, plot_Hx, plot_Hy, color="orange", arrow_length_ratio=0)
    ax.plot(plot_posE, plot_Ex, plot_Ey, color="blue")
    ax.plot(plot_posH, plot_Hx, plot_Hy, color="orange")

    if "eps_yy" in data:
        color_i = 0
        for rang in setRanges(data["eps_yy"], posE):
            for lims in rang[0]:
                xc, yc, zc = np.indices((2, 2, 2))
                xc = xc * (lims[U] - lims[L]) + lims[L]
                yc = yc * 2*ymax - ymax
                zc = zc * 2*ymax - ymax
                ax.voxels(xc, yc, zc, np.array([[[True]]]), facecolors=np.array([[[colors[color_i]]]]))
            color_i = (color_i + 1) % len(colors)
    elif "sigmaE" in data:
        color_i = 0
        for rang in setRanges(data["sigmaE"], posE):
            for lims in rang[0]:
                xc, yc, zc = np.indices((2, 2, 2))
                xc = xc * (lims[U] - lims[L]) + lims[L]
                yc = yc * 2 * ymax - ymax
                zc = zc * 2 * ymax - ymax
                ax.voxels(xc, yc, zc, np.array([[[True]]]), facecolors=np.array([[[colors[color_i]]]]))
            color_i = (color_i + 1) % len(colors)
    
    ax.set_ylim(-1.1 * ymax, 1.1 * ymax)
    ax.set_zlim(-1.1 * ymax, 1.1 * ymax)
    ax.set_xlim(posE[0], posE[-1])


def iteration(i):
    plt.cla()
    
    plot_posE = posE[::plot_step]
    zeroE = np.zeros(len(plot_posE))
    plot_Ex = (abs(Ex) * np.cos(t_scale * i + np.angle(Ex)))[::plot_step]
    plot_Ey = (abs(Ey) * np.cos(t_scale * i + np.angle(Ey)))[::plot_step]
    
    plot_posH = posH[::plot_step]
    zeroH = np.zeros(len(plot_posH))
    plot_Hx = (abs(Hx) * np.cos(t_scale * i + np.angle(Hx)))[::plot_step]
    plot_Hy = (abs(Hy) * np.cos(t_scale * i + np.angle(Hy)))[::plot_step]
    
    ax.quiver(plot_posE, zeroE, zeroE, zeroE, plot_Ex, plot_Ey, color="blue", arrow_length_ratio=0)
    ax.quiver(plot_posH, zeroH, zeroH, zeroH, plot_Hx, plot_Hy, color="orange", arrow_length_ratio=0)
    ax.plot(plot_posE, plot_Ex, plot_Ey, color="blue")
    ax.plot(plot_posH, plot_Hx, plot_Hy, color="orange")

    if "eps_yy" in data:
        color_i = 0
        for rang in setRanges(data["eps_yy"], posE):
            for lims in rang[0]:
                xc, yc, zc = np.indices((2, 2, 2))
                xc = xc * (lims[U] - lims[L]) + lims[L]
                yc = yc * 2*ymax - ymax
                zc = zc * 2*ymax - ymax
                ax.voxels(xc, yc, zc, np.array([[[True]]]), facecolors=np.array([[[colors[color_i]]]]))
            color_i = (color_i + 1) % len(colors)
    elif "sigmaE" in data:
        color_i = 0
        for rang in setRanges(data["sigmaE"], posE):
            for lims in rang[0]:
                xc, yc, zc = np.indices((2, 2, 2))
                xc = xc * (lims[U] - lims[L]) + lims[L]
                yc = yc * 2 * ymax - ymax
                zc = zc * 2 * ymax - ymax
                ax.voxels(xc, yc, zc, np.array([[[True]]]), facecolors=np.array([[[colors[color_i]]]]))
            color_i = (color_i + 1) % len(colors)
    
    ax.set_ylim(-1.1 * ymax, 1.1 * ymax)
    ax.set_zlim(-1.1 * ymax, 1.1 * ymax)
    ax.set_xlim(posE[0], posE[-1])


anim = FuncAnimation(fig, iteration, init_func=init, frames=np.arange(1, 2 * np.pi / t_scale + 1), interval=100)
plt.show()