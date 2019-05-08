import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import c as c0
import json
import argparse
import sys
import h5py as h5

X = 0
Y = 1

L = 0
U = 1

c_alpha = 0.3
colors = np.array([
    [ 50, 205,  50, c_alpha],
    [138,  43, 226, c_alpha],
    [255, 255,   0, c_alpha],
    [178,  34,  34, c_alpha]
])
colors[:,:-1] /= 255


def gaussian(x, mean, sigma):
    return 1/np.sqrt(2*np.pi*sigma*sigma) * np.exp(-(x-mean)**2/(2*sigma*sigma))


def snap(x, v):
    dist = []
    for i in v:
        dist += [abs(x-i)]
    return np.argmin(dist)


def setRanges(data, pos):
    ranges = []
    for dat in data:
        val = dat["value"]
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


def h5save(output, **kwargs):
    fout = h5.File(output, "w")
    for arg in kwargs:
        if kwargs[arg] is None:
            pass
        elif not type(kwargs[arg][0]) == dict:
            fout[arg] = kwargs[arg][:]
        else:
            h5dict(fout, arg, kwargs[arg])
    fout.close()


def h5dict(group, string, dict_arr):
    subgroup = group.create_group(string)
    for i in range(len(dict_arr)):
        dic = dict_arr[i]
        subsubgroup = subgroup.create_group(str(i))
        for arg in dic:
            if not isIterable(dic[arg]):
                subsubgroup[arg] = [dic[arg]]
            elif not type(dic[arg][0]) == dict:
                subsubgroup[arg] = dic[arg][:]
            else:
                h5dict(subsubgroup, arg, dic[arg])
    return


def isIterable(obj):
    try:
        dump = iter(obj)
        return True
    except:
        return False
    

#   Read data   #
parser = argparse.ArgumentParser(description='Python FDFD 1D')
parser.add_argument('-i', '--input', nargs=1, type=str)
args = parser.parse_args()
if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

inputFilename = ''.join(args.input).strip()
data = json.load(open(inputFilename))


#   Set grid   #
grid_lims = data["grid"]["limits"]
grid_L = grid_lims[U]-grid_lims[L]
Dz = data["grid"]["step"]
posE = np.arange(grid_lims[L], grid_lims[U]+Dz, Dz)
posH = posE[:-1]+Dz/2
N = len(posH)


#   Set boundaries   #
bound = data["boundaries"]


#   Set epsilon and mu   #
eps_xx = np.ones(len(posE), dtype=complex)
eps_yy = np.ones(len(posE), dtype=complex)
if "epsilon" in data:
    if "xx" in data["epsilon"]:
        for eps_ in data["epsilon"]["xx"]:
            lims = (snap(eps_["limits"][L], posE), snap(eps_["limits"][U], posE))
            eps_xx[lims[L]:lims[U] + 1] = eps_["value"]
    if "yy" in data["epsilon"]:
        for eps_ in data["epsilon"]["yy"]:
            lims = (snap(eps_["limits"][L], posE), snap(eps_["limits"][U], posE))
            eps_yy[lims[L]:lims[U] + 1] = eps_["value"]

mu_xx = np.ones(len(posH))
mu_yy = np.ones(len(posH))
if "mu" in data:
    if "xx" in data["mu"]:
        for mu_ in data["mu"]["xx"]:
            lims = (snap(mu_["limits"][L], posH), snap(mu_["limits"][U], posH))
            mu_xx[lims[L]:lims[U] + 1] = mu_["value"]
    if "yy" in data["mu"]:
        for mu_ in data["mu"]["yy"]:
            lims = (snap(mu_["limits"][L], posH), snap(mu_["limits"][U], posH))
            mu_yy[lims[L]:lims[U] + 1] = mu_["value"]

# c = 1/np.sqrt(eps*mu) # Esto peta, eps y mu tienen distinta longitud


#   Set sigmaE   #
sigmaE = np.zeros(len(posE), dtype=complex)
if "sigmaE" in data:
    for sigma_ in data["sigmaE"]:
        lims = (snap(sigma_["limits"][L], posE), snap(sigma_["limits"][U], posE))
        sigmaE[lims[L]:lims[U]+1] = sigma_["value"]


#   Set sigmaH   #
sigmaH = np.zeros(len(posH), dtype=complex)
if "sigmaH" in data:
    for sigma_ in data["sigmaH"]:
        lims = (snap(sigma_["limits"][L], posH), snap(sigma_["limits"][U], posH))
        sigmaH[lims[L]:lims[U]+1] = sigma_["value"]


#   Set frequency and wavelength   #
if "wavelength_0" in data:
    wavelength_0 = data["wavelength_0"]
    freq = 1 / wavelength_0
    w = 2*np.pi*freq
elif "w" in data:
    w = data["w"]
    freq = w/(2*np.pi)
    wavelength_0 = 1 / freq
elif "frequency" in data:
    freq = data["frequency"]
    w = 2*np.pi*freq
    wavelength_0 = 1 / freq
else:
    raise ValueError("Invalid or missing wave frequency: data must contain \"frequency\", \"w\" or \"wavelength_0\"")


#   Set sources   #
J_x = np.zeros(len(posE), dtype=complex)
J_y = np.zeros(len(posE), dtype=complex)
JM_x = np.zeros(len(posH), dtype=complex)
JM_y = np.zeros(len(posH), dtype=complex)
if "sources" in data:
    if "electric" in data["sources"]:
        for source in data["sources"]["electric"]:
            lims = (snap(source["limits"][L], posE), snap(source["limits"][U], posE))
            if source["type"] == "gaussian":
                J = source["magnitude"] * gaussian(posE[lims[L]:lims[U]+1], source["mean"], source["sigma"]) * complex(np.cos(source["phase"]), np.sin(source["phase"]))
            elif source["type"] == "uniform":
                J = source["magnitude"] * complex(np.cos(source["phase"]), np.sin(source["phase"]))
            else:
                raise ValueError("Invalid source type: "+source["type"])
            direction = np.array(source["direction"])
            direction = direction / np.linalg.norm(direction)
            J_x[lims[L]:lims[U]+1] += J * direction[X]
            J_y[lims[L]:lims[U]+1] += J * direction[Y]

    if "magnetic" in data["sources"]:
        for source in data["sources"]["magnetic"]:
            lims = (snap(source["limits"][L], posH), snap(source["limits"][U], posH))
            if source["type"] == "gaussian":
                J = source["magnitude"] * gaussian(posH[lims[L]:lims[U]+1], source["mean"], source["sigma"]) * complex(np.cos(source["phase"]), np.sin(source["phase"]))
            elif source["type"] == "uniform":
                J = source["magnitude"] * complex(np.cos(source["phase"]), np.sin(source["phase"]))
            else:
                raise ValueError("Invalid source type: "+source["type"])
            direction = np.array(source["direction"])
            direction = direction / np.mod(direction)
            JM_x[lims[L]:lims[U]+1] += J * direction[X]
            JM_y[lims[L]:lims[U]+1] += J * direction[Y]


#            Ex Hy            #
#   Set boundary conditions for the system   #
if bound == "periodic":
    BE1 = np.concatenate(([1j*w*eps_xx[0]*Dz+sigmaE[0]*Dz], np.zeros(N)),  axis=None).reshape(1, N+1).astype(complex)
    BE2 = np.concatenate((np.zeros(N), [1j*w*eps_xx[-1]*Dz+sigmaE[-1]*Dz]),  axis=None).reshape(1, N+1).astype(complex)
    BH1 = np.concatenate(([1], np.zeros(N-2), [-1]), axis=None).reshape(1, N).astype(complex)
    BH2 = BH1[:]
    BJ1 = BJ2 = J_x[0]
elif type(bound) == list and \
     len(bound) == 2:
    if bound[L] == "pec":
        BE1 = np.concatenate(([1], np.zeros(N)), axis=None).reshape(1, N+1).astype(complex)
        BH1 = np.zeros((1, N), dtype=complex)
        BJ1 = 0
    elif bound[L] == "mur":
        BE1 = np.concatenate(([1+1j*w*Dz*np.sqrt(eps_xx[0]*mu_xx[0]), -1], np.zeros(N-1)), axis=None).reshape(1, N+1)
        BH1 = np.zeros((1,N), dtype=complex)
        BJ1 = 0
    else:
        raise ValueError("Invalid lower boundary: "+bound[L])

    if bound[U] == "pec":
        BE2 = np.concatenate((np.zeros(N), [1]), axis=None).reshape(1, N+1).astype(complex)
        BH2 = np.zeros((1, N), dtype=complex)
        BJ2 = 0
    elif bound[U] == "mur":
        BE2 = np.concatenate((np.zeros(N-1), [-1, 1+1j*w*Dz*np.sqrt(eps_xx[-1]*mu_xx[-1])]), axis=None).reshape(1, N+1)
        BH2 = np.zeros((1,N), dtype=complex)
        BJ2 = 0
    else:
        raise ValueError("Invalid upper boundary: "+bound[U])
else:
    raise ValueError("Boundary format can only be a 2-element list or \"periodic\"")


#   Set system   #
A = np.c_[-1*np.eye(N), np.zeros(N)] + np.c_[np.zeros(N), np.eye(N)].astype(complex)
B = np.r_[BH1, A[:-1, :-1], BH2]

eps_xx_m = np.r_[BE1, 1j*w*Dz*np.diag(eps_xx)[1:-1]+Dz*np.diag(sigmaE)[1:-1], BE2]
mu_xx_m = 1j*w*Dz*np.diag(mu_xx) + Dz*np.diag(sigmaH)
mu_xx_m_ = np.diag(1/np.diag(mu_xx_m))

J_x[0] = BJ1
J_x[-1] = BJ2

# A * Ex = - [ i w mu  Dz + sigmaM Dz ] * Hy - Dz JM_y = - mu_m  * Hy - Dz * JM_y
# B * Hy = - [ i w eps Dz + sigmaE Dz ] * Ex - Dz J_x  = - eps_m * Ex - Dz * J_x
#     mu_m_ * A * Ex = -     Hy -     mu_m_ * Dz * JM_y
# [B * mu_m_ * A] * Ex = - [B * Hy] - [B * mu_m_ * Dz * JM_y]
# [B * mu_m_ * A] * Ex = [eps_m] * Ex + [Dz * J_x - B * mu_m_ * Dz * JM_y]
# [B * mu_m_ * A - eps_m] * Ex = [Dz * J_x - B * mu_m_ * Dz * JM_y]
# M * Ex =      [Dz * J_x - B * mu_m_ * Dz * JM_y]
#     Ex = M_ * [Dz * J_x - B * mu_m_ * Dz * JM_y]
# Hy = - mu_m_ * (Dz * JM_y + A * Ex)
M = np.matmul(np.matmul(B, mu_xx_m_), A) - eps_xx_m


#   Solve system   #
M_ = np.linalg.inv(M)
Ex = Dz * np.matmul(M_, J_x - np.matmul(np.matmul(B, mu_xx_m_), JM_y))
Hy = -1 * np.matmul(mu_xx_m_, Dz*JM_y+np.matmul(A, Ex))


#            Ey Hx            #
#   Set boundary conditions for the system   #
if bound == "periodic":
    BE1 = np.concatenate(([1j*w*eps_yy[0]*Dz+sigmaE[0]*Dz], np.zeros(N)),  axis=None).reshape(1, N+1).astype(complex)
    BE2 = np.concatenate((np.zeros(N), [1j*w*eps_yy[-1]*Dz+sigmaE[-1]*Dz]),  axis=None).reshape(1, N+1).astype(complex)
    BH1 = np.concatenate(([1], np.zeros(N-2), [-1]), axis=None).reshape(1, N).astype(complex)
    BH2 = BH1[:]
    BJ1 = BJ2 = J_y[0]
elif type(bound) == list and \
     len(bound) == 2:
    if bound[L] == "pec":
        BE1 = np.concatenate(([1], np.zeros(N)), axis=None).reshape(1, N+1).astype(complex)
        BH1 = np.zeros((1, N), dtype=complex)
        BJ1 = 0
    elif bound[L] == "mur":
        BE1 = np.concatenate(([1+1j*w*Dz*np.sqrt(eps_yy[0]*mu_yy[0]), -1], np.zeros(N-1)), axis=None).reshape(1, N+1)
        BH1 = np.zeros((1, N), dtype=complex)
        BJ1 = 0
    else:
        raise ValueError("Invalid lower boundary: "+bound[L])

    if bound[U] == "pec":
        BE2 = np.concatenate((np.zeros(N), [1]), axis=None).reshape(1, N+1).astype(complex)
        BH2 = np.zeros((1, N), dtype=complex)
        BJ2 = 0
    elif bound[U] == "mur":
        BE2 = np.concatenate((np.zeros(N-1), [-1, 1+1j*w*Dz*np.sqrt(eps_yy[-1]*mu_yy[-1])]), axis=None).reshape(1, N+1)
        BH2 = np.zeros((1, N), dtype=complex)
        BJ2 = 0
    else:
        raise ValueError("Invalid upper boundary: "+bound[U])
else:
    raise ValueError("Boundary format can only be a 2-element list or \"periodic\"")


#   Set system   #
A = np.c_[-1*np.eye(N), np.zeros(N)] + np.c_[np.zeros(N), np.eye(N)].astype(complex)
B = np.r_[BH1, A[:-1, :-1], BH2]

eps_yy_m = np.r_[BE1, 1j*w*Dz*np.diag(eps_yy)[1:-1]+Dz*np.diag(sigmaE)[1:-1], BE2]
mu_yy_m = 1j*w*Dz*np.diag(mu_yy) + Dz*np.diag(sigmaH)
mu_yy_m_ = np.diag(1/np.diag(mu_yy_m))

J_y[0] = BJ1
J_y[-1] = BJ2

M = np.matmul(np.matmul(B, -mu_yy_m_), A) + eps_yy_m


#   Solve system   #
M_ = np.linalg.inv(M)
Ey = -Dz * np.matmul(M_, J_y - np.matmul(np.matmul(B, mu_yy_m_), JM_x))
Hx = -1 * np.matmul(-mu_yy_m_, -Dz*JM_x+np.matmul(A, Ey))


#   HDF5   #
if "output" in data:
    outputFilename = data["output"].strip()
    if outputFilename[-3:] == ".h5":
        eps_xx_s = None
        eps_yy_s = None
        mu_xx_s = None
        mu_yy_s = None
        sigmaE_s = None
        sigmaH_s = None
        if "epsilon" in data:
            if "xx" in data["epsilon"]:
                eps_xx_s = data["epsilon"]["xx"][:]
            if "yy" in data["epsilon"]:
                eps_yy_s = data["epsilon"]["yy"][:]
        if "mu" in data:
            if "xx" in data["mu"]:
                mu_xx_s = data["mu"]["xx"][:]
            if "yy" in data["mu"]:
                mu_yy_s = data["mu"]["yy"][:]
        if "sigmaE" in data:
            sigmaE_s = data["sigmaE"][:]
        if "sigmaH" in data:
            sigmaH_s = data["sigmaH"][:]
        h5save(outputFilename, posE=posE, posH=posH, Ex=Ex, Ey=Ey, Hx=Hx, Hy=Hy, eps_xx=eps_xx_s, eps_yy=eps_yy_s, mu_xx=mu_xx_s, mu_yy=mu_yy_s, sigmaE=sigmaE_s, sigmaH=sigmaH_s)
    else:
        raise ValueError()
else:
    eps_xx_s = None
    eps_yy_s = None
    mu_xx_s = None
    mu_yy_s = None
    sigmaE_s = None
    sigmaH_s = None
    if "epsilon" in data:
        if "xx" in data["epsilon"]:
            eps_xx_s = data["epsilon"]["xx"][:]
        if "yy" in data["epsilon"]:
            eps_yy_s = data["epsilon"]["yy"][:]
    if "mu" in data:
        if "xx" in data["mu"]:
            mu_xx_s = data["mu"]["xx"][:]
        if "yy" in data["mu"]:
            mu_yy_s = data["mu"]["yy"][:]
    if "sigmaE" in data:
        sigmaE_s = data["sigmaE"][:]
    if "sigmaH" in data:
        sigmaH_s = data["sigmaH"][:]
    h5save(inputFilename.replace(".json", ".h5"), posE=posE, posH=posH, Ex=Ex, Ey=Ey, Hx=Hx, Hy=Hy, eps_xx=eps_xx_s, eps_yy=eps_yy_s, mu_xx=mu_xx_s, mu_yy=mu_yy_s, sigmaE=sigmaE_s, sigmaH=sigmaH_s)
