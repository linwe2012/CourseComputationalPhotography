from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

def plt_circle(params):
    theta = np.arange(0, 8, 0.1)
    y = np.add(np.multiply(np.sin(theta), params[2]), params[1])
    x = np.add(np.multiply(np.cos(theta), params[2]), params[0])
    plt.plot(x,y, 'c')


def plt_eclipse_sphere(ax, params):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    A = params[0]
    B = params[1]
    C = params[2]
    x = A * np.sin(u) * np.cos(v)
    y = B * np.sin(u) * np.sin(v)
    z = C * np.cos(u)
    ax.plot_wireframe(x, y, z)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.3, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

def load_data(filename):
    """
    Load parameters & datapoints from C++ output
    """
    with open(filename, 'rt') as f:
        lines = f.readlines()
        lines = list(filter(lambda line: len(line)>1, lines))
        def linesep(line:str):
            prefiltered = filter(lambda x: len(x) > 0 and x != ' ', line.strip().split(' '))
            return list(map(lambda x: float(x), prefiltered))
        params = linesep(lines[0])
        dots = list(map(lambda x: linesep(x), lines[1:]))
        dots = list(zip(*dots))

        return (params, dots)


"""
Draw Circles
"""
params, dots = load_data('circle_result.txt')
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(dots[0], dots[1])
plt_circle(params)


"""
Plot 3d eclispe
"""
params, dots = load_data('ecclipse753_result.txt')
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(dots[0], dots[1], dots[2], c=dots[2], cmap='viridis')#cmap='Greens')
plt_eclipse_sphere(ax, params)
plt.show()