import numpy as np
import matplotlib.pyplot as plt
import qutip
from matplotlib.patches import FancyArrowPatch  # Needed in the class below
from mpl_toolkits.mplot3d import proj3d         # Needed in the class below
# from mpl_toolkits.mplot3d import Axes3D         # Needed for coordinate system below

# NOT SURE IF THIS IS NEEDED
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    # Needed because of this: https://github.com/matplotlib/matplotlib/issues/21688
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def paths(alpha_0, beta_0, flow1=True, nr_steps=1000):
    # Related to interaction strength/time
    theta_minus = 0.1
    theta_pluss = 0.01 #theta_minus/5

    # Length of simulation/Number of steps in simulation
    N = nr_steps

    # State alpha |0> + beta |1>
    alpha = np.zeros(N+1,dtype=complex)
    beta  = np.zeros(N+1,dtype=complex)
    alpha[0] = alpha_0
    beta[0]  = beta_0

    if flow1:
        for k in range(N):
            # H_int^+ interaction
            c_00 = alpha[k]
            c_01 = -1j*beta[k]*np.sin(theta_pluss/2)
            c_10 = beta[k]*np.cos(theta_pluss/2)
            c_11 = 0
            p_up = 0.5 - np.imag(alpha[k]*np.conj(beta[k]))*np.sin(theta_pluss/2)

            alpha[k+1] = (c_00+c_01)/np.sqrt(2*p_up)
            beta[k+1]  = (c_10+c_11)/np.sqrt(2*p_up)

    else:
        for k in range(N):
            # H_int^- interaction    
            c_00 = alpha[k+1]*np.cos(theta_minus/2)
            c_01 = 0
            c_10 = beta[k+1]
            c_11 = -1j*alpha[k+1]*np.sin(theta_minus/2)
            p_up = 0.5 - np.imag(beta[k+1]*np.conj(alpha[k+1]))*np.sin(theta_minus/2)

            alpha[k+2] = (c_00-c_01)/np.sqrt(2*(1-p_up))
            beta[k+2]  = (c_10-c_11)/np.sqrt(2*(1-p_up))

    # Convert alpha and beta to Bloch vector
    bloch_vec = np.zeros((3,N+1))#,dtype=complex)
    bloch_vec[2,:] = np.real(2*alpha*np.conj(alpha)-1)
    bloch_vec[1,:] = 2*np.imag(beta*np.conj(alpha))
    bloch_vec[0,:] = 2*np.real(alpha*np.conj(beta))
    
    return bloch_vec

# Create the mesh in polar coordinates and compute corresponding Z
r = np.linspace(1., 1.2, 1)
t = np.linspace(0, 2*np.pi, 8)
p = np.linspace(0, np.pi, 8)
t_sphere = np.linspace(0, 2*np.pi, 25)    # For plotting surface
p_sphere = np.linspace(0, np.pi, 25)

R, T, P = np.meshgrid(r, t, p)
T_sphere, P_sphere = np.meshgrid(t_sphere, p_sphere)  # For plotting surface

# Express the mesh in the cartesian system
X, Y, Z = R*np.cos(T)*np.sin(P), R*np.sin(T)*np.sin(P), R*np.cos(P)
X_sphere, Y_sphere, Z_sphere = np.cos(T_sphere)*np.sin(P_sphere), np.sin(T_sphere)*np.sin(P_sphere), np.cos(P_sphere) # For plotting surface

# """
# Streamlines does not appear to exist in 3D. Trying lineplots instead.
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
sphere = qutip.Bloch(fig=fig, axes=ax)
sphere.make_sphere()
# ax.streamplot(X, Y, Z, Z/2, -Z/2, (Y - X)/2)    # Does not exist(?)
line1 = paths(1./np.sqrt(2), 1./np.sqrt(2), nr_steps=10000)
# print(line1)
ax.plot(line1[0,:],line1[1,:],line1[2,:])
# plt.show()
# """

# Figure 1/Flow 1
fig1 = plt.figure()
ax11 = fig1.add_subplot(121,projection='3d')
ax12 = fig1.add_subplot(122,projection='3d')
sphere1 = qutip.Bloch(fig=fig1, axes=ax11)
sphere1.make_sphere()

# Figure 2/Flow 2
fig2 = plt.figure()
ax21 = fig2.add_subplot(121,projection='3d')
ax22 = fig2.add_subplot(122,projection='3d')
sphere2 = qutip.Bloch(fig=fig2, axes=ax21)
sphere2.make_sphere()

# Plot the surface
# ax11.plot_surface(X_sphere, Y_sphere, Z_sphere, color='#FFDDDD', alpha=0.2)  # Stolen from https://qutip.org/docs/4.0.2/modules/qutip/bloch.html
# ax11.plot_wireframe(X_sphere, Y_sphere, Z_sphere, color='gray', alpha=0.2)   # Stolen from https://qutip.org/docs/4.0.2/modules/qutip/bloch.html
# ax11.plot(1.0 * np.cos(t_sphere), 1.0 * np.sin(t_sphere), zs=0, zdir='z', lw=1, color='gray')
# ax11.plot(1.0 * np.cos(t_sphere), 1.0 * np.sin(t_sphere), zs=0, zdir='x', lw=1, color='gray')

ax11.quiver(X, Y, Z, -Y*X/2, (-Z+1-Y*Y)/2, (Y*(1 - Z))/2, length=0.3)#, headwidth=5.)
# ax11.axis('off') # https://stackoverflow.com/questions/9295026/how-to-remove-axis-legends-and-white-padding

# Plot the surface
# ax12.plot_surface(X_sphere, Y_sphere, Z_sphere, color='#FFDDDD', alpha=0.2)
#
# ax12.plot([-1.2,1.2], [0,0], [0,0], alpha=0.4, color='black')
# ax12.plot([0,0], [-1.2,1.2], [0,0], alpha=0.4, color='black')
# ax12.plot([0,0], [0,0], [-1.2,1.2], alpha=0.4, color='black')
#
# If we want arrows on coordinate system: https://stackoverflow.com/questions/57015852/is-there-a-way-to-plot-a-3d-cartesian-coordinate-system-with-matplotlib
# Here we create the arrows:
arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
#
a = Arrow3D([-1.2, 1.2], [0, 0], [0, 0], **arrow_prop_dict, color='k', alpha=0.4)
ax12.add_artist(a)
a = Arrow3D([0, 0], [-1.2, 1.2], [0, 0], **arrow_prop_dict, color='k', alpha=0.4)
ax12.add_artist(a)
a = Arrow3D([0, 0], [0, 0], [-1.2, 1.2], **arrow_prop_dict, color='k', alpha=0.4)
ax12.add_artist(a)
# Give them a name:
ax12.text(1.3, 0, 0, r'$x$')
ax12.text(0, 1.3, 0, r'$y$')
ax12.text(0, 0, 1.3, r'$z$')
#
ax12.quiver(X, Y, Z, Z/2, -Z/2, (Y - X)/2, length=0.3)
ax12.set_axis_off() # Remove background
# ax12.quiver(X, Y, Z, -Z/2, -Z/2, (Y + X)/2, length=0.2)


ax21.quiver(X, Y, Z, -Z/2, -Z/2, (Y + X)/2, length=0.2)
# Here we create the arrows:
arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
#
a = Arrow3D([-1.2, 1.2], [0, 0], [0, 0], **arrow_prop_dict, color='k', alpha=0.4)
ax22.add_artist(a)
a = Arrow3D([0, 0], [-1.2, 1.2], [0, 0], **arrow_prop_dict, color='k', alpha=0.4)
ax22.add_artist(a)
a = Arrow3D([0, 0], [0, 0], [-1.2, 1.2], **arrow_prop_dict, color='k', alpha=0.4)
ax22.add_artist(a)
# Give them a name:
ax22.text(1.3, 0, 0, r'$x$')
ax22.text(0, 1.3, 0, r'$y$')
ax22.text(0, 0, 1.3, r'$z$')
#
# ax22.quiver(X, Y, Z, Z/2, -Z/2, (Y - X)/2, length=0.3)
ax22.quiver(X, Y, Z, -Z/2, -Z/2, (Y + X)/2, length=0.2)
ax22.set_axis_off() # Remove background

plt.show()
# """

# # Attempt 4
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d', polar=True)

# radii = np.linspace(0.5,1,10)
# thetas = np.linspace(0,2*np.pi,20)
# phis = np.linspace(0,np.pi,20)
# phi, theta, r = np.meshgrid(phis, thetas, radii)

# ax.quiver(phi, theta, r, dr * np.cos(theta) - dt * np.sin (theta), dr * np.sin(theta) + dt * np.cos(theta), r)

# plt.show()

# # Attempt 3
# radii = np.linspace(0.5,1,10)
# thetas = np.linspace(0,2*np.pi,20)
# theta, r = np.meshgrid(thetas, radii)

# dr = 1
# dt = 1

# f = plt.figure()
# ax = f.add_subplot(111, polar=True)
# ax.quiver(theta, r, dr * np.cos(theta) - dt * np.sin (theta), dr * np.sin(theta) + dt * np.cos(theta))

# plt.show()

# # Attempt 2
# fig = plt.figure()
# ax = Axes3D(fig, auto_add_to_figure=False, azim=-40, elev=30)
# fig.add_axes(ax)

# sphere = qutip.Bloch(fig=fig, axes=ax)
# sphere.render(sphere.fig, sphere.axes)

# new_arrow = qutip.bloch.Arrow3D(xs=[1, 1], ys=[0, .5], zs=[0, .5],
#                     mutation_scale=b.vector_mutation,
#                     lw=b.vector_width, arrowstyle=b.vector_style, color='blue')
# sphere.axes.add_artist(new_arrow)

# # Attempt 1
# ax = plt.figure().add_subplot(projection='3d')

# # Make the grid
# x, y, z = np.meshgrid(np.arange(-1, 1, 0.2),
#                       np.arange(-1, 1, 0.2),
#                       np.arange(-1, 1, 0.2))

# # Make the direction data for the arrows
# u = z
# v = -z
# w = y-x

# ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)

# plt.show()