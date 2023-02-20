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

def paths(alpha_0, beta_0, flow=0, nr_steps=1000):
    """
    Compute path of two-level system (TLS) interacting with another TLS
    with interaction Hamiltonian 
    H_+=(theta_+/4)*(sigma_x \otimes sigma_x + sigma_y \otimes sigma_y)
    or
    H_-=(theta_-/4)*(sigma_x \otimes sigma_x - sigma_y \otimes sigma_y)
    from initial state alpha_0 |0> + beta_0 |1>. We use the 
    representation |0> = (1 0), |1> = (0 1), sigma_z |0> = |0>,
    sigma_z |1> = -|1>.
    After each time translation, the environment is measured in the
    basis { |x_+>, |x_-> } (which is a weak measurement) to create a
    path.

    Parameters
    ----------
    alpha_0 : numpy.complex128
        Inital (complex) value for |0>.
    beta_0 : numpy.complex128
        Inital (complex) value for |1>.
    flow : int
        Choice of interaction Hamiltonian and measurement. 
        Flow 1: H_+ and measure |x_+>   Flow 2: H_+ and measure |x_->
        Flow 3: H_- and measure |x_+>   Flow 2: H_- and measure |x_->
    nr_steps : int
        Number of steps/measurements. This is equivalent to the amount
        of time that we simulate.
    """
    # Related to interaction strength/time
    theta_minus = 0.1
    theta_pluss = 0.01 #theta_minus/5

    # Length of simulation/Number of steps in simulation
    N = nr_steps

    # State alpha |0> + beta |1>
    alpha = np.zeros(N+1,dtype=complex)
    beta  = np.zeros(N+1,dtype=complex)
    alpha[0] = alpha_0  # Initial state
    beta[0]  = beta_0   # Initial state

    # Type of interaction and measurement
    if flow == 0:
        for k in range(N):
            # H_int^+ interaction
            c_00 = alpha[k]
            c_01 = -1j*beta[k]*np.sin(theta_pluss/2)
            c_10 = beta[k]*np.cos(theta_pluss/2)
            c_11 = 0
            # Measure up in x
            p_up = 0.5 - np.imag(alpha[k]*np.conj(beta[k]))*np.sin(theta_pluss/2)
            alpha[k+1] = (c_00+c_01)/np.sqrt(2*p_up)
            beta[k+1]  = (c_10+c_11)/np.sqrt(2*p_up)
    
    elif flow == 1:
        for k in range(N):
            # H_int^+ interaction
            c_00 = alpha[k]
            c_01 = -1j*beta[k]*np.sin(theta_pluss/2)
            c_10 = beta[k]*np.cos(theta_pluss/2)
            c_11 = 0
            # Measure down in x
            p_up = 0.5 - np.imag(alpha[k]*np.conj(beta[k]))*np.sin(theta_pluss/2)
            alpha[k+1] = (c_00-c_01)/np.sqrt(2*(1-p_up))
            beta[k+1]  = (c_10-c_11)/np.sqrt(2*(1-p_up))
            

    elif flow == 2:
        for k in range(N):
            # H_int^- interaction    
            c_00 = alpha[k]*np.cos(theta_minus/2)
            c_01 = 0
            c_10 = beta[k]
            c_11 = -1j*alpha[k]*np.sin(theta_minus/2)
            # Measure up in x
            p_up = 0.5 - np.imag(beta[k]*np.conj(alpha[k]))*np.sin(theta_minus/2)
            alpha[k+1] = (c_00+c_01)/np.sqrt(2*p_up)
            beta[k+1]  = (c_10+c_11)/np.sqrt(2*p_up)


    else:
        for k in range(N):
            # H_int^- interaction    
            c_00 = alpha[k]*np.cos(theta_minus/2)
            c_01 = 0
            c_10 = beta[k]
            c_11 = -1j*alpha[k]*np.sin(theta_minus/2)
            # Measure down in x
            p_up = 0.5 - np.imag(beta[k]*np.conj(alpha[k]))*np.sin(theta_minus/2)
            alpha[k+1] = (c_00-c_01)/np.sqrt(2*(1-p_up))
            beta[k+1]  = (c_10-c_11)/np.sqrt(2*(1-p_up))

    # Convert alpha and beta to Bloch vector
    bloch_vec = np.zeros((3,N+1))
    bloch_vec[2,:] = np.real(2*alpha*np.conj(alpha)-1)
    bloch_vec[1,:] = 2*np.imag(beta*np.conj(alpha))
    bloch_vec[0,:] = 2*np.real(alpha*np.conj(beta))
    
    return bloch_vec

def plot_vector_flow(X, Y, Z, U, V, W):
    """
    Function for plotting the flow of the Bloch vector on the Bloch
    sphere for a given interaction Hamiltonian and measurement. Both
    the meshgrid of point where to plot (X, Y, Z) and pre-calculated
    flow (U, V, W) must be given.

    Parameters
    ----------
    X : numpy.ndarray
        Meshgrid in x-direction of point where we plot the vector flow.
    Y : numpy.ndarray
        Meshgrid in y-direction of point where we plot the vector flow.
    Z : numpy.ndarray
        Meshgrid in z-direction of point where we plot the vector flow.
    U : numpy.ndarray
        Meshgrid of the vector flow in x-direction.
    V : numpy.ndarray
        Meshgrid of the vector flow in y-direction.
    W : numpy.ndarray
        Meshgrid of the vector flow in z-direction.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(121,projection='3d')
    ax2 = fig.add_subplot(122,projection='3d')

    # Fixing Bloch sphere: x- and y-label is interchaged in source code
    sphere = qutip.Bloch(fig=fig, axes=ax1)
    sphere.xlabel = ['$\\left|y_+\\right>$', ' ']
    sphere.xlpos = [-1.3,-1.3]
    sphere.ylabel = ['$\\left|x_+\\right>$', ' ']
    sphere.ylpos = [1.3,-1.3]
    sphere.make_sphere()

    # Vector flow with Bloch sphere
    # ax1.quiver(X, Y, Z, -Y*X/2, (-Z + 1 - Y*Y)/2, (Y*(1 - Z))/2, length=0.3)
    ax1.quiver(X, Y, Z, U, V, W, length=0.3)
    # If we want arrows on coordinate system: https://stackoverflow.com/questions/57015852/is-there-a-way-to-plot-a-3d-cartesian-coordinate-system-with-matplotlib
    # Here we create the arrows:
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
    
    # Vector flow without Bloch sphere
    a = Arrow3D([-1.2, 1.2], [0, 0], [0, 0], **arrow_prop_dict, color='k', alpha=0.4)
    ax2.add_artist(a)
    a = Arrow3D([0, 0], [-1.2, 1.2], [0, 0], **arrow_prop_dict, color='k', alpha=0.4)
    ax2.add_artist(a)
    a = Arrow3D([0, 0], [0, 0], [-1.2, 1.2], **arrow_prop_dict, color='k', alpha=0.4)
    ax2.add_artist(a)
    # Give them a name:
    ax2.text(1.3, 0, 0, r'$x$')
    ax2.text(0, 1.3, 0, r'$y$')
    ax2.text(0, 0, 1.3, r'$z$')

    ax2.quiver(X, Y, Z, U, V, W, length=0.3)
    ax2.set_axis_off() # Remove background

    return fig, ax1, ax2

def plot_streamlines(alpha_0, beta_0, figure=False, axis=False, plot_sphere=True, flow=0, nr_steps=1000, color=False, label=False):
    """
    Function for plotting the flow of the Bloch vector on the Bloch
    sphere for a given interaction Hamiltonian and measurement. Both
    the meshgrid of point where to plot (X, Y, Z) and pre-calculated
    flow (U, V, W) must be given.

    Parameters
    ----------
    alpha_0 : numpy.complex128
        Inital (complex) value for |0>.
    beta_0 : numpy.complex128
        Inital (complex) value for |1>.
    figure : matplotlib.figure.Figure
        Already existing figure to overwrite or plot more on
    axis : matplotlib.axes._subplots.Axes3DSubplot
        Already existing axis object to overwrite or plot more on
    plot_sphere : bool
        Option to plot a (new) Bloch sphere
    flow : int
        Choice of interaction Hamiltonian and measurement. 
        Flow 1: H_+ and measure |x_+>   Flow 2: H_+ and measure |x_->
        Flow 3: H_- and measure |x_+>   Flow 2: H_- and measure |x_->
    nr_steps : int
        Number of steps/measurements. This is equivalent to the amount
        of time that we simulate.
    color : str
        Choice of linecolor
    label : str
        Label for the plotted line
    """
    # Streamlines does not appear to exist in 3D. Trying lineplots instead.
    if not figure:
        fig = plt.figure()
    else:
        fig = figure

    if not axis:
        ax = fig.add_subplot(111,projection='3d')
    else:
        ax = axis

    if plot_sphere:
        # Fixing Bloch sphere: x- and y-label is interchaged in source code
        # https://groups.google.com/g/qutip/c/LPt0niROuPA
        sphere = qutip.Bloch(fig=fig, axes=ax)
        sphere.xlabel = ['$\\left|y_+\\right>$', ' ']
        sphere.xlpos = [-1.3,-1.3]
        sphere.ylabel = ['$\\left|x_+\\right>$', ' ']
        sphere.ylpos = [1.3,-1.3]
        sphere.make_sphere()

    line = paths(alpha_0, beta_0, flow=flow, nr_steps=nr_steps)
    
    if not color and not label:
        ax.plot(line[0,:],line[1,:],line[2,:])
    elif not color:
        ax.plot(line[0,:],line[1,:],line[2,:], label=label)
    elif not label:
        ax.plot(line[0,:],line[1,:],line[2,:], color=color)
    else:
        ax.plot(line[0,:],line[1,:],line[2,:], color=color, label=label)

    return fig, ax

def dN_k_dtheta(X, Y, Z, flow=0):
    """
    Function returning the Jacobi matrix of the coordinate 
    transformation from cartesian to spherical coordinates. The 
    expressions used in this implementation are derived assuming radius
    equal to 1, i.e. x^2 + y^2 + z^2 = 1.

    Parameters
    ----------
    X : numpy.ndarray
        Meshgrid in x-direction of point where we plot the vector flow.
    Y : numpy.ndarray
        Meshgrid in y-direction of point where we plot the vector flow.
    Z : numpy.ndarray
        Meshgrid in z-direction of point where we plot the vector flow.
    flow : int
        Choice of interaction Hamiltonian and measurement. 
        Flow 1: H_+ and measure |x_+>   Flow 2: H_+ and measure |x_->
        Flow 3: H_- and measure |x_+>   Flow 2: H_- and measure |x_->
    """
    if flow == 0:
        return -X*Y/2, (1 - Z - Y*Y)/2, Y*(1 - Z)/2
    
    elif flow == 1:
        return X*Y/2, -(1 - Z - Y*Y)/2, -Y*(1 - Z)/2    

    elif flow == 2:
        return X*Y/2, -(1 + Z - Y*Y)/2, Y*(1 + Z)/2

    else:
        return -X*Y/2, (1 + Z - Y*Y)/2, -Y*(1 + Z)/2

def coor_trans_from_cartesian_to_sphere_coor(x, y, z):
    """
    Function returning the coordinate transformation from cartesian to 
    spherical coordinates. The expressions used in this implementation
    are derived assuming radius equal to 1, i.e. x^2 + y^2 + z^2 = 1.

    Parameters
    ----------
    x : float
        x-component of Bloch vector
    y : float
        y-component of Bloch vector
    z : float
        z-component of Bloch vector
    """
    denominator = x*x + y*y
    if denominator == 0:
        return 0, np.arccos(z)
    else:
        return np.arccos( x/np.sqrt(denominator) ), np.arccos(z)

def jacobian_from_cartesian_to_sphere_coor(x, y, z):
    """
    Function returning the Jacobi matrix of the coordinate 
    transformation from cartesian to spherical coordinates. The 
    expressions used in this implementation are derived assuming radius
    equal to 1, i.e. x^2 + y^2 + z^2 = 1.

    Parameters
    ----------
    x : float
        x-component of Bloch vector
    y : float
        y-component of Bloch vector
    z : float
        z-component of Bloch vector
    """
    jacobi = np.zeros((2,3))
    denominator = x*x + y*y
    
    if denominator == 0:
        return jacobi
    
    else:
        jacobi[0,0] = x*z/np.sqrt(denominator)
        jacobi[0,1] = y*z/np.sqrt(denominator)
        jacobi[0,2] = -np.sqrt(denominator)

        jacobi[1,0] = -y/denominator
        jacobi[1,1] = x/denominator

    return jacobi

def winkel_tripel_proj(phi, lam, cos_phi_0=2/np.pi):
    """
    Function returning the Winkel tripel map projection.

    Parameters
    ----------
    phi : float
        Latitude (angle from 'equator')
    lam : float
        Longitude (angle from central meridian, i.e. the defined east
        west midpoint)
    cos_phi_0 : float
        phi_0 is the standard parallel for the equirectangular 
        projection. That is a constant defining where the scale of the 
        projection is true
    """    
    alpha = np.arccos(np.cos(phi)*np.cos(lam/2))

    if np.sinc(alpha) == 0:
        raise ValueError("Projection will not work as angle is to large.")

    x = 0.5*( 2*np.cos(phi)*np.sin(lam/2)/np.sinc(alpha) + lam*cos_phi_0 )
    y = 0.5*( np.sin(phi)/np.sinc(alpha) + phi )

    return x, y

def jacobian_winkel_tripel(phi, lam, cos_phi_0=2/np.pi):
    """
    Function returning the Jacobi matrix of the Winkel tripel map 
    projection.

    Parameters
    ----------
    phi : float
        Latitude (angle from 'equator')
    lam : float
        Longitude (angle from central meridian, i.e. the defined east
        west midpoint)
    cos_phi_0 : float
        phi_0 is the standard parallel for the equirectangular 
        projection. That is a constant defining where the scale of the 
        projection is true
    """    
    jacobi = np.zeros((2,2))
    c = 1 - np.cos(phi)*np.cos(phi)*np.cos(lam/2)*np.cos(lam/2)
    alpha = np.arccos(np.cos(phi)*np.cos(lam/2))
    
    if (np.sinc(alpha) == 0) or (c == 0):
        raise ValueError("Projection will not work as angle is to large.")

    jacobi[0,0] = np.sin(lam)*np.sin(2*phi) / (4*c) - np.sin(lam/2)*np.sin(phi) / (np.sqrt(c)*np.sinc(alpha))
    jacobi[0,1] = 0.5*( np.cos(phi)*np.cos(phi)*np.sin(lam/2)*np.sin(lam/2) / c - np.cos(phi)*np.cos(lam/2)*np.sin(phi)*np.sin(phi) / (np.sqrt(c)*np.sinc(alpha)) + cos_phi_0 )
    jacobi[1,0] = 0.5*( np.sin(phi)*np.sin(phi)*np.cos(lam/2)*np.cos(lam/2) / c - (1 - np.cos(lam/2)*np.cos(lam/2))*np.cos(phi) / (np.sqrt(c)*np.sinc(alpha)) + 1 )
    jacobi[1,1] = 0.125*( np.sin(2*phi)*np.cos(lam/2) / c - np.sin(phi)*np.cos(phi)*np.cos(phi)*np.sin(lam) / (np.sqrt(c)*np.sinc(alpha)) )

    return jacobi

# Create the mesh in polar coordinates and compute corresponding Z
r = np.linspace(1., 1.2, 1)
t = np.linspace(0, 2*np.pi, 15)
p = np.linspace(0, np.pi, 15)
t_sphere = np.linspace(0, 2*np.pi, 15)    # For plotting surface
p_sphere = np.linspace(0.01, np.pi, 15)   # NB! Start at 0.01 because Winkel triple projection does not 'like' all angles.

R, T, P = np.meshgrid(r, t, p)
T_sphere, P_sphere = np.meshgrid(t_sphere, p_sphere)  # For plotting surface

# Express the mesh in the cartesian system
X, Y, Z = R*np.cos(T)*np.sin(P), R*np.sin(T)*np.sin(P), R*np.cos(P)
X_sphere, Y_sphere, Z_sphere = np.cos(T_sphere)*np.sin(P_sphere), np.sin(T_sphere)*np.sin(P_sphere), np.cos(P_sphere) # For plotting surface

U1, V1, W1 = dN_k_dtheta(X, Y, Z, flow=0)
U2, V2, W2 = dN_k_dtheta(X, Y, Z, flow=1)
U3, V3, W3 = dN_k_dtheta(X, Y, Z, flow=2)
U4, V4, W4 = dN_k_dtheta(X, Y, Z, flow=3)

# # Plot vector flow with the above flow
fig_H_up_x_up, ax1_H_up_x_up, ax2_H_up_x_up             = plot_vector_flow(X, Y, Z, U1, V1, W1)
# fig_H_up_x_down, ax1_H_up_x_down, ax2_H_up_x_down       = plot_vector_flow(X, Y, Z, U2, V2, W2)
# fig_H_down_x_up, ax1_H_down_x_up, ax2_H_down_x_up       = plot_vector_flow(X, Y, Z, U3, V3, W3)
# fig_H_down_x_down, ax1_H_down_x_down, ax2_H_down_x_down = plot_vector_flow(X, Y, Z, U4, V4, W4)

# # Give title to all figures
fig_H_up_x_up.suptitle(r"Vector flow for $H_+$ interaction and $|x_+\rangle$ measurement")
# fig_H_up_x_down.suptitle(r"Vector flow for $H_+$ interaction and $|x_-\rangle$ measurement")
# fig_H_down_x_up.suptitle(r"Vector flow for $H_-$ interaction and $|x_+\rangle$ measurement")
# fig_H_down_x_down.suptitle(r"Vector flow for $H_-$ interaction and $|x_-\rangle$ measurement")

# # Plot streamlines of H_+ and |x_+> starting from |y_->
# fig_stream, ax_stream = plot_streamlines(1./np.sqrt(2), -1j/np.sqrt(2), flow=0, nr_steps=10000, color="red", label=r"$H_+$ and $|x_+\rangle$")

# # Plot streamlines of all flows starting from |x_+>
# fig_stream, ax_stream = plot_streamlines(1./np.sqrt(2), 1./np.sqrt(2), flow=0, nr_steps=10000, color="red", label=r"$H_+$ and $|x_+\rangle$")
# fig_stream, ax_stream = plot_streamlines(1./np.sqrt(2), 1./np.sqrt(2), figure=fig_stream, axis=ax_stream, plot_sphere=False, flow=1, nr_steps=10000, color="green", label=r"$H_+$ and $|x_-\rangle$")
# fig_stream, ax_stream = plot_streamlines(1./np.sqrt(2), 1./np.sqrt(2), figure=fig_stream, axis=ax_stream, plot_sphere=False, flow=2, nr_steps=10000, color="blue", label=r"$H_-$ and $|x_+\rangle$")
# fig_stream, ax_stream = plot_streamlines(1./np.sqrt(2), 1./np.sqrt(2), figure=fig_stream, axis=ax_stream, plot_sphere=False, flow=3, nr_steps=10000, color="black", label=r"$H_-$ and $|x_-\rangle$")

# # Add legend with correct labels (it adds X, Y and Z for some reason)
# handles, labels = ax_stream.get_legend_handles_labels()
# ax_stream.legend(handles[3:], labels[3:], loc="best")
# fig_stream.suptitle(r"Streamlines starting in $|x_+\rangle$")

# # Plot streamlines of a flow starting from a point (toy model)
# alpha = np.cos(np.pi/4-0.5)*np.exp(1j*np.pi/4)
# beta  = np.sin(np.pi/4-0.5)*np.exp(-1j*np.pi/4)
# fig, ax = plot_streamlines(alpha, beta, flow=1, nr_steps=20000, color="red")
# fig.suptitle(rf"Streamline starting in ({np.real(alpha):0.2}+{np.imag(alpha):0.2}i)$|0\rangle+$({np.real(beta):0.2}+{np.imag(beta):0.2}i)$|1\rangle$")

# # Plot stereographic projection (think something is wrong here)
# r_stereo = np.linspace(0,1,10)
# t_stereo = np.linspace(0,2*np.pi,10)
# R_stereo, T_stereo = np.meshgrid(r_stereo, t_stereo)
# X_stereo, Y_stereo = R_stereo*np.cos(T_stereo), R_stereo*np.sin(T_stereo)
# U_stereo = 0*X_stereo
# V_stereo = U_stereo+0.5

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.quiver(X_stereo, Y_stereo, U_stereo, V_stereo)

plt.show()

"""
# Idea for projection
U_winkel, V_winkel, W_winkel = dN_k_dtheta(X_sphere, Y_sphere, Z_sphere, flow=0)
x_prime = np.zeros_like(X_sphere)
y_prime = np.zeros_like(Y_sphere)
new_u = np.zeros_like(U_winkel)
new_v = np.zeros_like(V_winkel)

idx_i = new_u.shape[0]
idx_j = new_u.shape[1]

for i in range(idx_i):
    for j in range(idx_j):
        theta, phi = coor_trans_from_cartesian_to_sphere_coor(X_sphere[i,j], Y_sphere[i,j], Z_sphere[i,j])
        long, lat = theta - np.pi, phi - np.pi/2
        
        push_forward_sphere = jacobian_from_cartesian_to_sphere_coor(X_sphere[i,j], Y_sphere[i,j], Z_sphere[i,j])
        push_forward_winkel = jacobian_winkel_tripel(lat, long)
        tangent = np.array([U_winkel[i,j], V_winkel[i,j], W_winkel[i,j]])
        tan_sphere = np.matmul(push_forward_sphere, tangent)
        new_u[i,j], new_v[i,j] = np.matmul(push_forward_winkel, tan_sphere)

        x_prime[i,j], y_prime[i,j] = winkel_tripel_proj(lat, long)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.quiver(x_prime,y_prime,new_u,new_v)
"""