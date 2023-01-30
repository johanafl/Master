import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import time
import qutip    # https://qutip.org/docs/latest/guide/guide-bloch.html
                # https://qutip.org/docs/latest/guide/guide-states.html
                # https://stackoverflow.com/questions/50321057/how-to-show-qutip-bloch-sphere-as-onset-figure-on-the-bigger-plot-with-other-dat

def main():
    rng = np.random.default_rng()
    # rng = np.random.default_rng(1234)

    # Choose random point on Bloch sphere
    theta = rng.uniform(low=0.0, high=np.pi)
    phi   = rng.uniform(low=0.0, high=2*np.pi)

    # Related to interaction strength/time
    theta_minus = 0.1
    theta_pluss = 0.01 #theta_minus/5

    # Half the length of simulation/half the number of steps in simulation
    N = 1050

    # State alpha |0> + beta |1>
    alpha = np.zeros(2*N+1,dtype=complex)
    beta  = np.zeros(2*N+1,dtype=complex)
    alpha[0] = np.cos(theta/2)*np.exp(-1j*phi/2) #np.cos(np.pi/4) #np.cos(np.pi/2+0.1)
    beta[0]  = np.sin(theta/2)*np.exp(1j*phi/2) #np.sin(np.pi/4) #np.sin(np.pi/2+0.1)
    
    """
    for k in range(2*N):
        c_00 = alpha[k]
        c_01 = -1j*beta[k]*np.sin(theta_pluss/2)
        c_10 = beta[k]*np.cos(theta_pluss/2)
        c_11 = 0
        p_up = 0.5 - np.imag(alpha[k]*np.conj(beta[k]))*np.sin(theta_pluss/2)

        alpha[k+1] = (c_00+c_01)/np.sqrt(2*p_up)
        beta[k+1]  = (c_10+c_11)/np.sqrt(2*p_up)

    # NB! Testing the loop above.
    """
    for k in range(N):
        # H_int^+ interaction
        c_00 = alpha[2*k]
        c_01 = -1j*beta[2*k]*np.sin(theta_pluss/2)
        c_10 = beta[2*k]*np.cos(theta_pluss/2)
        c_11 = 0
        p_up = 0.5 - np.imag(alpha[2*k]*np.conj(beta[2*k]))*np.sin(theta_pluss/2)

        if rng.uniform() < np.sqrt(p_up):
            alpha[2*k+1] = (c_00+c_01)/np.sqrt(2*p_up)
            beta[2*k+1]  = (c_10+c_11)/np.sqrt(2*p_up)
        else:
            alpha[2*k+1] = (c_00-c_01)/np.sqrt(2*(1-p_up))
            beta[2*k+1]  = (c_10-c_11)/np.sqrt(2*(1-p_up))

        # H_int^- interaction    
        c_00 = alpha[2*k+1]*np.cos(theta_minus/2)
        c_01 = 0
        c_10 = beta[2*k+1]
        c_11 = -1j*alpha[2*k+1]*np.sin(theta_minus/2)
        p_up = 0.5 - np.imag(beta[2*k+1]*np.conj(alpha[2*k+1]))*np.sin(theta_minus/2)

        if rng.uniform() < np.sqrt(p_up):
            alpha[2*k+2] = (c_00+c_01)/np.sqrt(2*p_up)
            beta[2*k+2]  = (c_10+c_11)/np.sqrt(2*p_up)
        else:
            alpha[2*k+2] = (c_00-c_01)/np.sqrt(2*(1-p_up))
            beta[2*k+2]  = (c_10-c_11)/np.sqrt(2*(1-p_up))
    # """

    # Convert alpha and beta to Bloch vector
    bloch_vec = np.zeros((3,2*N+1))#,dtype=complex)
    # bloch_vec[2,:] = np.real(2*alpha*np.conj(alpha)-1)
    # bloch_vec[1,:] = 2*np.imag(beta*np.conj(alpha))
    # bloch_vec[0,:] = 2*np.real(alpha*np.conj(beta))
    bloch_vec[2,:] = np.real(1 - 2*alpha*np.conj(alpha))
    bloch_vec[1,:] = 2*np.imag(alpha*np.conj(beta))
    bloch_vec[0,:] = 2*np.real(alpha*np.conj(beta))

    # print(bloch_vec)
    # norm = bloch_vec[0,:]**2+bloch_vec[1,:]**2+bloch_vec[2,:]**2
    # norm = [int(x) for x in norm]
    # not_normal = np.where(norm!=1)[0]
    # x = np.where(norm==1)[0]
    # print(len(not_normal))
    # print(norm, len(norm))
    # print(not_normal, len(not_normal))

    # """
    # Animation on Bloch sphere
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False, azim=-40, elev=30)
    fig.add_axes(ax)
    sphere = qutip.Bloch(fig=fig, axes=ax)

    def animate(i):
        sphere.clear()
        sphere.add_points(bloch_vec[:,:2*i+1:2]) # Only use every second point
        #    sphere.add_points(bloch_vec[:,:i+1])   # Use every point
        sphere.make_sphere()
        return ax

    def init():
        return ax

    # Only use every second point
    # ani = animation.FuncAnimation(fig, animate, np.arange(N+1),
    #                                 init_func=init, blit=False, repeat=False, interval=1)

    
    # # Use every point
    ani = animation.FuncAnimation(fig, animate, np.arange(2*N+1),
                                  init_func=init, blit=False, repeat=False)

    # ani.save("test.mp4", fps=60)
    ani.save(f"bloch_sphere_{phi=}_{theta=}_tm={theta_minus}_tm={theta_pluss}.mp4", fps=60)
    # ani.save(f'bloch_sphere_{phi=}_{theta=}_tm={theta_minus}_tm={theta_pluss}.mp4', fps=20)
    # # Trying to fix bug: Process keeps running after animation closed/done
    # try:
    # plt.show()
    # except KeyboardInterrupt:
    #     sys.exit()
    # exit()
    # plt.close()
    # fig.close()
    # """

    """
    # Plotting on Bloch sphere
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False, azim=-40, elev=30)
    fig.add_axes(ax)
    sphere = qutip.Bloch(fig=fig, axes=ax)
    sphere.add_points(bloch_vec[:,0])
    sphere.add_points(bloch_vec[:,1:],alpha=0.005)
    sphere.make_sphere()

    # b = qutip.Bloch()
    # b.make_sphere()
    # pnt = [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
    # b.add_points(pnt)
    # b.render()
    plt.show()
    """

    """
    # Earlier idea
    a = rng.uniform(-1,1)
    b = rng.uniform(-1,1)
    c = rng.uniform(-1,1)
    d = rng.uniform(-1,1)

    theta_minus = 0.01
    theta_pluss = theta_minus/5
    N = 1000

    norm = a*a+b*b+c*c+d*d
    sys_state = np.array([a+b*1j,c+d*1j])/norm

    for k in range(N):
        # H_int^+ interaction
        state = np.array([sys_state[0],-1j*sys_state[1]*np.sin(theta_pluss/2),sys_state[1]*np.cos(theta_pluss/2),0])
        p_up = 0.5 - np.imag(sys_state[0]*np.conj(sys_state[1]))*np.sin(theta_pluss/2)

        if rng.uniform() < np.sqrt(p_up):
            sys_state = np.array([state[0]+state[1],state[2]+state[3]])/np.sqrt(2*p_up)
        else:
            sys_state = np.array([state[0]-state[1],state[2]-state[3]])/np.sqrt(2*(1-p_up))

        # H_int^- interaction
        state = np.array([sys_state[0]*np.cos(theta_minus/2),0,sys_state[1],-1j*sys_state[0]*np.sin(theta_minus/2)])
        p_up = 0.5 - np.imag(sys_state[1]*np.conj(sys_state[0]))*np.sin(theta_minus/2)

        if rng.uniform() < np.sqrt(p_up):
            sys_state = np.array([state[0]+state[1],state[2]+state[3]])/np.sqrt(2*p_up)
        else:
            sys_state = np.array([state[0]-state[1],state[2]-state[3]])/np.sqrt(2*(1-p_up))
    """

if __name__ == "__main__":
    main()