import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import robot
import trajectory
import network
import progress
import os

def generate_trajectory(points, step):
    net.reset()
    traj_x = np.zeros((0, 1))
    traj_y = np.zeros((0, 1))
    
    for point in points:
        x_temp, y_temp = trajectory.line(point[0], point[1], point[2], point[3], step=step)
        traj_x = np.concatenate([traj_x, x_temp])
        traj_y = np.concatenate([traj_y, y_temp])
    return traj_x, traj_y

def recurrent_trajectory(net:network.RecurrentNetwork, impulse):
    net.reset()
    trajectory = []
    for t in range(impulse.shape[0]):
        net.step(impulse[t, :], noise=False)
        trajectory.append(net.r)
        
    trajectory = np.array(trajectory)
    return trajectory

def train_recurrent(net:network.RecurrentNetwork, impulse, recurrent_trajectory, t_start):
    net.reset()
    bar = progress.Bar(50)
    
    # Recording
    for t in range(impulse.shape[0]):
        net.step(impulse[t, :, :], noise=True)

        if t < t_start:
            continue
        if(t%5==0):
            net.train_recurrent(target=recurrent_trajectory[t, :, :])   
        bar.progress((t + 1) / impulse.shape[0])
    bar.delete()

def train_recadout(net:network.RecurrentNetwork, impulse, target, t_start, learning_rate, batch_size):
    net.reset()
    bar = progress.Bar(50)
    error = np.zeros((net.No, 1))

    if batch_size > 0:
        samples = np.random.choice(np.arange(t_start, impulse.shape[0]), batch_size)

    for t in range(impulse.shape[0]):
        net.step(impulse[t], noise=True)

        if t < t_start:
            continue
        if batch_size > 0 and not t in samples:
            continue

        error += net.train_readout(target[t, :], learning_rate)
        #error += np.abs(np.absolute(error_t).sum())
        bar.progress((t + 1) / impulse.shape[0])
    bar.delete()
    return (error**2).sum()

def test_net(net:network.RecurrentNetwork, impulse):
    net.reset()
    z = []
    for t in range(impulse.shape[0]):
        net.step(impulse[t], noise=False)
        z.append(net.z)
    return np.array(z)

def plot_error(errors):
    plt.figure(figsize=(15, 10))
    plt.xlabel('trial')
    plt.ylabel('error')
    plt.minorticks_on()
    plt.xlim(0, len(errors[0])-1)
    plt.grid(which='both')
    plt.title('Error')
    for figure in figures:
        plt.plot(errors[figure], label='error figure {0}'.format(figure))

    plt.legend()
    plt.savefig('plots/error.png')
    plt.close()

def plot_recurrent(figures, impulses, recurrent_trajectories, plots = 3):
    for figure in figures:
        plt.figure(figsize=(18, 12))
        plt.minorticks_on()
        plt.xlim(0, impulses[figure].shape[0])
        plt.grid(which='both')

        plots = 3
        for i in range(plots):
            traj = recurrent_trajectory(net, impulses[figure])
            #plt.subplot(plots, 1, i+1)
            plt.plot(recurrent_trajectories[figure][:, i]+2*i, label='trajectory untrained {0}'.format(i))
            plt.plot(traj[:, i]+2*i, label='trained trajectory {0}'.format(i))
                
        plt.legend()
        #plt.show()
        filename = 'plots/recurrent_figure{0}.png'.format(figure)
        plt.savefig(filename)
        plt.close()
        print('Recurrent plot saved as {0}'.format(filename))

def plot_readout(net, figures, impulses, targets, t_start, trial, threshold = 0.9, markersize = 3.):
    for figure in figures:
        points = figures[figure]

        z = test_net(net, impulses[figure])
        target = targets[figure]

        target = target[t_start:]
        target_j1_pos, target_j2_pos = target[:,0][target[:,2]>=threshold], target[:,1][target[:,2]>=threshold]
        target_x_pos, target_y_pos = ik_model.to_cartesian(target_j1_pos / 180 * np.pi, target_j2_pos / 180 * np.pi)

        z = z[t_start:]
        z_j1_pos, z_j2_pos = z[:, 0][z[:,2]>=threshold], z[:, 1][z[:,2]>=threshold]
        z_j1_neg, z_j2_neg = z[:, 0][z[:,2]<threshold], z[:, 1][z[:,2]<threshold]

        z_x_pos, z_y_pos = ik_model.to_cartesian(z_j1_pos / 180 * np.pi, z_j2_pos / 180 * np.pi)
        z_x_neg, z_y_neg = ik_model.to_cartesian(z_j1_neg / 180 * np.pi, z_j2_neg / 180 * np.pi)

        plt.figure(figsize=(8, 12))
        plt.subplot(2, 1, 1)
        plt.xlim(0, 30)
        plt.ylim(-10, 40)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.minorticks_on()
        plt.grid(which='both')
        plt.title('Figure {0}'.format(figure))
        plt.scatter(target_x_pos, target_y_pos, s=markersize, label='target', color='blue')
        plt.scatter(z_x_pos, z_y_pos, s = markersize, label='z: draw', color='green')
        plt.scatter(z_x_neg, z_y_neg, s = markersize, label='z: don\'t draw', color='red')
        plt.legend(loc='upper right')

        plt.subplot(2, 1, 2)
        plt.xlim(0, impulses[figure].shape[0]-t_start)
        plt.ylim(-60, 160)
        plt.xlabel('t')
        plt.ylabel('signal')
        plt.minorticks_on()
        plt.grid(which='both')
        plt.plot(target[:,0], label='$target_{joint1}$')
        plt.plot(target[:,1], label='$target_{joint2}$')
        plt.plot(target[:,2], label='$target_{should\,draw}$')
        plt.plot(z[:,0], label='$z_{joint1}$')
        plt.plot(z[:,1], label='$z_{joint2}$')
        plt.plot(z[:,2], label='$z_{should\,draw}$')
        
        plt.legend(loc='upper right')
        filename = 'readout_figure{0}'.format(figure)
        if figure == 0:
            appendix = '{0}'.format(trial)
            while len(appendix) < 5:
                appendix = '0' + appendix
            plt.savefig('plots/gif/' + filename + '_trial{0}.png'.format(appendix))
        plt.savefig('plots/' + filename + '.png')
        plt.close()
        print('Readout figure saved as {0}'.format(filename))

if not os.path.isdir('plots'):
    os.mkdir('plots')
if not os.path.isdir('plots/gif'):
    os.mkdir('plots/gif')
if not os.path.isdir('data'):
    os.mkdir('data')

limb1 = 25
limb2 = 25
refX = 0
refY = 0

ik_model = robot.Transform_2DoF(limb1, limb2, refX, refY)

figures = {
    # house
    0: [[5, 5, 5, 25],
        [5, 25, 15, 35],
        [15, 35, 25, 25],
        [25, 25, 25, 5],
        [25, 5, 5, 5],
        [5, 5, 25, 25],
        [25, 25, 5, 25],
        [5, 25, 25, 5]],
    # triangle
    1: [[5, 5, 15, 35],
        [15, 35, 25, 5],
        [25, 5, 5, 5]],
    # square
    2: [[5, 5, 5, 35],
        [5, 35, 25, 35],
        [25, 35, 25, 5],
        [25, 5, 5, 5]]
}

Ni = len(figures) # Number of inputs
N = 1200 # Number of recurrent neurons
No = 3 # Number of read-out neurons
tau = 4.#4.#10.0 # Time constant of the neurons
g = 1.5 # Synaptic strength scaling Spectral radius
pc = 0.1 # Connection probability
Io = 0.0001#0.001 # Noise variance
P_plastic = 0.6 # Percentage of neurons receiving plastic synapses

load_filename = 'data/net.pkl'
save_filename = 'data/net.pkl'

if os.path.isfile(load_filename):
    net = network.load(load_filename)
else:
    net = network.RecurrentNetwork(
        Ni = Ni, # Number of inputs
        N = N, # Number of recurrent neurons
        No = No, # Number of read-out neurons
        tau = tau, # Time constant of the neurons
        g = g, # Synaptic strength scaling
        pc = pc, # Connection probability
        Io = Io, # Noise variance
        P_plastic = P_plastic, # Percentage of neurons receiving plastic synapses
    )

recurrent_trials = 0
readout_trials = 100000

t_start = 200
impulse_start = 200
impulse_end = impulse_start + 50
trial_duration = 1700
learning_rate = 0.1
batch_size = -1

plot_error_trials = 1
plot_recurrent_trials = 1
plot_readout_trials = 1
save_net_trials = 1

net.reinitialize_readout_weights()

print('##### Generate data #####')
targets_cartesian = []
targets_joints = []
recurrent_trajectories = []
impulses = []
errors = []
for figure in figures:
    points = figures[figure]
    traj_x_cartesian, traj_y_cartesian = generate_trajectory(points, step=0.18)

    impulse = np.zeros((trial_duration, Ni, 1))
    impulse[impulse_start:impulse_end, figure] = 2.
    impulses.append(impulse)

    target_cartesian = np.full((impulse.shape[0], No, 1), -5.)
    target_cartesian[impulse_end:impulse_end+traj_x_cartesian.shape[0], 0] = traj_x_cartesian
    target_cartesian[impulse_end:impulse_end+traj_x_cartesian.shape[0], 1] = traj_y_cartesian
    target_cartesian[impulse_end:impulse_end+traj_x_cartesian.shape[0], 2] = 5.
    targets_cartesian.append(target_cartesian)

    target_joints = np.full((impulse.shape[0], No, 1), -50.)

    traj_x_joints, traj_y_joints = ik_model.to_euler(traj_x_cartesian, traj_y_cartesian)

    target_joints[impulse_end:impulse_end+traj_x_joints.shape[0], 0] = traj_x_joints / np.pi * 180
    target_joints[impulse_end:impulse_end+traj_x_joints.shape[0], 1] = traj_y_joints / np.pi * 180
    target_joints[impulse_end:impulse_end+traj_x_joints.shape[0], 2] = 50.
    targets_joints.append(target_joints)

    errors.append([])

    r_trajectory = recurrent_trajectory(net, impulse)
    recurrent_trajectories.append(r_trajectory)

    plt.figure(figsize=(12, 8))
    for i in range(impulse.shape[1]):
        plt.subplot(impulse.shape[1], 1, i+1)
        plt.title('Impulse {0}'.format(i))
        plt.xlabel('t')
        plt.ylabel('signal')
        plt.plot(impulse[:, i])
        plt.minorticks_on()
        plt.xlim(0, impulse.shape[0])
        plt.grid(which='both')
    plt.savefig('plots/impulse_figure{0}'.format(figure))
    plt.close()

print('##### Train recurrent #####')
for trial in range(recurrent_trials):
    for figure in figures:
        print('Train recurrent trial {0}/{1} figure {2}/{3}'.format(trial+1, recurrent_trials, figure+1, len(figures)))

        train_recurrent(net, impulses[figure], recurrent_trajectories[figure], t_start)
    if plot_recurrent_trials > 0 and trial%plot_recurrent_trials == 0:
        plot_recurrent(figures, impulses, recurrent_trajectories, plots=3)

    if save_net_trials > 0 and trial%save_net_trials == 0:
        net.save(save_filename)

plot_readout(net, figures, impulses, targets_joints, t_start, 0, threshold = 0., markersize = 3.)
print('##### Train readout #####')
for trial in range(readout_trials):
    shuffled_figures = np.arange(len(figures))
    np.random.shuffle(shuffled_figures)
    error = 0.
    for i, figure in enumerate(shuffled_figures):
        points = figures[figure]
        print('Train readout trial {0}/{1} figure {2} ({3}/{4})'.format(trial+1, readout_trials, figure, i+1, len(figures)))
        net.reset()

        error = train_recadout(net, impulses[figure], targets_joints[figure], t_start, learning_rate, batch_size)
        errors[figure].append(error)

    if plot_error_trials > 0 and trial%plot_error_trials == 0:
        plot_error(errors)
    if save_net_trials > 0 and trial%save_net_trials == 0:
        net.save(save_filename)

    if plot_readout_trials > 0 and trial%plot_readout_trials == 0:
        plot_readout(net, figures, impulses, targets_joints, t_start, trial + 1, threshold = 0., markersize = 3.)

net.save(save_filename)