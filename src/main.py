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

def train_readout(net:network.RecurrentNetwork, impulse, target, t_start):
    net.reset()
    bar = progress.Bar(50)
    error = 0.
    for t in range(impulse.shape[0]):
        net.step(impulse[t], noise=True)

        if t < t_start:
            continue
        error_t = net.train_readout(target[t, :])
        error += np.abs(error_t.sum())
        bar.progress((t + 1) / impulse.shape[0])
    return error / impulse.shape[0]

def test_net(net:network.RecurrentNetwork, impulse):
    net.reset()
    z = []
    for t in range(impulse.shape[0]):
        net.step(impulse[t], noise=False)
        z.append(net.z)
    return np.array(z)

def plot_error(errors):
    plt.figure(figsize=(15, 10))
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

if not os.path.isdir('plots'):
    os.mkdir('plots')

limb1 = 10
limb2 = 10
refX = 0
refY = 0

ik_model = robot.Transform_2DoF(limb1, limb2, refX, refY)

figures = {
    # house
    0: [[10, 10, 10, 20],
        [10, 20, 15, 25],
        [15, 25, 20, 20],
        [20, 20, 20, 10],
        [20, 10, 10, 10],
        [10, 10, 20, 20],
        [20, 20, 10, 20],
        [10, 20, 20, 10]],
    # triangle
    1: [[10, 10, 15, 15],
        [15, 15, 20, 10],
        [20, 10, 10, 10]],
    # square
    2: [[10, 10, 10, 20],
        [10, 20, 20, 20],
        [20, 20, 20, 10],
        [20, 10, 10, 10]]
}

Ni = len(figures) + 1 # Number of inputs
N = 2000 # Number of recurrent neurons
No = 3 # Number of read-out neurons
tau = 4.#4.#10.0 # Time constant of the neurons
g = 1.5 # Synaptic strength scaling Spectral radius
pc = 0.1 # Connection probability
Io = 0.001 # Noise variance
P_plastic = 0.6 # Percentage of neurons receiving plastic synapses

load_filename = 'net.pkl'
save_filename = 'net.pkl'

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

recurrent_trials = 200
readout_trials = 200

t_start = 100
impulse_start = 200
impulse_end = impulse_start + 50
trial_duration = 1700

plot_error_trials = 2
plot_recurrent_trials = 1
save_net_trials = 1

net.save(save_filename)

print('##### Generate data #####')
targets = []
recurrent_trajectories = []
impulses = []
errors = []
for figure in figures:
    points = figures[figure]
    traj_x, traj_y = generate_trajectory(points, step=0.07)

    impulse = np.zeros((trial_duration, Ni, 1))
    impulse[impulse_start:impulse_end, 0] = 2.
    impulse[impulse_start:impulse_end, figure + 1] = 2.
    impulses.append(impulse)

    target = np.full((impulse.shape[0], No, 1), -1.)
    target[impulse_end:impulse_end+traj_x.shape[0], 0] = traj_x
    target[impulse_end:impulse_end+traj_x.shape[0], 1] = traj_y
    target[impulse_end:impulse_end+traj_x.shape[0], 2] = 1.
    targets.append(target)

    errors.append([])

    r_trajectory = recurrent_trajectory(net, impulse)
    recurrent_trajectories.append(r_trajectory)

    plt.figure(figsize=(12, 8))
    plt.title('Impulse')
    for i in range(impulse.shape[1]):
        plt.subplot(impulse.shape[1], 1, i+1)
        plt.plot(impulse[:, i])
        plt.minorticks_on()
        plt.xlim(0, impulse.shape[0])
        plt.grid(which='both')
    plt.savefig('plots/impulse_figure{0}'.format(figure))
    plt.close()

print('##### Train recurrent #####')
for trial in range(recurrent_trials):
    print('trial {0}/{1}'.format(trial+1, recurrent_trials))

    for figure in figures:
        print('figure', figure)

        train_recurrent(net, impulses[figure], recurrent_trajectories[figure], t_start)
    if trial%plot_recurrent_trials == 0:
        plot_recurrent(figures, impulses, recurrent_trajectories, plots=3)

    if trial%save_net_trials == 0:
        net.save(save_filename)

print('##### Train readout #####')
for trial in range(readout_trials):
    print('trial {0}/{1}'.format(trial+1, readout_trials))
    for figure in figures:
        points = figures[figure]
        print('figure', figure)
        net.reset()

        error = train_readout(net, impulses[figure], targets[figure], 0)
        errors[figure].append(error)

    if trial%plot_error_trials == 0:
        plot_error(errors)
    if trial%save_net_trials == 0:
        net.save(save_filename)

for figure in figures:
    points = figures[figure]
    print('figure', figure)

    z = test_net(net, impulses[figure])
    target = targets[figure]

    threshold = 0.9

    target = target[t_start:]
    target_x_pos, target_y_pos = target[:,0][target[:,2]>=threshold], target[:,1][target[:,2]>=threshold]

    z = z[t_start:]
    z_x_pos, z_y_pos = z[:, 0][z[:,2]>=threshold], z[:, 1][z[:,2]>=threshold]
    z_x_neg, z_y_neg = z[:, 0][z[:,2]<threshold], z[:, 1][z[:,2]<threshold]

    
    markersize = 3.

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.minorticks_on()
    plt.grid(which='both')
    plt.title('Figure {0}'.format(figure))
    plt.scatter(target_x_pos, target_y_pos, s=markersize, label='target', color='blue')
    plt.scatter(z_x_pos, z_y_pos, s = markersize, label='z: draw', color='green')
    plt.scatter(z_x_neg, z_y_neg, s = markersize, label='z: don\'t draw', color='red')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel('t')
    plt.ylabel('signal')
    plt.minorticks_on()
    plt.xlim(0, impulses[figure].shape[0])
    plt.grid(which='both')
    plt.plot(target[:,0], label='$target_x$')
    plt.plot(target[:,1], label='$target_y$')
    plt.plot(target[:,2], label='$target_{should\,draw}$')
    plt.plot(z[:,0], label='$z_x$')
    plt.plot(z[:,1], label='$z_y$')
    plt.plot(z[:,2], label='$z_{should\,draw}$')
    
    plt.legend(loc='upper right')
    plt.savefig('plots/readout_figure{0}.png'.format(figure))
    plt.close()

net.save(save_filename)