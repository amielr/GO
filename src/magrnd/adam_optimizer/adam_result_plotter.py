import matplotlib.pyplot as plt
import numpy as np
import mag_calc_functions_with_torch as mag_functions

def plot(B_scan, B_simulation, scan_pos_mat, levels=50):
    
    B_simulation_numpy = B_simulation.cpu().data.numpy()
    B_scan_numpy = B_scan.cpu().data.numpy()
    scan_pos_mat_numpy = scan_pos_mat.cpu().data.numpy()
    
    plt.figure(figsize  = (18,13))

    plt.subplot(2,2,1)
    plt.title("simulation")
    plot1 = plt.tricontourf(scan_pos_mat_numpy[:,0],scan_pos_mat_numpy[:,1], B_simulation_numpy, levels=levels)
    plt.colorbar(plot1)

    plt.subplot(2,2,2)
    plt.title("scan")
    plot2 = plt.tricontourf(scan_pos_mat_numpy[:,0],scan_pos_mat_numpy[:,1], B_scan_numpy, levels=levels)
    plt.colorbar(plot2)

    plt.subplot(2,2,3)
    plt.title("diff")
    diff = B_scan_numpy - B_simulation_numpy
    plot3 = plt.tricontourf(scan_pos_mat_numpy[:,0],scan_pos_mat_numpy[:,1], diff, levels=levels)
    plt.colorbar(plot3)

    plt.show()

def plot_summary(scan_path, parameters, xyz_averages):
    
    print("Predicted x,y,z (centered):", parameters.cpu().detach().numpy()[:,:3])
    print("Predicted of x,y,z:", parameters.cpu().detach().numpy()[:,:3] + xyz_averages)
    print("Predicted of m_x, m_y, m_z:", parameters.cpu().detach().numpy()[:,3:])
    print("Predicted M:", np.linalg.norm(parameters.cpu().detach().numpy()[:,3:]))
    print("parameters:", type(parameters))
    print("PSO-metric loss value:", mag_functions.POS_loss_for_trained_parameters(scan_path, parameters))
    
    B_scan, scan_pos_mat, _ = mag_functions.create_scan_pos_mat(scan_path, around_zero=True, nano_tesla=False)
    B_simulation = mag_functions.create_B_simulation(parameters, scan_pos_mat, nano_tesla = False)
    
    plot(B_scan, B_simulation, scan_pos_mat)