import torch
import numpy as np
import pandas as pd
import time

device = torch.device("cuda:1")


class AdamOptimizer():
    
    def __init__(self,scan_path, learning_rate, num_iterations, device,  init_params = [[0,0,-10,0,100,-100]] ,around_zero = True, nano_tesla = True):
        
        self.scan_path = scan_path
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.device = device
        self.around_zero = around_zero
        self.nano_tesla = nano_tesla
        
        self.parameters =  torch.tensor(init_params, device = device, dtype = torch.float64, requires_grad = True)
        self.open_scan()
    
        self.opt = torch.optim.Adam(params = [self.parameters], lr = self.learning_rate)
        self.scheduler = None
        self.loss_history = []
       

    def open_scan(self):
        
        self.B_scan, self.scan_pos_mat, self.xyz_averages = create_scan_pos_mat(self.scan_path, around_zero = self.around_zero, nano_tesla = self.nano_tesla)
        self.B_simulation = create_B_simulation(self.parameters, self.scan_pos_mat, nano_tesla = self.nano_tesla)
        self.N = len(self.B_scan)
        
        
    def run_optimizer(self, stop_threshold = None):
        
        loss = lambda: torch.sqrt((1/self.N)*torch.sum(torch.square(self.B_scan - self.B_simulation)))
        
        tic = time.time()
        for i in range(self.num_iterations):
            
            self.opt.zero_grad() # Reset grads buffer 
            
            self.B_simulation = create_B_simulation(self.parameters, self.scan_pos_mat, nano_tesla = self.nano_tesla)
            loss_fn = loss()
            loss_fn.backward(retain_graph=True)
            
            self.opt.step()
            self.loss_history.append(loss_fn.cpu().detach().numpy())
            
            if not stop_threshold is None:
                if len(self.loss_history) > 100:
                     if np.std(self.loss_history[-100:]) < stop_threshold:
                        print(f"Stop Threshold Activated! after: {i+1} iters")
                        break
            
            if not self.scheduler is None:
                self.scheduler.step(loss_fn)
            
        toc = time.time()
        print("Time:", toc - tic)
        print("loss of", i+1, "iteration:", str(loss_fn.cpu().detach().numpy()))
        
        return loss_fn.cpu().detach().numpy(), self.loss_history, i+1
        
    
    def set_learning_rate_scheduler(self, mode = 'min', factor = 0.1, patience = 10, 
                                    threshold = 1e-4, threshold_mode = 'rel', cooldown = 0,
                                    min_lr = 0, eps = 1e-08, verbose = False):
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, 
                                                                    mode = mode, 
                                                                    factor = factor, 
                                                                    patience = patience, 
                                                                    threshold = threshold,                                                                               threshold_mode = threshold_mode,
                                                                    cooldown = cooldown,
                                                                    min_lr = min_lr,
                                                                    eps = eps, 
                                                                    verbose = verbose)
        
    


    
class Constants:
    B_external = torch.tensor([0, 4.4 * 10 ** -5, -4.4 * 10 ** -5], dtype = torch.float64, requires_grad = True) / np.sqrt(2)  # [T]
    

def load_gz_file(scan_path):
    """
    :param scan_path: the path of the mag scan
    :return: DataFrame of the scan
    """
    df_scan = pd.read_csv(scan_path, delimiter="\t")
    df_scan = df_scan.drop("time", axis = 1)
    
    return df_scan


def create_scan_pos_mat(scan_path, around_zero=True, nano_tesla=True): 
    """
    :param scan_path: the path of the mag scan
    :param around_zero: create scan around zero if True
    :return: array of positions of x,y,z from real scan
    """
    scan = load_gz_file(scan_path)
    
    B_column = "B"
    if not B_column in scan.columns:
        B_column = "original B"
    
    B_scan = scan[B_column] - scan[B_column].mean()
    
    if not nano_tesla:
        B_scan = B_scan.values*10**-9
    
    scan_pos_mat = np.array([scan['x'].values, scan['y'].values, scan['height'].values])

    xyz_averages = []
    if around_zero:
        #create scan positions around (0,0,0) axis
        x_average = scan['x'].mean()
        y_average = scan['y'].mean()
        z_average = scan['height'].mean()
        xyz_averages = [x_average, y_average, z_average]
        
        scan_pos_mat = np.array([scan_pos_mat[0] - x_average, scan_pos_mat[1] - y_average, scan_pos_mat[2] - z_average]).T   
    else:
        #transpose to match schapes with create B simulations
        scan_pos_mat = scan_pos_mat.T
        
    B_scan = torch.tensor(B_scan, dtype = torch.float64)
    scan_pos_mat = torch.tensor(scan_pos_mat, dtype = torch.float64)
    return B_scan, scan_pos_mat, xyz_averages


def distance_from_predicted_source(parameters, scan_pos_mat):
    """
    return: vector of differences between the scan and the source
    """
    pos_source = parameters[0,:3]
    temp = scan_pos_mat - torch.reshape(pos_source, (3,))
    return temp


def create_B_simulation(parameters, scan_pos_mat, nano_tesla=True):
    """
    :param parameters: array of [x,y,z,mx,my,mz]
    :param scan_pos_mat: array of positions of x,y,z from real scan
    :return: scan like array of B (magnetic field) given parameters
    """
    # get distance matrix from predicted source
    r = distance_from_predicted_source(parameters, scan_pos_mat)

    # calculate scalar of r for each scan point
    #scalar_r = np.linalg.norm(r, axis=1).reshape(-1,1)[:,0] # from (1,n) to (n,)
    scalar_r =  torch.reshape(torch.linalg.norm(r,dim = 1), shape = (1,-1))
    
    
    predicted_moment = torch.t(parameters)[-3:]
    predicted_moment = predicted_moment[:,0]
    

    distance_moment_prod = torch.matmul(r, predicted_moment) # from (1,n) to (n,)
    
    B_mat = 10 ** -7 * torch.t(3 * torch.t(torch.t(r) * distance_moment_prod / (scalar_r ** 2)) - predicted_moment) / scalar_r ** 3

    magnetic_field = torch.t(B_mat) + Constants.B_external
    scalar_magnetic_field = torch.linalg.norm(magnetic_field,dim = 1)
    scalar_magnetic_field = scalar_magnetic_field - torch.mean(scalar_magnetic_field)
    
    if nano_tesla:
        scalar_magnetic_field = scalar_magnetic_field*10**9
    
    return scalar_magnetic_field


def POS_loss_for_trained_parameters(scan_path, parameters):
    """
    this functions take the parameters that optimized our function value, 
    and insert it into the PSO loss to evaluate the precision of Adam optimizer.
    """
    comp_B_scan, scan_pos_mat, _ = create_scan_pos_mat(scan_path, around_zero = False, nano_tesla= False)

    PSO_loss = torch.sqrt(torch.sum(torch.square(comp_B_scan - create_B_simulation(parameters, scan_pos_mat, nano_tesla = False))))
    return PSO_loss


