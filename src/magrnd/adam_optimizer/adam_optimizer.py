import os
import pandas as pd
import tkinter as tk
import tkinter.filedialog
import matplotlib.pyplot as plt
import adam_result_plotter as mag_plotter
import mag_calc_functions_with_torch as mag_functions


check_set = []
for root, dirs, files in os.walk('data'):
    for filename in files:
        check_set.append(os.path.join(root, filename))

"""  
#for testing the rates and thresholds

learning_rates = [5, 3, 1, 0.3, 0.1]
stop_thresholds = [1e-6, 1e-7, 1e-8, 1e-9]
factors = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
patiences = [10, 50, 100, 200]
thresholds = [1e-3, 1e-4, 1e-5, 1e-6]
"""

# initialize parameters
num_of_iters = 6000
factor = 0.5
patience = 50
threshold = 1e-5
verbose = True
learning_rate = 1

# get scan
scan_path = tk.filedialog.askopenfilename(title="choose mag scan")

# create optimizer
opt = mag_functions.AdamOptimizer(scan_path, learning_rate, num_of_iters,device=None)

# start optimizing
opt.set_learning_rate_scheduler(factor=factor, patience=patience,
                                threshold=threshold,
                                verbose=verbose)
loss_val, loss_hist, iters = opt.run_optimizer(stop_threshold=1e-7)

# plot result
mag_plotter.plot_summary(opt.scan_path, opt.parameters, opt.xyz_averages)
fig = plt.figure(figsize=(18, 13))
plt.plot(loss_hist)

# a dataframe with useful data
df = pd.DataFrame(columns=["scan_name", "learning_rate", "num_of_iters",
                           "factor", "patience", "threshold", "optimized_params",
                           "nomalized_loss", "PSO_like_loss", "nomalized_loss_history"])
