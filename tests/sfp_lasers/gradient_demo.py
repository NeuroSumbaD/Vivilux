import __main__
import os
from time import time, sleep
from threading import Thread
from queue import Queue
import json

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import TextBox, Button
from matplotlib import gridspec

import numpy as np

from sfp_board_config_6x6 import fpga, netlist

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True)

total_iteration_count = 0
total_iteration_time = 0.0

# Get the path to the directory containing the main script
main_script_dir = os.path.dirname(os.path.abspath(__main__.__file__))
tests_dir = os.path.dirname(main_script_dir)
print(f"{main_script_dir=}, {tests_dir=}")

# bar_state_json = os.path.join(tests_dir, "4x4_bar_state_voltages.json")
identity_json_path = os.path.join(main_script_dir, "central_difference_descent_parameters.json")
output_json_path = os.path.join(main_script_dir, "demo_params.json")

with open(identity_json_path, 'r') as bar_file:
    identity_params = json.load(bar_file)

output_PDs = ["PD_1_5","PD_2_5", "PD_3_5", "PD_4_5", "PD_5_5", "PD_6_5"]
theta_nets = ["3_1_i", "2_2_i", "4_2_i", "3_3_i", "2_4_i", "4_4_i", "3_5_i",
              "1_1_i", "5_1_i", "1_3_i", "5_3_i", "1_5_i", "5_5_i"]

ni_board = netlist.board_dict['NI']

def set_params(params: np.ndarray) -> None:
    '''Set the parameters in the netlist according to the provided dictionary.'''
    for net, value in zip(theta_nets, params):
        if value > 5.2:
            value = 5.2
        elif value < 0.0:
            value = 0.0
        netlist[net].vout(value)

def measure_matrix() -> np.ndarray:
    '''Measure the current 4x4 transfer matrix of the mesh using
        one-hot input vector (single laser on at a time). The initial
        offset of should be a measurement of the readout voltage with
        all lasers turned to their lowest state.

        Note: We use the vibration state rather than directly
        turning on and off the lasers because the power settles more quickly
    '''
    measured_matrix = np.zeros((6,4))
    for one_hot_index in range(4):
        vibration_state = [0, 0, 0, 0]
        fpga.update_vibration(vibration_state)
        sleep(1e-3) # 1 ms sleep for power to settle
        pd_offsets = ni_board.group_vin(output_PDs)

        vibration_state[one_hot_index] = 1
        fpga.update_vibration(vibration_state[::-1]) # Laser indices are inversed because least significant bit is laser 1 and is big endian
        # Wait for power to settle
        sleep(1e-3) # 1 ms sleep for power to settle

        col = pd_offsets.copy()
        col -= ni_board.group_vin(output_PDs)
        col /= np.sum(col) # Normalize to input power of 1
        # col /= norm_factors[one_hot_index]  # Normalize to input power of 1
        measured_matrix[:, one_hot_index] = col
    return measured_matrix

def training_loop(data_queue, target_matrix):
    with netlist:
        laser_off_readout = ni_board.group_vin(output_PDs)
        print(f"{laser_off_readout=}")
        
        fpga.update_lasers([1,1,1,1])  # Turn on all lasers
        fpga.update_vibration([0,0,0,0])  # Set to min state to calculate offset

        print("Waiting 10 seconds for thermal equilibrium and lasers to stabilize...")
        sleep(10) # Wait for system to stabilize
        pd_offsets = ni_board.group_vin(output_PDs)
        print("Measured PD offsets with all lasers on, min vibration state:")
        print(pd_offsets)

        # Calculate each normalization factor using the sum of PD outputs for each column
        first_measurement = measure_matrix()
        print(f"First measured matrix:")
        print(first_measurement)

        print("Setting initial theta parameters to mean 2pi voltage from bar state calibration in row 1 and 6...")
        params = np.array([identity_params[net] for net in theta_nets])
        params += np.random.uniform(-0.25, 0.25, size=len(theta_nets)) # Small random perturbation
        # params = np.array([0 for net in theta_nets])
        set_params(params)

        initial_matrix = measure_matrix()
        print("Initial measured matrix with guessed theta parameters:")
        print(initial_matrix)

        init_delta = target_matrix - initial_matrix
        init_delta_mag = np.linalg.norm(init_delta, 'fro')
        print("Initial error from target matrix (Frobenius norm):", init_delta_mag)
        print("Delta matrix:")
        print(init_delta)

        # Gradient descent one parameter at a time
        learning_rate = 0.1
        delta_voltage = 0.1  # Small voltage change for central difference approximation
        num_iterations = 500
        history = np.full(num_iterations + 1, np.nan)  # Store delta magnitude at each step
        history[0] = init_delta_mag
        current_delta = init_delta.copy()
        data_queue.put((0, init_delta_mag))
        for iteration in range(num_iterations):
            start_time = time()
            print(f"Iteration {iteration+1}/{num_iterations}...")
            gradients = np.zeros(len(theta_nets))
            for param_name in theta_nets:
                param_index = theta_nets.index(param_name)
                print(f"\tOptimizing parameter {param_name}")
                # Central difference approximation
                params_plus = params.copy()
                params_plus[param_index] += delta_voltage
                set_params(params_plus)
                # meas_plus = measure_matrix(pd_offsets, norm_factors)
                meas_plus = measure_matrix()
                delta_plus = meas_plus

                params_minus = params.copy()
                params_minus[param_index] -= delta_voltage
                set_params(params_minus)
                # meas_minus = measure_matrix(pd_offsets, norm_factors)
                meas_minus = measure_matrix()
                delta_minus = meas_minus

                # Gradient calculation
                grad_matrix = (delta_plus - delta_minus) / (2 * delta_voltage)
                grad_mag = -2*np.sum(grad_matrix * current_delta)  # Chain rule for Frobenius norm squared
                print(f"\tGradient magnitude for {param_name}: {grad_mag}")
                gradients[param_index] = grad_mag

            # Update parameter
            # params[param_index] -= learning_rate * grad_mag
            params -= learning_rate * gradients
            set_params(params)

            # Measure new matrix and error
            # new_meas = measure_matrix(pd_offsets, norm_factors)
            new_meas = measure_matrix()
            new_delta = target_matrix - new_meas
            new_delta_mag = np.linalg.norm(new_delta, 'fro')
            history[iteration + 1] = new_delta_mag
            current_delta = new_delta.copy()

            print(f"Updated parameters: {np.round(params, 3)}")
            print(f"New error from target matrix (Frobenius norm): {new_delta_mag}")

            total_iteration_time += time() - start_time
            total_iteration_count += 1
            avg_time = total_iteration_time / total_iteration_count
            print(f"Average time per iteration: {avg_time:.3f} seconds")
            data_queue.put((iteration + 1, new_delta_mag))
    print("Final measured matrix after optimization:")
    print(new_meas)

    # Save final parameters to JSON
    with open(output_json_path, 'w') as out_file:
        json.dump({net: float(params[i]) for i, net in enumerate(theta_nets)}, out_file, indent=4)
        print(f"Final parameters saved to {output_json_path}")

def main(): # interactive plotting
    data_queue = Queue()
    training_thread = None

    # Create control figure with GridSpec
    fig_control = plt.figure(figsize=(6, 8))
    fig_control.canvas.manager.set_window_title('Calibration Target')
    
    gs = gridspec.GridSpec(8, 4, figure=fig_control, 
                           left=0.1, right=0.9, top=0.95, bottom=0.1,
                           hspace=0.3, wspace=0.3)
    
    # Create 6x4 grid of text boxes
    textboxes = []
    for i in range(6):
        row = []
        for j in range(4):
            ax_text = fig_control.add_subplot(gs[i, j])
            tb = TextBox(ax_text, '', initial='0.0')
            tb.set_val(f"{0.0 if (i-1)!=j else 1.0}")
            row.append(tb)
        textboxes.append(row)
    
    # Add button in the bottom row
    ax_button = fig_control.add_subplot(gs[7, 1:3])
    button = Button(ax_button, 'Calibrate')


    # Set up plotting figure
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    xdata, ydata = [], []
    plt.xlabel('Epoch')
    line, = ax.plot([], [])
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')

    # Add annotation for global minimum (do not show until data is available)
    min_annot = ax.annotate('', xy=(0,0), xytext=(95,0.95),
                            horizontalalignment='right', verticalalignment='top',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->',
                                            connectionstyle='angle3,angleA=15,angleB=90',
                                            # connectionstyle='bar, fraction=0.3',
                                            color='black'),
                            )
    min_annot.set_visible(False)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        while not data_queue.empty():
            epoch, loss = data_queue.get()
            xdata.append(epoch)
            ydata.append(loss)
        line.set_data(xdata, ydata)

        # Data is not empty, anotate the global minimum
        if ydata:
            min_loss = min(ydata)
            min_epoch = xdata[ydata.index(min_loss)]

            # Update annotation
            nonlocal min_annot
            min_annot.set_visible(True)
            min_annot.set_text(f'Min Loss: {min_loss:.4g} at Epoch {min_epoch}')
            min_annot.xy = (min_epoch, min_loss)
        return line, min_annot

    ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=250, cache_frame_data=False)
    ani.pause()  # Start paused


    def on_calibrate_click(event):
        nonlocal training_thread

        # Stop previous training thread and start a new one
        if training_thread is not None and training_thread.is_alive():
            training_thread.join(timeout=0)

        if len(line.get_xdata()) > 0:
            line.set_data([], [])
            xdata.clear()
            ydata.clear()
            min_annot.set_visible(False)


        # Extract target matrix from textboxes
        try:
            target_matrix = []
            for i in range(6):
                row = []
                for j in range(4):
                    value = float(textboxes[i][j].text)
                    row.append(value)
                target_matrix.append(row)
        except ValueError:
            print("Invalid matrix values!")
            return
        
        ani.resume()  # Resume animation
        
        # Start training
        training_thread = Thread(target=training_loop, 
                               args=(data_queue,np.array(target_matrix)), 
                               daemon=True)
        training_thread.start()
        button.label.set_text('Running...')
        button.ax.set_facecolor('lightcoral')
        fig_control.canvas.draw_idle()

    button.on_clicked(on_calibrate_click)

    plt.show()


if __name__ == "__main__":
    main()
