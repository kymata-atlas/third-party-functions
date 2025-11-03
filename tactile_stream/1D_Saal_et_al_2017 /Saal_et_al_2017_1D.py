import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter

#############
# This model implements a 1D version of the Saal et al 2017 model
#
# H.P. Saal, B.P. Delhaye, B.C. Rayhaun, & S.J. Bensmaia, Simulating tactile signals
# from the whole hand with millisecond precision, Proc. Natl. Acad. Sci. U.S.A. 114 (28)
# E5693-E5702, https://doi.org/10.1073/pnas.1704856114 (2017).
#############


# --- Configuration Section ---
# This class holds all the configurable parameters for the model.
# Values marked as 'PLACEHOLDER' should be refined based on experimental data
# or more detailed analysis from the supplementary materials of the original papers.

class Config:
    def __init__(self):
        # Simulation parameters
        self.dt = 0.001  # Simulation timestep in seconds (1 ms)
        self.sampling_rate = 1 / self.dt  # Hz

        # Skin Mechanics Parameters
        # These are simplified scaling factors for the 1D input.
        # In a more complex model, these would emerge from detailed biomechanical
        # simulations of skin deformation (e.g., using finite element models or
        # elastic half-space theory as in Sneddon, 1965, Ref. 69 in Saala et al. 2017).
        # SA1 and RA are sensitive to position/velocity, PC to acceleration/velocity.
        self.skin_mechanics_quasistatic_scale = 10.0  # PLACEHOLDER: Scales displacement to a stress-like input (e.g., Pa/mm)
        self.skin_mechanics_dynamic_velocity_scale = 50.0  # PLACEHOLDER: Scales velocity to stress-like input
        self.skin_mechanics_dynamic_acceleration_scale = 0.1  # PLACEHOLDER: Scales acceleration to stress-like input

        # Integrate-and-Fire Neuron Parameters (PLACEHOLDERS)
        # These parameters are based on the general structure described in Kim et al. 2010
        # and Dong et al. 2013, but their specific fitted values are not
        # readily available in the main text of these papers.
        # Kim et al. 2010 references Supplemental Figs. S4 and S5 for some details.
        # Dong et al. 2013 describes a more parsimonious model with fewer parameters (4 vs up to 13).
        # The values below are illustrative and need empirical fitting.

        # General IF parameters (can be shared or specific)
        self.V_rest = 0.0  # Resting potential (arbitrary units, often 0)
        self.V_reset = 0.0  # Reset potential after spike
        self.V_threshold = 1.0  # Spike threshold
        self.refractory_period_ms = 1  # Absolute refractory period in ms (e.g., 1ms for high-frequency coding)
        self.noise_std = 0.1  # Standard deviation of Gaussian noise added to input current

        # Filters and Weights for each afferent type
        # Input filters (linear prefilters) are crucial for frequency sensitivity
        # These are conceptual, representing the frequency tuning of each afferent type.
        # In actual implementation, these would be coefficients of FIR/IIR filters.
        # For simplification, we will apply a basic low-pass filter to the combined input to the IF model.

        self.afferent_params = {
            'SA1': {
                'membrane_tau': 0.005,  # Membrane time constant (s) - PLACEHOLDER
                'rect_pos_weight': 1.0,  # Weight for positive rectified input
                'rect_neg_weight': 0.0,
                # Weight for negative rectified input (SA1 typically responds to indentation, not retraction)
                'saturation_threshold': 5.0,
                # Input current saturation threshold - SA1 generally has higher linearity
                'post_spike_inhibition_strength': 0.0,  # Post-spike inhibition strength - PLACEHOLDER
                'post_spike_inhibition_tau': 0.005,  # Post-spike inhibition decay time (s) - PLACEHOLDER
                'conduction_delay_ms': 5,
                # Conduction delay (ms) - PLACEHOLDER (Saala et al. 2017 SI mentions 3-8ms)
                'response_modality_weights': {
                    'quasistatic': 1.0,  # SA1 strongly driven by displacement (quasistatic)
                    'dynamic_velocity': 0.2,  # Some velocity sensitivity
                    'dynamic_acceleration': 0.0  # Little to no acceleration sensitivity
                }
            },
            'RA': {
                'membrane_tau': 0.002,  # Membrane time constant (s) - PLACEHOLDER
                'rect_pos_weight': 1.0,  # Weight for positive rectified input
                'rect_neg_weight': 1.0,
                # Weight for negative rectified input (RA responds to both onset/offset, i.e., full-wave rectified velocity)
                'saturation_threshold': 3.0,
                # Input current saturation threshold - RA saturates more readily
                'post_spike_inhibition_strength': 0.5,  # Post-spike inhibition strength - PLACEHOLDER
                'post_spike_inhibition_tau': 0.003,  # Post-spike inhibition decay time (s) - PLACEHOLDER
                'conduction_delay_ms': 5,  # Conduction delay (ms) - PLACEHOLDER
                'response_modality_weights': {
                    'quasistatic': 0.0,  # RA generally not sensitive to sustained displacement
                    'dynamic_velocity': 1.0,  # RA strongly driven by velocity
                    'dynamic_acceleration': 0.1  # Some acceleration sensitivity
                }
            },
            'PC': {
                'membrane_tau': 0.001,  # Membrane time constant (s) - PLACEHOLDER (very fast)
                'rect_pos_weight': 1.0,  # Weight for positive rectified input
                'rect_neg_weight': 1.0,
                # Weight for negative rectified input (PC responds to high-frequency vibration, full-wave rectified acceleration)
                'saturation_threshold': 2.0,  # Input current saturation threshold - PC saturates readily
                'post_spike_inhibition_strength': 0.8,  # Post-spike inhibition strength - PLACEHOLDER
                'post_spike_inhibition_tau': 0.001,  # Post-spike inhibition decay time (s) - PLACEHOLDER
                'conduction_delay_ms': 5,  # Conduction delay (ms) - PLACEHOLDER
                'response_modality_weights': {
                    'quasistatic': 0.0,  # PC not sensitive to sustained displacement
                    'dynamic_velocity': 0.5,  # PC also sensitive to velocity
                    'dynamic_acceleration': 1.0  # PC strongly driven by acceleration
                }
            }
        }

        # Afferent population density (approximate, per mm^2)
        # These values influence how many afferents are simulated within the 1mm x 1mm patch.
        # In reality, densities vary significantly across the hand.
        self.afferent_density_per_mm2 = {
            'SA1': 70,  # PLACEHOLDER: Typical density for fingertip
            'RA': 140,  # PLACEHOLDER: Typical density for fingertip
            'PC': 15  # PLACEHOLDER: Typical density for fingertip
        }
        self.patch_width_mm = 1.0
        self.patch_length_mm = 1.0
        self.patch_area_mm2 = self.patch_width_mm * self.patch_length_mm

        # New: Afferent Depths (approximate, in mm from skin surface)
        # These are based on typical anatomical locations.
        self.afferent_depth_mm = {
            'SA1': 0.3,  # Superficial, epidermis/superficial dermis. Approx. 0.2-0.5mm.
            'RA': 0.7,  # Intermediate, dermal papillae. Approx. 0.5-1.0mm.
            'PC': 3.0   # Deep, subcutaneous tissue. Approx. 2-4mm+.
        }

        # Indenter parameters for the spatial model
        self.indenter_radius_mm = 0.5  # 1mm diameter rod means 0.5mm radius
        self.indenter_center_xy_mm = (self.patch_width_mm / 2, self.patch_length_mm / 2) # Center of the 1x1mm patch

        # Store afferent locations after initialization
        self.afferent_locations = self._generate_afferent_locations()


    def _generate_afferent_locations(self):
        """Generates random (x, y) locations for each afferent within the patch,
           and assigns their specific depth."""
        locations = {'SA1': [], 'RA': [], 'PC': []}
        for afferent_type in ['SA1', 'RA', 'PC']:
            num_afferents_in_population = int(
                self.afferent_density_per_mm2[afferent_type] * self.patch_area_mm2
            )
            num_afferents_in_population = max(1, num_afferents_in_population) # Ensure at least one

            for _ in range(num_afferents_in_population):
                # Random (x, y) within the 1mm x 1mm patch
                x = np.random.uniform(0, self.patch_width_mm)
                y = np.random.uniform(0, self.patch_length_mm)
                z = self.afferent_depth_mm[afferent_type]
                locations[afferent_type].append((x, y, z))
        return locations
    
# --- Spatial Mechanics Model (Simplified Elastic Half-Space Approximation) ---
# These functions estimate the mechanical signal at a specific (r, z) location
# given the input displacement. They are highly simplified approximations
# of full elastic half-space solutions like Sneddon (1965).

def calculate_local_quasistatic_signal(displacement_mm, r_mm, z_mm, indenter_radius_mm, config):
    """
    Estimates the local quasistatic stress/strain at (r, z) from a central indentation.
    Simplified model: decays with radial distance (r) and depth (z).
    Higher z (deeper) means weaker signal. Higher r (further from center) means weaker.
    """
    # Avoid division by zero if r is exactly 0 and z is very small
    r_effective = max(r_mm, 1e-6)
    z_effective = max(z_mm, 1e-6)

    # Simple decay model: scales inversely with depth, and exponentially with radial distance.
    # The constants (e.g., 5.0, 2.0) are tunable to control fall-off steepness.
    # The idea is that at the surface (z=0) directly under the indenter (r=0),
    # the signal is strongest. Deeper or further away, it's weaker.

    # Normalized radial distance relative to indenter radius
    norm_r = r_effective / indenter_radius_mm
    # A simple decay model where signal falls off faster for SA1s with distance
    # and deeper location
    radial_decay = np.exp(-norm_r * 2.5) # Tune this constant for fall-off steepness
    depth_decay = np.exp(-z_effective / (indenter_radius_mm * 1.5)) # Tune depth sensitivity

    # Combine overall displacement with spatial decay and scaling factor
    local_signal = displacement_mm * config.skin_mechanics_quasistatic_scale * radial_decay * depth_decay
    return local_signal

def calculate_local_dynamic_signal(velocity_m_per_s, acceleration_m_per_s2, r_mm, z_mm, indenter_radius_mm, config):
    """
    Estimates the local dynamic stress/strain at (r, z).
    Similar decay principles as quasistatic, but applied to velocity/acceleration components.
    PCs are very sensitive to high frequencies and generally found deeper.
    """
    r_effective = max(r_mm, 1e-6)
    z_effective = max(z_mm, 1e-6)

    norm_r = r_effective / indenter_radius_mm
    # Dynamic signals (especially PC) can propagate further and are less sensitive
    # to very superficial r-decay compared to SA1s, but still decay.
    # They are strongly affected by depth.
    radial_decay = np.exp(-norm_r * 1.5) # Dynamic signals might decay slower radially
    depth_decay = np.exp(-z_effective / (indenter_radius_mm * 2.0)) # Deeper penetration for dynamic

    # Combine velocity and acceleration components
    dynamic_base_signal = (
        velocity_m_per_s * config.skin_mechanics_dynamic_velocity_scale +
        acceleration_m_per_s2 * config.skin_mechanics_dynamic_acceleration_scale
    )

    local_signal = dynamic_base_signal * radial_decay * depth_decay
    return local_signal

# --- Integrate-and-Fire Neuron Model ---
# This class implements the Integrate-and-Fire neuron model described in Kim et al. 2010
# and Dong et al. 2013.
# The model transforms a filtered mechanical input into spike trains.
class IntegrateFireNeuron:
    def __init__(self, afferent_type, config):
        self.type = afferent_type
        self.params = config.afferent_params[afferent_type]
        self.config = config

        self.V = config.V_rest  # Current membrane potential
        self.last_spike_time_ms = -np.inf  # Track last spike time for refractory period

        # Initialize post-spike inhibition effect
        self.post_spike_inhibition_current = 0.0

        # Filters for the dynamic component, if needed (conceptual for now)
        # In the original papers, each input (pos, vel, acc, jerk) has specific filters.
        # For this simplified model, we'll apply a conceptual low-pass based on membrane_tau.

    def _apply_linear_filters_and_rectification(self, signal, weights):
        # A simplified representation of the linear filtering and rectification
        # described in Kim et al. 2010 and Dong et al. 2013.
        # In full models, each derivative (pos, vel, acc, jerk) would have specific filters.
        # Here, we'll use the weighted sum of inputs and then rectify.

        # Rectification: Separate positive and negative components
        positive_component = np.maximum(0, signal)
        negative_component = np.minimum(0, signal)

        # Apply rectification weights
        rectified_signal = (
                self.params['rect_pos_weight'] * positive_component +
                abs(self.params['rect_neg_weight'] * negative_component)
        # abs because negative component is already negative
        )
        return rectified_signal

    def _apply_saturation(self, current):
        # Apply saturation nonlinearity
        return np.minimum(current, self.params['saturation_threshold'])

    def step(self, mechanical_input, current_time_ms):
        # Determine effective mechanical input based on afferent type's sensitivity
        # This is where the 'stimulus quantity' sensitivity is applied.
        effective_input_signal = (
                mechanical_input['quasistatic'] * self.params['response_modality_weights']['quasistatic'] +
                mechanical_input['raw_velocity'] * self.params['response_modality_weights']['dynamic_velocity'] +
                mechanical_input['raw_acceleration'] * self.params['response_modality_weights']['dynamic_acceleration']
        )

        # Apply linear filtering (conceptually, inherent in membrane dynamics for now)
        # and rectification based on afferent type.
        rectified_filtered_input = self._apply_linear_filters_and_rectification(
            effective_input_signal, self.params['response_modality_weights']
        )

        # Apply saturation nonlinearity
        saturated_input_current = self._apply_saturation(rectified_filtered_input)

        # Add Gaussian noise
        noise = np.random.normal(0, self.config.noise_std)
        total_input_current = saturated_input_current + noise

        # Integrate-and-Fire dynamics
        spike = 0

        # Check refractory period
        if (current_time_ms - self.last_spike_time_ms) < self.params[
            'conduction_delay_ms'] + self.config.refractory_period_ms:
            # If still in refractory period, no spike, voltage stays reset or clamped
            self.V = self.config.V_reset
            return spike

        # Update membrane potential (Leaky Integrate-and-Fire equation)
        # dV/dt = (-(V - V_rest) + I_input) / tau_m
        # V(t+dt) = V(t) + dt/tau_m * (-(V(t) - V_rest) + I_input)
        dV = (-(self.V - self.config.V_rest) + total_input_current - self.post_spike_inhibition_current) / self.params[
            'membrane_tau'] * self.config.dt
        self.V += dV

        # Decay post-spike inhibition
        self.post_spike_inhibition_current *= np.exp(-self.config.dt / self.params['post_spike_inhibition_tau'])

        # Check for spike
        if self.V >= self.config.V_threshold:
            spike = 1
            self.V = self.config.V_reset  # Reset potential
            self.last_spike_time_ms = current_time_ms
            # Apply post-spike inhibition if relevant for the afferent type
            # Kim et al. 2010 mentions "postspike inhibitory current".
            self.post_spike_inhibition_current = self.params['post_spike_inhibition_strength']

        return spike

# --- Main Simulation Function ---
def simulate_tactile_model(displacement_input_array_mm, config):
    """
    Simulates the tactile afferent responses with spatial variation.

    Args:
        displacement_input_array_mm (np.ndarray): 1D array of skin's vertical displacement
                                                in millimeters, sampled at 1ms intervals.
        config (Config): An instance of the Config class containing all model parameters.

    Returns:
        dict: A dictionary where keys are afferent types ('SA1', 'RA', 'PC') and values
              are lists of numpy arrays, each array representing a binary spike train
              (1 for spike, 0 for no spike) for an individual afferent.
    """
    num_time_steps = len(displacement_input_array_mm)
    time_ms = np.arange(num_time_steps) * config.dt * 1000  # Time in milliseconds

    # Calculate global velocity and acceleration from the input displacement
    displacement_m = displacement_input_array_mm * 1e-3 # Convert to meters for calculation
    global_velocity_m_per_s = np.gradient(displacement_m, config.dt)
    global_acceleration_m_per_s2 = np.gradient(global_velocity_m_per_s, config.dt)

    all_afferent_spike_trains = {
        'SA1': [],
        'RA': [],
        'PC': []
    }

    progress_interval_steps = int(1 / config.dt) # 1000 steps for 1 second

    # Step 2: Simulate populations of Integrate-and-Fire neurons
    for afferent_type in ['SA1', 'RA', 'PC']:
        afferent_locations_for_type = config.afferent_locations[afferent_type]
        num_afferents_in_population = len(afferent_locations_for_type)

        print(f"Simulating {num_afferents_in_population} {afferent_type} afferents...")

        for i, (x_loc, y_loc, z_depth) in enumerate(afferent_locations_for_type):
            neuron = IntegrateFireNeuron(afferent_type, config)
            spike_train = np.zeros(num_time_steps, dtype=int)

            # Calculate radial distance for this specific afferent from the indenter center
            # Assumes indenter is centered at config.indenter_center_xy_mm
            r_mm = np.sqrt(
                (x_loc - config.indenter_center_xy_mm[0])**2 +
                (y_loc - config.indenter_center_xy_mm[1])**2
            )

            for t_idx in range(num_time_steps):
                current_time_ms = time_ms[t_idx]

                # Get the global displacement, velocity, acceleration at this time step
                current_global_disp_mm = displacement_input_array_mm[t_idx]
                current_global_vel_mps = global_velocity_m_per_s[t_idx]
                current_global_acc_mps2 = global_acceleration_m_per_s2[t_idx]

                # Calculate the local mechanical input for *this specific afferent*
                # based on its (r, z) location
                local_quasistatic = calculate_local_quasistatic_signal(
                    current_global_disp_mm, r_mm, z_depth, config.indenter_radius_mm, config
                )
                local_dynamic = calculate_local_dynamic_signal(
                    current_global_vel_mps, current_global_acc_mps2, r_mm, z_depth, config.indenter_radius_mm, config
                )

                # The `calculate_local_dynamic_signal` already combines vel and acc.
                # However, the IF neuron expects them separately for its modality weights.
                # So we need to pass components influenced by spatial decay.
                # A simple way for now is to apply the decay factor to the raw components.
                # This is a simplification; a full model would derive local vel/acc from local disp.
                # For now, let's derive the effective local velocity and acceleration from the local dynamic signal
                # by reversing the scaling, but applying the spatial decay.
                # This part is a heuristic given the simplified fall-off functions.
                # A more rigorous approach would compute local displacement first, then its derivatives.

                # Let's refine this to directly pass the spatially decayed raw components
                # or a scaled version of them, because the IF neuron's weights are on raw_velocity/acceleration.
                radial_decay_dyn = np.exp(-(r_mm / config.indenter_radius_mm) * 1.5)
                depth_decay_dyn = np.exp(-z_depth / (config.indenter_radius_mm * 2.0))
                spatial_decay_factor_dynamic = radial_decay_dyn * depth_decay_dyn

                current_mechanoreceptor_input = {
                    'quasistatic': local_quasistatic,
                    'dynamic': local_dynamic, # This is the combined dynamic, not used directly by IF
                    'raw_velocity': current_global_vel_mps * spatial_decay_factor_dynamic,
                    'raw_acceleration': current_global_acc_mps2 * spatial_decay_factor_dynamic
                }

                spike_train[t_idx] = neuron.step(current_mechanoreceptor_input, current_time_ms)

                # --- Progress Printing ---
                if (t_idx + 1) % progress_interval_steps == 0:
                    processed_seconds = (t_idx + 1) * config.dt
                    print(f"\r  Afferent {i+1}/{num_afferents_in_population}: {processed_seconds:.0f} seconds processed.", end='')
            
            print("") # Newline after each afferent simulation
            all_afferent_spike_trains[afferent_type].append(spike_train)

    return all_afferent_spike_trains

# --- Example Usage ---
if __name__ == '__main__':
    # Initialize configuration
    model_config = Config()

    # --- Change starts here ---
    # Define the path to your CSV file
    csv_file_path = 'tactile_stream/input/LH_thumb_stimulus_short.csv'

    try:
        # Read the displacement input from the CSV file
        # Assumes a single row of comma-separated values
        displacement_input_mm = np.loadtxt(csv_file_path, delimiter=',')
        print(f"Successfully loaded displacement input from {csv_file_path}")
        print(f"Input shape: {displacement_input_mm.shape}")

        # Ensure the input is 1D if it was read as a 2D array with one row
        if displacement_input_mm.ndim > 1 and displacement_input_mm.shape[0] == 1:
            displacement_input_mm = displacement_input_mm.flatten()
        # Scale up to 4mm
        displacement_input_mm = displacement_input_mm * 4 # Scale up for more activity

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        print("Please ensure the CSV file is in the correct directory or provide the full path.")
        # Create a dummy input for demonstration if file not found
        print("Creating dummy displacement input for demonstration (1000ms ramp up and down).")
        time_points = np.arange(0, 1.0, model_config.dt) # 1 second
        displacement_input_mm = np.sin(time_points * np.pi) * 0.5 # A smooth bump
        displacement_input_mm = np.tile(displacement_input_mm, 2) # Repeat twice for 2 seconds
        displacement_input_mm = displacement_input_mm * 4 # Scale up for more activity


    # Determine simulation duration based on the loaded input length
    duration_s = len(displacement_input_mm) * model_config.dt
    time_points = np.arange(0, duration_s, model_config.dt) # Re-align time_points with loaded data
    num_time_steps = len(time_points) # Re-align num_time_steps

    print(f"Duration of stimulus is '{duration_s}'s")

    # Run the simulation
    afferent_responses = simulate_tactile_model(displacement_input_mm, model_config)

    # Compute rolling average (window = 2ms) for each afferent's spike train (for easier visualisation)
    rolling_window = 6  # in ms (since dt=1ms, window=2 samples)
    afferent_responses_rolling = {}
    for afferent_type, trains in afferent_responses.items():
        afferent_responses_rolling[afferent_type] = [
            np.convolve(train, np.ones(rolling_window)/rolling_window, mode='same') for train in trains
        ]

    # --- Output Analysis Example with Plotting ---
    print("\n--- Simulation Results Summary ---")
    
    # Define time window for plotting
    plot_duration_ms = 2000
    plot_time_indices = min(int(plot_duration_ms / model_config.dt / 1000), num_time_steps) # Ensure it doesn't exceed data length

    # Prepare plot
    fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True) # 4 subplots: displacement + 3 afferent types
    time_ms_plot = np.arange(0, plot_time_indices * model_config.dt * 1000, model_config.dt * 1000) # Time axis for plotting

    # Plot 1: Displacement Input
    axs[0].plot(time_ms_plot, displacement_input_mm[:plot_time_indices], color='blue')
    axs[0].set_title('Skin Displacement Input')
    axs[0].set_ylabel('Displacement (mm)')
    axs[0].grid(True)

    # Plot 2, 3, 4: Afferent Population Activity
    afferent_types = ['SA1', 'RA', 'PC']
    for i, afferent_type in enumerate(afferent_types):
        spike_trains = afferent_responses_rolling[afferent_type]
        num_afferents = len(spike_trains)

        if num_afferents > 0:
            # Sum spikes across all afferents of this type at each millisecond
            population_spike_sum = np.sum(spike_trains, axis=0) # Sums along the afferent dimension
            
            total_spikes = np.sum(population_spike_sum)
            # Ensure duration_s is not zero to avoid division by zero
            avg_firing_rate_hz = (total_spikes / num_afferents) / duration_s if duration_s > 0 else 0
            print(f"{afferent_type} Population (n={num_afferents}): Average Firing Rate = {avg_firing_rate_hz:.2f} Hz")

            # Plot summed activity
            axs[i+1].plot(time_ms_plot, population_spike_sum[:plot_time_indices], color='red')
            axs[i+1].set_title(f'{afferent_type} Population Activity')
            axs[i+1].set_ylabel('Spikes per ms')
            axs[i+1].grid(True)
        else:
            print(f"{afferent_type} Population: No afferents simulated (density/patch size too small).")
            axs[i+1].set_title(f'{afferent_type} Population Activity (No Afferents Simulated)')
            axs[i+1].set_ylabel('Spikes per ms')
            axs[i+1].grid(True)

    axs[-1].set_xlabel('Time (ms)') # Set x-label only on the bottom-most subplot
    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    plt.savefig('tactile_stream/output/tactile_model_output.png') # You can specify a different filename and format (e.g., .pdf, .jpeg)
    plt.show()
