import numpy as np
from scipy.signal import butter, lfilter

#############
# This model implements a simplifed version of the Saal et al 2017 model
#
# H.P. Saal, B.P. Delhaye, B.C. Rayhaun,  & S.J. Bensmaia,   Simulating tactile signals
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
        # elastic half-space theory as in Sneddon, 1965, Ref. 69 in Saala et al. 2017)[cite: 2200].
        # SA1 and RA are sensitive to position/velocity, PC to acceleration/velocity[cite: 2669].
        self.skin_mechanics_quasistatic_scale = 1000.0  # PLACEHOLDER: Scales displacement to a stress-like input (e.g., Pa/mm)
        self.skin_mechanics_dynamic_velocity_scale = 50.0  # PLACEHOLDER: Scales velocity to stress-like input
        self.skin_mechanics_dynamic_acceleration_scale = 0.1  # PLACEHOLDER: Scales acceleration to stress-like input

        # Integrate-and-Fire Neuron Parameters (PLACEHOLDERS)
        # These parameters are based on the general structure described in Kim et al. 2010 [cite: 2280]
        # and Dong et al. 2013[cite: 3633], but their specific fitted values are not
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
        # Input filters (linear prefilters) are crucial for frequency sensitivity [cite: 2299, 2637, 2639]
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
                # Input current saturation threshold - SA1 generally has higher linearity [cite: 2587]
                'post_spike_inhibition_strength': 0.0,  # Post-spike inhibition strength - PLACEHOLDER
                'post_spike_inhibition_tau': 0.005,  # Post-spike inhibition decay time (s) - PLACEHOLDER
                'conduction_delay_ms': 5,
                # Conduction delay (ms) - PLACEHOLDER (Saala et al. 2017 SI mentions 3-8ms) [cite: 2407]
                'response_modality_weights': {
                    'quasistatic': 1.0,  # SA1 strongly driven by displacement (quasistatic) [cite: 2669]
                    'dynamic_velocity': 0.2,  # Some velocity sensitivity [cite: 2669]
                    'dynamic_acceleration': 0.0  # Little to no acceleration sensitivity
                }
            },
            'RA': {
                'membrane_tau': 0.002,  # Membrane time constant (s) - PLACEHOLDER
                'rect_pos_weight': 1.0,  # Weight for positive rectified input
                'rect_neg_weight': 1.0,
                # Weight for negative rectified input (RA responds to both onset/offset, i.e., full-wave rectified velocity) [cite: 2483]
                'saturation_threshold': 3.0,
                # Input current saturation threshold - RA saturates more readily [cite: 4057]
                'post_spike_inhibition_strength': 0.5,  # Post-spike inhibition strength - PLACEHOLDER
                'post_spike_inhibition_tau': 0.003,  # Post-spike inhibition decay time (s) - PLACEHOLDER
                'conduction_delay_ms': 5,  # Conduction delay (ms) - PLACEHOLDER
                'response_modality_weights': {
                    'quasistatic': 0.0,  # RA generally not sensitive to sustained displacement
                    'dynamic_velocity': 1.0,  # RA strongly driven by velocity [cite: 2669]
                    'dynamic_acceleration': 0.1  # Some acceleration sensitivity
                }
            },
            'PC': {
                'membrane_tau': 0.001,  # Membrane time constant (s) - PLACEHOLDER (very fast)
                'rect_pos_weight': 1.0,  # Weight for positive rectified input
                'rect_neg_weight': 1.0,
                # Weight for negative rectified input (PC responds to high-frequency vibration, full-wave rectified acceleration)
                'saturation_threshold': 2.0,  # Input current saturation threshold - PC saturates readily [cite: 4057]
                'post_spike_inhibition_strength': 0.8,  # Post-spike inhibition strength - PLACEHOLDER
                'post_spike_inhibition_tau': 0.001,  # Post-spike inhibition decay time (s) - PLACEHOLDER
                'conduction_delay_ms': 5,  # Conduction delay (ms) - PLACEHOLDER
                'response_modality_weights': {
                    'quasistatic': 0.0,  # PC not sensitive to sustained displacement
                    'dynamic_velocity': 0.5,  # PC also sensitive to velocity [cite: 2669]
                    'dynamic_acceleration': 1.0  # PC strongly driven by acceleration [cite: 2669]
                }
            }
        }

        # Afferent population density (approximate, per mm^2)
        # These values influence how many afferents are simulated within the 1mm x 1mm patch.
        # In reality, densities vary significantly across the hand[cite: 2191].
        self.afferent_density_per_mm2 = {
            'SA1': 70,  # PLACEHOLDER: Typical density for fingertip [cite: 374]
            'RA': 140,  # PLACEHOLDER: Typical density for fingertip [cite: 374]
            'PC': 15  # PLACEHOLDER: Typical density for fingertip [cite: 374]
        }
        self.patch_width_mm = 1.0
        self.patch_length_mm = 1.0
        self.patch_area_mm2 = self.patch_width_mm * self.patch_length_mm


# --- Skin Mechanics Model ---
# This function calculates the mechanical input to the afferents based on vertical displacement.
# It is a highly simplified version of the detailed skin mechanics described in Saala et al. 2017,
# which uses an elastic half-space model (Ref. 69 in Saala et al. 2017) to compute stress fields
# from spatiotemporal indentation patterns.
# Given the 1D input, we approximate the stress at the receptor location.
def calculate_mechanoreceptor_input(displacement_input_mm, config):
    # Convert displacement from mm to meters for consistency if using physics equations later.
    # For now, keep in mm and scale appropriately.
    displacement_m = displacement_input_mm * 1e-3

    # Calculate velocity and acceleration using numerical differentiation
    # Use central difference for better accuracy, handling edges
    velocity_m_per_s = np.gradient(displacement_m, config.dt)
    acceleration_m_per_s2 = np.gradient(velocity_m_per_s, config.dt)

    # Apply simplified scaling for quasistatic and dynamic components
    # These are heuristic approximations of how the original model
    # would derive "stress" or "strain" at the mechanoreceptor site.
    # The actual model in Saala et al. 2017 involves spatial integration and filtering.
    quasistatic_signal = displacement_m * config.skin_mechanics_quasistatic_scale
    dynamic_signal = (
            velocity_m_per_s * config.skin_mechanics_dynamic_velocity_scale +
            acceleration_m_per_s2 * config.skin_mechanics_dynamic_acceleration_scale
    )

    # For SA1, the primary input is quasistatic.
    # For RA, the primary input is dynamic (velocity).
    # For PC, the primary input is dynamic (acceleration and high-frequency velocity).
    # We combine them here and let the IF model's `response_modality_weights`
    # differentiate sensitivity.
    # This combines the 'stimulus quantity' identification discussed in Kim et al. 2010[cite: 2557, 2669].
    return {
        'quasistatic': quasistatic_signal,
        'dynamic': dynamic_signal,  # This 'dynamic' signal is further processed by afferent type
        'raw_velocity': velocity_m_per_s,  # Keep raw velocity for explicit use below
        'raw_acceleration': acceleration_m_per_s2  # Keep raw acceleration for explicit use below
    }


# --- Integrate-and-Fire Neuron Model ---
# This class implements the Integrate-and-Fire neuron model described in Kim et al. 2010 [cite: 2280]
# and Dong et al. 2013[cite: 3633].
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
        # In the original papers, each input (pos, vel, acc, jerk) has specific filters[cite: 2637].
        # For this simplified model, we'll apply a conceptual low-pass based on membrane_tau.

    def _apply_linear_filters_and_rectification(self, signal, weights):
        # A simplified representation of the linear filtering and rectification
        # described in Kim et al. 2010 [cite: 2299, 2321] and Dong et al. 2013[cite: 3615].
        # In full models, each derivative (pos, vel, acc, jerk) would have specific filters.
        # Here, we'll use the weighted sum of inputs and then rectify.

        # Rectification: Separate positive and negative components [cite: 2458]
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
        # Apply saturation nonlinearity [cite: 4057]
        return np.minimum(current, self.params['saturation_threshold'])

    def step(self, mechanical_input, current_time_ms):
        # Determine effective mechanical input based on afferent type's sensitivity
        # This is where the 'stimulus quantity' sensitivity [cite: 2669] is applied.
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

        # Add Gaussian noise [cite: 2322]
        noise = np.random.normal(0, self.config.noise_std)
        total_input_current = saturated_input_current + noise

        # Integrate-and-Fire dynamics [cite: 2302]
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
            self.V = self.config.V_reset  # Reset potential [cite: 2302]
            self.last_spike_time_ms = current_time_ms
            # Apply post-spike inhibition if relevant for the afferent type
            # Kim et al. 2010 mentions "postspike inhibitory current"[cite: 2344].
            self.post_spike_inhibition_current = self.params['post_spike_inhibition_strength']

        return spike


# --- Main Simulation Function ---
def simulate_tactile_model(displacement_input_array_mm, config):
    """
    Simulates the tactile afferent responses.

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

    # Step 1: Calculate mechanical inputs for afferents
    # This simplified function takes a 1D displacement and derives a single
    # stress-like signal for all afferents within the 1mm patch.
    # FUTURE WORK: For a more biophysically accurate model, particularly with
    # 2D spatial inputs, consider implementing a detailed biomechanical skin model
    # (e.g., as in Saala et al. 2017 [cite: 2191] or Dandekar and Srinivasan 1998 [cite: 2744])
    # to calculate spatially varying stresses at each mechanoreceptor location.
    # This would involve elastic half-space solutions for indenter contact.
    mechanoreceptor_input_signals = calculate_mechanoreceptor_input(
        displacement_input_array_mm, config
    )

    all_afferent_spike_trains = {
        'SA1': [],
        'RA': [],
        'PC': []
    }

    # Step 2: Simulate populations of Integrate-and-Fire neurons
    for afferent_type in ['SA1', 'RA', 'PC']:
        # Calculate number of afferents in the 1mm x 1mm patch
        num_afferents_in_population = int(
            config.afferent_density_per_mm2[afferent_type] * config.patch_area_mm2
        )
        # Ensure at least one afferent if density is very low
        num_afferents_in_population = max(1, num_afferents_in_population)

        print(f"Simulating {num_afferents_in_population} {afferent_type} afferents...")

        for i in range(num_afferents_in_population):
            neuron = IntegrateFireNeuron(afferent_type, config)
            spike_train = np.zeros(num_time_steps, dtype=int)

            # FUTURE WORK: For higher realism, particularly if a spatial skin model
            # is implemented, each afferent in the population could have a unique
            # (randomly sampled) receptive field location within the 1mm patch.
            # This would lead to slightly different `effective_input_signal` for
            # each neuron, even from a single stimulus. With the current 1D input
            # and uniform stress assumption, all afferents of the same type receive
            # the same base mechanical input; variability comes primarily from noise.
            for t_idx in range(num_time_steps):
                current_time_ms = time_ms[t_idx]
                # Pass the mechanical input at the current time step
                current_mechanoreceptor_input = {
                    'quasistatic': mechanoreceptor_input_signals['quasistatic'][t_idx],
                    'dynamic': mechanoreceptor_input_signals['dynamic'][t_idx],
                    'raw_velocity': mechanoreceptor_input_signals['raw_velocity'][t_idx],
                    'raw_acceleration': mechanoreceptor_input_signals['raw_acceleration'][t_idx]
                }
                spike_train[t_idx] = neuron.step(current_mechanoreceptor_input, current_time_ms)

            all_afferent_spike_trains[afferent_type].append(spike_train)

    return all_afferent_spike_trains


# --- Example Usage ---
if __name__ == '__main__':
    # Initialize configuration
    model_config = Config()

    # Create a dummy displacement input array (e.g., 10 seconds at 1ms intervals)
    # This simulates your description: pseudo-sinusoidal flutter (e.g., 20-50 Hz)
    # as a carrier for a 0.5mm high 200Hz vibration, appearing and disappearing randomly.
    duration_s = 10  # 10 seconds
    time_points = np.arange(0, duration_s, model_config.dt)
    num_time_steps = len(time_points)

    displacement_input_mm = np.zeros(num_time_steps)

    # Simulate pseudo-sinusoidal flutter (carrier)
    flutter_frequency = 40  # Hz
    flutter_amplitude = 2.0  # mm (0 to 4mm range implies 2mm amplitude around 2mm baseline)
    baseline_displacement = 2.0  # mm
    displacement_input_mm = baseline_displacement + flutter_amplitude * np.sin(
        2 * np.pi * flutter_frequency * time_points)

    # Add 200Hz vibration appearing and disappearing randomly
    vibration_frequency = 200  # Hz
    vibration_amplitude = 0.25  # mm (0.5mm high implies +/- 0.25mm from carrier)

    # Introduce random vibration segments
    segment_duration_s = 0.5  # Each segment lasts 0.5 seconds
    num_segments = int(duration_s / segment_duration_s)

    for i in range(num_segments):
        if np.random.rand() > 0.5:  # 50% chance to have vibration in a segment
            start_idx = int(i * segment_duration_s / model_config.dt)
            end_idx = int((i + 1) * segment_duration_s / model_config.dt)

            # Ensure indices are within bounds
            start_idx = min(start_idx, num_time_steps)
            end_idx = min(end_idx, num_time_steps)

            if start_idx < end_idx:
                segment_time = time_points[start_idx:end_idx] - time_points[start_idx]
                displacement_input_mm[start_idx:end_idx] += vibration_amplitude * np.sin(
                    2 * np.pi * vibration_frequency * segment_time)

    # Ensure displacement stays within 0-4mm range
    displacement_input_mm = np.clip(displacement_input_mm, 0, 4)

    # Run the simulation
    afferent_responses = simulate_tactile_model(displacement_input_mm, model_config)

    # --- Output Analysis Example ---
    # You can now analyze the spike trains for each afferent type.
    # For instance, calculate the mean firing rate over the simulation duration.
    print("\n--- Simulation Results Summary ---")
    for afferent_type, spike_trains in afferent_responses.items():
        total_spikes = sum(np.sum(st) for st in spike_trains)
        num_afferents = len(spike_trains)

        if num_afferents > 0:
            avg_firing_rate_hz = (total_spikes / num_afferents) / duration_s
            print(f"{afferent_type} Population (n={num_afferents}): Average Firing Rate = {avg_firing_rate_hz:.2f} Hz")

            # Example: Look at the first afferent's spike train
            # print(f"First {afferent_type} afferent spike train (first 100ms):")
            # print(spike_trains[0][:100]) # Print first 100ms
        else:
            print(f"{afferent_type} Population: No afferents simulated (density/patch size too small).")

    # Further analysis would involve plotting spike rasters, PSTHs, etc.
    # using libraries like matplotlib.