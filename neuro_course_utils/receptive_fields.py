"""
Receptive Fields Lab - Interactive Visualization Module

This module provides functions for exploring Linear/Nonlinear/Poisson (LNP) models
and spike-triggered averaging for computational neuroscience course.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CORE MODEL PARAMETERS
# =============================================================================

# Default simulation parameters
DEFAULT_DT = 0.005  # 5 ms timestep
DEFAULT_N_HISTORY = 100  # 100 timesteps = 500 ms history


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_spike_times(spikes, time_points):
    """
    Extract spike times from binary spike train.

    Parameters
    ----------
    spikes : ndarray
        Binary array where 1 indicates a spike
    time_points : ndarray
        Array of time points

    Returns
    -------
    spike_times : ndarray
        Array of times when spikes occurred
    """
    return time_points[spikes == 1]


def create_linear_filter(n_history, dt, frequency=40, decay=8, amplitude=10):
    """
    Create an oscillatory linear filter with exponential decay.

    Parameters
    ----------
    n_history : int
        Number of history time points
    dt : float
        Time step in seconds
    frequency : float
        Oscillation frequency (affects sin wave)
    decay : float
        Exponential decay rate
    amplitude : float
        Filter amplitude

    Returns
    -------
    history_timepoints : ndarray
        Time points for the filter (negative, before spike)
    linear_filter : ndarray
        Filter values at each time point
    """
    history_timepoints = dt * np.arange(-n_history, 0)
    linear_filter = amplitude * np.sin(history_timepoints * frequency) * np.exp(history_timepoints * decay)
    return history_timepoints, linear_filter


def nonlinearity_arctan(L, gain=40):
    """
    Saturating nonlinearity using arctan.

    Parameters
    ----------
    L : float or ndarray
        Linear filter output
    gain : float
        Controls saturation level

    Returns
    -------
    float or ndarray
        Nonlinear output
    """
    return np.arctan(L / gain) * gain


def nonlinearity_relu(L):
    """ReLU nonlinearity (rectified linear)."""
    return np.maximum(0, L)


def nonlinearity_sigmoid(L, gain=1, threshold=0):
    """Sigmoid nonlinearity."""
    return gain / (1 + np.exp(-(L - threshold)))


# =============================================================================
# CORE LNP MODEL FUNCTIONS
# =============================================================================

def apply_filter(k, stimulus_history):
    """
    Apply a linear filter to stimulus history.

    Parameters
    ----------
    k : ndarray
        Linear filter weights
    stimulus_history : ndarray
        Recent stimulus values (same length as k)

    Returns
    -------
    float
        Dot product of filter and stimulus history
    """
    return np.sum(k * stimulus_history)


def gen_spiking(stimulus, model_params, dt):
    """
    Generate spikes from an LNP model.

    Parameters
    ----------
    stimulus : ndarray
        Stimulus values over time
    model_params : dict
        Dictionary containing:
        - 'k': linear filter
        - 'r_0': baseline firing rate
        - 'F': nonlinearity function
    dt : float
        Time step in seconds

    Returns
    -------
    spikes : ndarray
        Binary spike train
    rate : ndarray
        Firing rate over time
    """
    k = model_params['k']
    F = model_params['F']
    r_0 = model_params['r_0']

    n_time_indices = len(stimulus)
    filter_length = len(k)

    rate = np.zeros(n_time_indices)
    spikes = np.zeros(n_time_indices)

    for i in range(filter_length, n_time_indices):
        L = apply_filter(k, stimulus[i - filter_length:i])
        rate[i] = np.maximum(0, r_0 + F(L))
        spikes[i] = (np.random.uniform() < rate[i] * dt)

    return spikes, rate


def spike_triggered_avg(stimulus, spikes, n_history_indices, dt):
    """
    Compute spike-triggered average.

    Parameters
    ----------
    stimulus : ndarray
        Stimulus values over time
    spikes : ndarray
        Binary spike train
    n_history_indices : int
        Number of history time points to include
    dt : float
        Time step (unused but kept for API consistency)

    Returns
    -------
    STA : ndarray
        Spike-triggered average of stimulus history
    """
    n_spikes = int(np.sum(spikes))
    if n_spikes == 0:
        return np.zeros(n_history_indices)

    all_indices = np.arange(len(spikes))
    spike_indices = all_indices[spikes == 1]

    # Only use spikes that have enough history
    spike_indices = spike_indices[spike_indices >= n_history_indices]

    STA_sum = np.zeros(n_history_indices)

    for j in spike_indices:
        STA_sum = STA_sum + stimulus[j - n_history_indices:j]

    return STA_sum / len(spike_indices) if len(spike_indices) > 0 else STA_sum


def create_model_params(baseline_rate=10, filter_freq=40, filter_decay=8,
                        filter_amplitude=10, n_history=100, dt=0.005):
    """
    Create model parameters dictionary.

    Parameters
    ----------
    baseline_rate : float
        Baseline firing rate r_0
    filter_freq : float
        Filter oscillation frequency
    filter_decay : float
        Filter decay rate
    filter_amplitude : float
        Filter amplitude
    n_history : int
        Number of history time points
    dt : float
        Time step

    Returns
    -------
    model_params : dict
        Dictionary with 'k', 'r_0', 'F' keys
    """
    _, linear_filter = create_linear_filter(n_history, dt, filter_freq, filter_decay, filter_amplitude)
    return {
        'k': linear_filter,
        'r_0': baseline_rate,
        'F': nonlinearity_arctan
    }


# Hidden model for "present_to_fish" function
_FISH_MODEL_PARAMS = None


def _get_fish_model():
    """Get or create the hidden fish model."""
    global _FISH_MODEL_PARAMS
    if _FISH_MODEL_PARAMS is None:
        _, k = create_linear_filter(100, 0.005, frequency=40, decay=8, amplitude=10)
        _FISH_MODEL_PARAMS = {
            'k': k,
            'r_0': 10,
            'F': nonlinearity_arctan
        }
    return _FISH_MODEL_PARAMS


def present_to_fish(stimulus, dt):
    """
    Present a stimulus to the "fish neuron" and record spikes.

    This simulates recording from a real neuron - students don't know
    the underlying model parameters.

    Parameters
    ----------
    stimulus : ndarray
        Stimulus to present
    dt : float
        Time step

    Returns
    -------
    spikes : ndarray
        Recorded spike train
    """
    model_params = _get_fish_model()
    spikes, _ = gen_spiking(stimulus, model_params, dt)
    return spikes


# =============================================================================
# INTERACTIVE PLOTTING FUNCTIONS
# =============================================================================

def plot_stimulus_response(amplitude=0.5, pulse_length=10, baseline_rate=10,
                           filter_freq=40, filter_decay=8):
    """
    Interactive plot showing LNP model response to a pulse stimulus.

    Parameters
    ----------
    amplitude : float
        Pulse amplitude
    pulse_length : int
        Pulse duration in timesteps
    baseline_rate : float
        Baseline firing rate
    filter_freq : float
        Filter oscillation frequency
    filter_decay : float
        Filter decay rate
    """
    dt = 0.005
    n_history = 100
    n_samples = 500
    t = np.arange(n_samples) * dt

    # Create filter
    history_timepoints, linear_filter = create_linear_filter(
        n_history, dt, filter_freq, filter_decay, amplitude=10
    )

    # Create stimulus
    stimulus = np.zeros(n_samples)
    stimulus[200:200 + pulse_length] = amplitude

    # Create model and generate spikes
    model_params = {
        'k': linear_filter,
        'r_0': baseline_rate,
        'F': nonlinearity_arctan
    }
    spikes, rate = gen_spiking(stimulus, model_params, dt)

    # Calculate filter output for display
    filter_output = np.zeros(n_samples)
    for i in range(n_history, n_samples):
        filter_output[i] = apply_filter(linear_filter, stimulus[i - n_history:i])

    # Plotting
    fig = plt.figure(figsize=(12, 8))

    # Stimulus
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(t, stimulus, 'b-', linewidth=1.5)
    ax1.set_ylabel('Stimulus')
    ax1.set_title('Input Stimulus')
    ax1.set_xlim(0, t[-1])
    ax1.grid(True, alpha=0.3)

    # Linear filter
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(history_timepoints * 1000, linear_filter, 'g-', linewidth=2)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('$k_i$')
    ax2.set_title('Linear Filter')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Firing rate
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(t, rate, 'r-', linewidth=1.5)
    ax3.axhline(y=baseline_rate, color='k', linestyle='--', alpha=0.5, label=f'$r_0$={baseline_rate}')
    ax3.set_ylabel('Rate (Hz)')
    ax3.set_title('Firing Rate')
    ax3.set_xlim(0, t[-1])
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Nonlinearity
    ax4 = fig.add_subplot(3, 2, 4)
    L_range = np.linspace(-50, 100, 200)
    ax4.plot(L_range, nonlinearity_arctan(L_range), 'purple', linewidth=2)
    ax4.set_xlabel('Filter Output (L)')
    ax4.set_ylabel('F(L)')
    ax4.set_title('Nonlinearity')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax4.grid(True, alpha=0.3)

    # Spikes
    ax5 = fig.add_subplot(3, 2, 5)
    spike_times = t[spikes == 1]
    ax5.eventplot(spike_times, lineoffsets=0.5, linelengths=0.8, colors='black')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Spikes')
    ax5.set_title(f'Generated Spikes ({int(np.sum(spikes))} total)')
    ax5.set_xlim(0, t[-1])
    ax5.set_ylim(0, 1)
    ax5.set_yticks([])
    ax5.grid(True, alpha=0.3)

    # Filter output
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.plot(t, filter_output, 'orange', linewidth=1.5)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('L')
    ax6.set_title('Linear Filter Output')
    ax6.set_xlim(0, t[-1])
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_filter_explorer(filter_freq=40, filter_decay=8, filter_amplitude=10, baseline_rate=10):
    """
    Explore how filter parameters affect the neural response.

    Parameters
    ----------
    filter_freq : float
        Filter oscillation frequency
    filter_decay : float
        Filter decay rate
    filter_amplitude : float
        Filter amplitude
    baseline_rate : float
        Baseline firing rate
    """
    dt = 0.005
    n_history = 100
    n_samples = 600
    t = np.arange(n_samples) * dt

    # Create filter
    history_timepoints, linear_filter = create_linear_filter(
        n_history, dt, filter_freq, filter_decay, filter_amplitude
    )

    # Create pulse stimulus
    stimulus = np.zeros(n_samples)
    stimulus[150:160] = 1.0

    # Generate response
    model_params = {
        'k': linear_filter,
        'r_0': baseline_rate,
        'F': nonlinearity_arctan
    }
    spikes, rate = gen_spiking(stimulus, model_params, dt)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    # Filter shape
    axes[0, 0].plot(history_timepoints * 1000, linear_filter, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time Before Spike (ms)')
    axes[0, 0].set_ylabel('Filter Weight $k_i$')
    axes[0, 0].set_title(f'Linear Filter (freq={filter_freq}, decay={filter_decay})')
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].grid(True, alpha=0.3)

    # Stimulus
    axes[0, 1].plot(t, stimulus, 'g-', linewidth=1.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Stimulus')
    axes[0, 1].set_title('Input: Unit Pulse')
    axes[0, 1].set_xlim(0, t[-1])
    axes[0, 1].grid(True, alpha=0.3)

    # Rate response
    axes[1, 0].plot(t, rate, 'r-', linewidth=1.5)
    axes[1, 0].axhline(y=baseline_rate, color='k', linestyle='--', alpha=0.5, label=f'$r_0$={baseline_rate}')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Rate (Hz)')
    axes[1, 0].set_title('Firing Rate Response')
    axes[1, 0].set_xlim(0, t[-1])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Spikes
    spike_times = t[spikes == 1]
    axes[1, 1].eventplot(spike_times, lineoffsets=0.5, linelengths=0.8, colors='black')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Spikes')
    axes[1, 1].set_title(f'Generated Spikes ({int(np.sum(spikes))} total)')
    axes[1, 1].set_xlim(0, t[-1])
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_yticks([])
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_sta_convergence(stimulus_duration=5, baseline_rate=10, filter_freq=40):
    """
    Show how STA converges to true filter with more data.

    Parameters
    ----------
    stimulus_duration : float
        Duration of white noise stimulus in seconds
    baseline_rate : float
        Baseline firing rate
    filter_freq : float
        Filter frequency for the true filter
    """
    dt = 0.005
    n_history = 100
    n_samples = int(stimulus_duration / dt)

    # Create true filter
    history_timepoints, true_filter = create_linear_filter(
        n_history, dt, filter_freq, decay=8, amplitude=10
    )

    # Create white noise stimulus
    stimulus = np.random.randn(n_samples)

    # Generate spikes
    model_params = {
        'k': true_filter,
        'r_0': baseline_rate,
        'F': nonlinearity_arctan
    }
    spikes, rate = gen_spiking(stimulus, model_params, dt)

    # Compute STA
    STA = spike_triggered_avg(stimulus, spikes, n_history, dt)

    # Scale STA to compare with true filter
    if np.max(np.abs(STA)) > 0:
        STA_scaled = STA / np.max(np.abs(STA)) * np.max(np.abs(true_filter))
    else:
        STA_scaled = STA

    # Calculate correlation
    correlation = np.corrcoef(true_filter, STA)[0, 1] if np.std(STA) > 0 else 0

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # True filter vs STA
    axes[0].plot(history_timepoints * 1000, true_filter, 'b-', linewidth=2, label='True Filter')
    axes[0].plot(history_timepoints * 1000, STA_scaled, 'r--', linewidth=2, label='STA (scaled)')
    axes[0].set_xlabel('Time Before Spike (ms)')
    axes[0].set_ylabel('Filter Weight')
    axes[0].set_title(f'Filter Comparison (r = {correlation:.3f})')
    axes[0].legend()
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].grid(True, alpha=0.3)

    # Spike count info
    n_spikes = int(np.sum(spikes))
    axes[1].bar(['Spikes'], [n_spikes], color='steelblue', edgecolor='black')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Total Spikes: {n_spikes}')
    axes[1].grid(True, alpha=0.3)

    # Stimulus sample
    t_plot = np.arange(min(1000, n_samples)) * dt
    axes[2].plot(t_plot, stimulus[:len(t_plot)], 'g-', linewidth=0.5, alpha=0.7)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Stimulus')
    axes[2].set_title('White Noise Stimulus (first 1s)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()

    print(f"Stimulus duration: {stimulus_duration:.1f} s")
    print(f"Number of spikes: {n_spikes}")
    print(f"Correlation between true filter and STA: {correlation:.3f}")


def plot_stimulus_comparison(stimulus_type='white_noise', duration=15, baseline_rate=10):
    """
    Compare STA quality for different stimulus types.

    Parameters
    ----------
    stimulus_type : str
        Type of stimulus: 'white_noise', 'sine_wave', 'binary_noise', 'pink_noise'
    duration : float
        Stimulus duration in seconds
    baseline_rate : float
        Baseline firing rate
    """
    dt = 0.005
    n_history = 100
    n_samples = int(duration / dt)

    # Create true filter
    history_timepoints, true_filter = create_linear_filter(
        n_history, dt, frequency=40, decay=8, amplitude=10
    )

    # Create stimulus based on type
    if stimulus_type == 'white_noise':
        stimulus = np.random.randn(n_samples)
        stim_label = 'Gaussian White Noise'
    elif stimulus_type == 'sine_wave':
        t = np.arange(n_samples) * dt
        stimulus = np.sin(t * 50)  # 50 rad/s sine wave
        stim_label = 'Sine Wave (correlated!)'
    elif stimulus_type == 'binary_noise':
        stimulus = np.random.choice([-1, 1], size=n_samples).astype(float)
        stim_label = 'Binary White Noise'
    elif stimulus_type == 'pink_noise':
        # Generate pink noise (1/f)
        white = np.random.randn(n_samples)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples)
        freqs[0] = 1  # Avoid division by zero
        fft = fft / np.sqrt(freqs)
        stimulus = np.fft.irfft(fft, n_samples)
        stimulus = stimulus / np.std(stimulus)
        stim_label = 'Pink Noise (1/f, correlated)'
    else:
        stimulus = np.random.randn(n_samples)
        stim_label = 'Unknown'

    # Generate spikes
    model_params = {
        'k': true_filter,
        'r_0': baseline_rate,
        'F': nonlinearity_arctan
    }
    spikes, rate = gen_spiking(stimulus, model_params, dt)

    # Compute STA
    STA = spike_triggered_avg(stimulus, spikes, n_history, dt)

    # Scale STA
    if np.max(np.abs(STA)) > 0:
        STA_scaled = STA / np.max(np.abs(STA)) * np.max(np.abs(true_filter))
    else:
        STA_scaled = STA

    # Calculate correlation
    correlation = np.corrcoef(true_filter, STA)[0, 1] if np.std(STA) > 0 else 0

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Stimulus sample
    t_plot = np.arange(min(500, n_samples)) * dt
    axes[0, 0].plot(t_plot, stimulus[:len(t_plot)], 'g-', linewidth=0.8)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Stimulus')
    axes[0, 0].set_title(f'Stimulus Type: {stim_label}')
    axes[0, 0].grid(True, alpha=0.3)

    # Stimulus autocorrelation
    autocorr = np.correlate(stimulus[:1000], stimulus[:1000], mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / autocorr[0]
    lags = np.arange(len(autocorr)) * dt * 1000
    axes[0, 1].plot(lags[:200], autocorr[:200], 'b-', linewidth=1.5)
    axes[0, 1].set_xlabel('Lag (ms)')
    axes[0, 1].set_ylabel('Autocorrelation')
    axes[0, 1].set_title('Stimulus Autocorrelation')
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].grid(True, alpha=0.3)

    # True filter vs STA
    axes[1, 0].plot(history_timepoints * 1000, true_filter, 'b-', linewidth=2, label='True Filter')
    axes[1, 0].plot(history_timepoints * 1000, STA_scaled, 'r--', linewidth=2, label='STA (scaled)')
    axes[1, 0].set_xlabel('Time Before Spike (ms)')
    axes[1, 0].set_ylabel('Filter Weight')
    axes[1, 0].set_title(f'Filter Recovery (r = {correlation:.3f})')
    axes[1, 0].legend()
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].grid(True, alpha=0.3)

    # Quality assessment
    quality_color = 'green' if correlation > 0.9 else ('orange' if correlation > 0.5 else 'red')
    quality_text = 'Excellent!' if correlation > 0.9 else ('Moderate' if correlation > 0.5 else 'Poor')

    axes[1, 1].text(0.5, 0.6, f'Correlation: {correlation:.3f}', fontsize=20,
                    ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.4, quality_text, fontsize=24, fontweight='bold',
                    color=quality_color, ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.2, f'{int(np.sum(spikes))} spikes', fontsize=14,
                    ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('STA Quality')

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_spatial_receptive_field(filter_type='gabor', center_width=0.2, frequency=6, baseline_rate=10):
    """
    Explore different spatial receptive field types.

    Parameters
    ----------
    filter_type : str
        Type of spatial filter: 'center_surround', 'gabor', 'edge_detector'
    center_width : float
        Width of the center (for center-surround) or envelope (for Gabor)
    frequency : float
        Spatial frequency (for Gabor)
    baseline_rate : float
        Baseline firing rate
    """
    n_spatial = 50
    spatial_locations = np.linspace(-1, 1, n_spatial)

    # Create spatial filter based on type
    if filter_type == 'center_surround':
        # Mexican hat / difference of Gaussians
        center = np.exp(-spatial_locations ** 2 / (2 * center_width ** 2))
        surround = np.exp(-spatial_locations ** 2 / (2 * (center_width * 3) ** 2))
        spatial_filter = center - 0.5 * surround
        filter_label = 'Center-Surround (like LGN)'
    elif filter_type == 'gabor':
        # Gabor filter (oriented edge detector)
        envelope = np.exp(-spatial_locations ** 2 / (2 * center_width ** 2))
        spatial_filter = envelope * np.sin(spatial_locations * frequency)
        filter_label = 'Gabor (like V1 Simple Cell)'
    elif filter_type == 'edge_detector':
        # Simple edge detector
        spatial_filter = np.zeros(n_spatial)
        mid = n_spatial // 2
        width = int(center_width * n_spatial)
        spatial_filter[mid - width:mid] = 1
        spatial_filter[mid:mid + width] = -1
        filter_label = 'Edge Detector'
    else:
        spatial_filter = np.exp(-spatial_locations ** 2 / (2 * center_width ** 2))
        filter_label = 'Gaussian'

    # Normalize
    spatial_filter = spatial_filter / np.max(np.abs(spatial_filter)) * 10

    # Calculate response to stimulus at each location
    responses = np.zeros(n_spatial)
    for i, pos in enumerate(spatial_locations):
        # Create point stimulus at this location
        stim_snapshot = np.exp(-(spatial_locations - pos) ** 2 / (2 * 0.1 ** 2))
        L = np.sum(spatial_filter * stim_snapshot)
        responses[i] = baseline_rate + nonlinearity_arctan(L)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Spatial filter
    axes[0, 0].plot(spatial_locations, spatial_filter, 'b-', linewidth=2)
    axes[0, 0].fill_between(spatial_locations, 0, spatial_filter,
                            where=spatial_filter > 0, alpha=0.3, color='blue', label='Excitatory')
    axes[0, 0].fill_between(spatial_locations, 0, spatial_filter,
                            where=spatial_filter < 0, alpha=0.3, color='red', label='Inhibitory')
    axes[0, 0].set_xlabel('Spatial Position')
    axes[0, 0].set_ylabel('Filter Weight $k_i$')
    axes[0, 0].set_title(f'Spatial Receptive Field: {filter_label}')
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2D visualization of receptive field
    X, Y = np.meshgrid(spatial_locations, spatial_locations)
    if filter_type == 'gabor':
        RF_2d = np.exp(-(X ** 2 + Y ** 2) / (2 * center_width ** 2)) * np.sin(X * frequency)
    elif filter_type == 'center_surround':
        center_2d = np.exp(-(X ** 2 + Y ** 2) / (2 * center_width ** 2))
        surround_2d = np.exp(-(X ** 2 + Y ** 2) / (2 * (center_width * 3) ** 2))
        RF_2d = center_2d - 0.5 * surround_2d
    else:
        RF_2d = np.outer(spatial_filter, np.ones(n_spatial))

    im = axes[0, 1].imshow(RF_2d, extent=[-1, 1, -1, 1], cmap='RdBu_r', origin='lower')
    axes[0, 1].set_xlabel('X Position')
    axes[0, 1].set_ylabel('Y Position')
    axes[0, 1].set_title('2D Receptive Field')
    plt.colorbar(im, ax=axes[0, 1], label='Weight')

    # Response profile
    axes[1, 0].plot(spatial_locations, responses, 'r-', linewidth=2)
    axes[1, 0].axhline(y=baseline_rate, color='k', linestyle='--', alpha=0.5, label=f'$r_0$={baseline_rate}')
    axes[1, 0].set_xlabel('Stimulus Position')
    axes[1, 0].set_ylabel('Firing Rate (Hz)')
    axes[1, 0].set_title('Response to Point Stimulus at Each Location')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Example stimuli and responses
    test_positions = [-0.5, 0, 0.5]
    colors = ['blue', 'green', 'orange']

    for pos, color in zip(test_positions, colors):
        stim = np.exp(-(spatial_locations - pos) ** 2 / (2 * 0.1 ** 2))
        axes[1, 1].plot(spatial_locations, stim, color=color, linewidth=1.5,
                        label=f'Stim at {pos}', alpha=0.7)

    axes[1, 1].set_xlabel('Spatial Position')
    axes[1, 1].set_ylabel('Stimulus Intensity')
    axes[1, 1].set_title('Example Point Stimuli')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()
