"""
Poisson Spiking Lab - Interactive Visualization Module

This module provides functions for generating and analyzing Poisson spike trains
for the computational neuroscience course.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# =============================================================================
# PART 1: Spike Train Generation
# =============================================================================

def generate_spike_train_basic(rate, duration=10, dt=0.001):
    """
    Generate a Poisson spike train using a basic loop implementation.

    Parameters
    ----------
    rate : float
        Firing rate in Hz
    duration : float
        Duration of spike train in seconds
    dt : float
        Time step in seconds

    Returns
    -------
    time_points : ndarray
        Array of time points
    spikes : ndarray
        Binary array where 1 indicates a spike
    """
    time_points = np.arange(0, duration, dt)
    spikes = np.zeros(len(time_points))
    for i in range(len(time_points)):
        rand = np.random.uniform()
        if rand < rate * dt:
            spikes[i] = 1
    return time_points, spikes


def generate_spike_train(rate, duration=10, dt=0.001):
    """
    Generate a Poisson spike train using vectorized operations.

    Parameters
    ----------
    rate : float
        Firing rate in Hz
    duration : float
        Duration of spike train in seconds
    dt : float
        Time step in seconds

    Returns
    -------
    time_points : ndarray
        Array of time points
    spikes : ndarray
        Binary array where 1 indicates a spike
    """
    time_points = np.arange(0, duration, dt)
    rand_array = np.random.uniform(size=len(time_points))
    spikes = (rand_array < rate * dt).astype(int)
    return time_points, spikes


def generate_periodic_spike_train(rate, duration=10, dt=0.001):
    """
    Generate a periodic (regular) spike train.

    Parameters
    ----------
    rate : float
        Firing rate in Hz
    duration : float
        Duration of spike train in seconds
    dt : float
        Time step in seconds

    Returns
    -------
    time_points : ndarray
        Array of time points
    spikes : ndarray
        Binary array where 1 indicates a spike
    """
    time_points = np.arange(0, duration, dt)
    spikes = np.zeros(len(time_points))

    if rate > 0:
        isi = 1.0 / rate  # Inter-spike interval
        spike_times = np.arange(isi, duration, isi)
        spike_indices = (spike_times / dt).astype(int)
        spike_indices = spike_indices[spike_indices < len(spikes)]
        spikes[spike_indices] = 1

    return time_points, spikes


# =============================================================================
# PART 1: Interactive Plots
# =============================================================================

def plot_spike_train(rate, duration=10):
    """
    Plot a Poisson spike train with given parameters.

    Parameters
    ----------
    rate : float
        Firing rate in Hz
    duration : float
        Duration in seconds
    """
    time_points, spikes = generate_spike_train(rate, duration)
    spike_times = time_points[spikes == 1]

    plt.figure(figsize=(12, 3))
    plt.eventplot(spike_times, lineoffsets=0.5, linelengths=0.8, colors='black')
    plt.title(f'Poisson Spike Train (Rate: {rate} Hz, Spikes: {int(np.sum(spikes))})')
    plt.xlabel('Time (s)')
    plt.ylabel('Spikes')
    plt.ylim(0, 1)
    plt.xlim(0, duration)
    plt.yticks([])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_varying_rate_train(rate_min, rate_max, duration=10):
    """
    Plot a spike train with linearly varying rate.

    Parameters
    ----------
    rate_min : float
        Minimum firing rate in Hz
    rate_max : float
        Maximum firing rate in Hz
    duration : float
        Duration in seconds
    """
    dt = 0.001
    time_points = np.arange(0, duration, dt)
    rates = np.linspace(rate_min, rate_max, len(time_points))

    rand_array = np.random.uniform(size=len(time_points))
    spikes = (rand_array < rates * dt).astype(int)
    spike_times = time_points[spikes == 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), height_ratios=[1, 1])

    # Plot instantaneous rate
    ax1.plot(time_points, rates, 'b-', linewidth=1.5)
    ax1.set_ylabel('Firing Rate (Hz)')
    ax1.set_title('Time-Varying Firing Rate (Linear Ramp)')
    ax1.set_ylim(0, max(rate_max + 1, 1))
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, duration)

    # Plot spikes
    ax2.eventplot(spike_times, lineoffsets=0.5, linelengths=0.8, colors='black')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Spikes')
    ax2.set_title(f'Spike Train (Total Spikes: {int(np.sum(spikes))})')
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, duration)
    ax2.yticks = []
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_poisson_vs_periodic(rate, duration=2):
    """
    Compare Poisson and periodic spike trains side by side.

    Parameters
    ----------
    rate : float
        Firing rate in Hz
    duration : float
        Duration in seconds
    """
    dt = 0.001

    # Generate both types
    time_poisson, spikes_poisson = generate_spike_train(rate, duration, dt)
    time_periodic, spikes_periodic = generate_periodic_spike_train(rate, duration, dt)

    spike_times_poisson = time_poisson[spikes_poisson == 1]
    spike_times_periodic = time_periodic[spikes_periodic == 1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)

    # Periodic
    axes[0].eventplot(spike_times_periodic, lineoffsets=0.5, linelengths=0.8, colors='blue')
    axes[0].set_ylabel('Periodic')
    axes[0].set_title(f'Periodic Spiking (Rate: {rate} Hz, Spikes: {int(np.sum(spikes_periodic))})')
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([])
    axes[0].grid(True, alpha=0.3)

    # Poisson
    axes[1].eventplot(spike_times_poisson, lineoffsets=0.5, linelengths=0.8, colors='red')
    axes[1].set_ylabel('Poisson')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title(f'Poisson Spiking (Rate: {rate} Hz, Spikes: {int(np.sum(spikes_poisson))})')
    axes[1].set_ylim(0, 1)
    axes[1].set_yticks([])
    axes[1].grid(True, alpha=0.3)

    plt.xlim(0, duration)
    plt.tight_layout()
    plt.show()


def plot_spike_count_distribution(rate, duration=0.5, n_trials=500):
    """
    Plot the distribution of spike counts across many trials.
    Overlay the theoretical Poisson distribution.

    Parameters
    ----------
    rate : float
        Firing rate in Hz
    duration : float
        Duration of each trial in seconds
    n_trials : int
        Number of trials to simulate
    """
    dt = 0.001

    # Generate spike counts for many trials
    spike_counts = np.zeros(n_trials)
    for i in range(n_trials):
        _, spikes = generate_spike_train(rate, duration, dt)
        spike_counts[i] = np.sum(spikes)

    # Calculate statistics
    mean_count = np.mean(spike_counts)
    var_count = np.var(spike_counts)
    ff = var_count / mean_count if mean_count > 0 else 0
    expected_count = rate * duration

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 5))

    max_count = int(np.max(spike_counts)) + 5
    bins = np.arange(-0.5, max_count + 1.5, 1)

    ax.hist(spike_counts, bins=bins, density=True, alpha=0.7, color='steelblue',
            edgecolor='black', label='Simulated')

    # Theoretical Poisson distribution
    x_theory = np.arange(0, max_count + 1)
    y_theory = stats.poisson.pmf(x_theory, expected_count)
    ax.plot(x_theory, y_theory, 'o-', color='orange', linewidth=2, markersize=6,
            label=f'Theoretical Poisson (λ={expected_count:.1f})')

    ax.axvline(x=mean_count, color='red', linestyle='--', linewidth=2,
               label=f'Mean = {mean_count:.2f}')
    ax.axvline(x=expected_count, color='orange', linestyle=':', linewidth=2,
               label=f'Expected = {expected_count:.1f}')

    ax.set_xlabel('Spike Count', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'Spike Count Distribution ({n_trials} trials)\n'
                 f'Rate={rate} Hz, Duration={duration}s, Fano Factor={ff:.3f}', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Expected spike count (rate × duration): {expected_count:.1f}")
    print(f"Observed mean: {mean_count:.2f}, Observed variance: {var_count:.2f}")
    print(f"Fano Factor (var/mean): {ff:.3f}  [Poisson prediction: 1.0]")


def plot_isi_distribution(rate, duration=10, n_bins=30):
    """
    Plot the inter-spike interval distribution and compare to theoretical exponential.

    Parameters
    ----------
    rate : float
        Firing rate in Hz
    duration : float
        Duration in seconds
    n_bins : int
        Number of histogram bins
    """
    dt = 0.001
    time_points, spikes = generate_spike_train(rate, duration, dt)
    spike_times = time_points[spikes == 1]

    if len(spike_times) < 2:
        print("Not enough spikes to compute ISI distribution. Try increasing rate or duration.")
        return

    # Calculate ISIs
    isis = np.diff(spike_times)

    # Calculate statistics
    mean_isi = np.mean(isis)
    std_isi = np.std(isis)
    cv = std_isi / mean_isi if mean_isi > 0 else 0
    expected_mean_isi = 1 / rate if rate > 0 else 0

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 5))

    # Histogram of ISIs
    counts, bin_edges, _ = ax.hist(isis, bins=n_bins, density=True, alpha=0.7,
                                   color='steelblue', edgecolor='black', label='Simulated ISIs')

    # Theoretical exponential distribution
    x_theory = np.linspace(0, np.max(isis), 200)
    y_theory = rate * np.exp(-rate * x_theory)
    ax.plot(x_theory, y_theory, '-', color='orange', linewidth=2.5,
            label=f'Theoretical Exponential (rate={rate} Hz)')

    ax.axvline(x=mean_isi, color='red', linestyle='--', linewidth=2,
               label=f'Mean ISI = {mean_isi * 1000:.1f} ms')
    ax.axvline(x=expected_mean_isi, color='orange', linestyle=':', linewidth=2,
               label=f'Expected = {expected_mean_isi * 1000:.1f} ms')

    ax.set_xlabel('Inter-Spike Interval (s)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'ISI Distribution ({len(isis)} intervals)\n'
                 f'CV = {cv:.3f}  [Poisson prediction: 1.0]', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Number of spikes: {len(spike_times)}, Number of ISIs: {len(isis)}")
    print(f"Expected mean ISI (1/rate): {expected_mean_isi * 1000:.2f} ms")
    print(f"Observed mean ISI: {mean_isi * 1000:.2f} ms, Std: {std_isi * 1000:.2f} ms")
    print(f"Coefficient of Variation (CV = std/mean): {cv:.3f}  [Poisson prediction: 1.0]")


# =============================================================================
# PART 2: Spike Train Analysis
# =============================================================================

def get_spike_times(spikes, time_points):
    """
    Extract spike times from a binary spike train.

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


def analyze_spike_train(spikes, time_points, bin_width=0.5, x_start=0, show_plots=True):
    """
    Analyze a spike train with multiple statistical measures.

    Parameters
    ----------
    spikes : ndarray
        Binary array of spike/no-spike
    time_points : ndarray
        Array of time points
    bin_width : float
        Width of bins for rate calculation (seconds)
    x_start : float
        Start time for plot display window
    show_plots : bool
        Whether to display plots

    Returns
    -------
    results : dict
        Dictionary containing calculated metrics
    """
    # Get spike times
    spike_times = get_spike_times(spikes, time_points)

    # Create time bins
    bin_edges = np.arange(time_points[0], time_points[-1], bin_width)
    bin_centers = bin_edges[:-1] + bin_width / 2

    # Calculate binned counts and rates
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    rates = counts / bin_width

    # Calculate Fano Factor (FF)
    mean_count = np.mean(counts)
    var_count = np.var(counts)
    FF = var_count / mean_count if mean_count > 0 else np.nan

    # Calculate Coefficient of Variation (CV) of ISIs
    if len(spike_times) > 1:
        ISIs = np.diff(spike_times)
        mean_isi = np.mean(ISIs)
        std_isi = np.std(ISIs)
        CV = std_isi / mean_isi if mean_isi > 0 else np.nan
    else:
        ISIs = np.array([])
        CV = np.nan

    # Calculate autocorrelation
    autocorr_max = 6  # Maximum lag in seconds
    autocorr_offsets = np.arange(0, autocorr_max, bin_width)
    auto_corr = np.zeros(len(autocorr_offsets))

    for j in range(len(auto_corr)):
        if len(counts) > j:
            shifted_product = (counts[j:] - mean_count) * (counts[:len(counts) - j] - mean_count)
            auto_corr[j] = shifted_product.sum()

    # Normalize autocorrelation
    if auto_corr[0] > 0:
        auto_corr = auto_corr / auto_corr[0]

    if show_plots:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), height_ratios=[1, 1, 1])

        # Raw spike train (raster style)
        spike_times_window = spike_times[(spike_times >= x_start) & (spike_times < x_start + 20)]
        axes[0].eventplot(spike_times_window, lineoffsets=0.5, linelengths=0.8, colors='black')
        axes[0].set_title(f'Raw Spike Train (CV={CV:.3f}, FF={FF:.3f})', fontsize=12)
        axes[0].set_ylabel('Spikes')
        axes[0].set_xlim([x_start, x_start + 20])
        axes[0].set_ylim(0, 1)
        axes[0].set_yticks([])
        axes[0].grid(True, alpha=0.3)

        # Binned rate
        bin_mask = (bin_centers >= x_start) & (bin_centers < x_start + 20)
        axes[1].bar(bin_centers[bin_mask], rates[bin_mask], width=bin_width * 0.9,
                    alpha=0.7, color='steelblue', edgecolor='black')
        axes[1].set_title(f'Firing Rate (bin width = {bin_width}s)', fontsize=12)
        axes[1].set_ylabel('Rate (Hz)')
        axes[1].set_xlim([x_start, x_start + 20])
        axes[1].grid(True, alpha=0.3)

        # Autocorrelation
        axes[2].plot(autocorr_offsets, auto_corr, 'b-o', linewidth=1.5, markersize=4)
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[2].set_title('Autocorrelation', fontsize=12)
        axes[2].set_xlabel('Lag (s)')
        axes[2].set_ylabel('Correlation')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"\nStatistics Summary:")
        print(f"  Total spikes: {len(spike_times)}")
        print(f"  Mean firing rate: {len(spike_times) / (time_points[-1] - time_points[0]):.2f} Hz")
        print(f"  Coefficient of Variation (CV): {CV:.3f}  [Poisson = 1.0]")
        print(f"  Fano Factor (FF): {FF:.3f}  [Poisson = 1.0]")

    return {
        'CV': CV,
        'FF': FF,
        'rates': rates,
        'counts': counts,
        'autocorr': auto_corr,
        'autocorr_offsets': autocorr_offsets,
        'spike_times': spike_times,
        'ISIs': ISIs
    }


# =============================================================================
# PART 3: Multi-Trial Analysis
# =============================================================================

def plot_spike_raster(all_spike_times, ax=None):
    """
    Plot raster of spike times across trials.

    Parameters
    ----------
    all_spike_times : list of ndarray
        List where each element contains spike times for one trial
    ax : matplotlib axis, optional
        Axis to plot on. If None, uses current axis.
    """
    if ax is None:
        ax = plt.gca()

    for trial in range(len(all_spike_times)):
        spike_times = all_spike_times[trial]
        ax.scatter(spike_times, np.full_like(spike_times, trial),
                   marker='|', s=10, c='black', linewidths=0.5)


def analyze_trial_data(spike_data, bin_width=0.1, trial_number=0):
    """
    Analyze spike data across multiple trials.

    Parameters
    ----------
    spike_data : dict
        Dictionary containing 'spikes', 'time_points', 'n_trials'
    bin_width : float
        Width of time bins for rate calculation
    trial_number : int
        Which trial to highlight in single-trial plot

    Returns
    -------
    results : dict
        Dictionary containing calculated metrics
    """
    # Extract basic info
    spikes = spike_data['spikes']
    time_points = spike_data['time_points']
    n_trials = spike_data['n_trials']

    # Get spike times for each trial
    all_spike_times = []
    spike_counts = np.zeros(n_trials)

    for trial in range(n_trials):
        trial_spike_times = time_points[spikes[:, trial] == 1]
        all_spike_times.append(trial_spike_times)
        spike_counts[trial] = len(trial_spike_times)

    # Calculate trial-by-trial Fano Factor
    mean_count = np.mean(spike_counts)
    var_count = np.var(spike_counts)
    trial_FF = var_count / mean_count if mean_count > 0 else np.nan

    # Create time bins
    bin_edges = np.arange(time_points[0], time_points[-1] + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + bin_width / 2

    # Calculate rates for each trial
    binned_counts = np.zeros((len(bin_centers), n_trials))
    for trial in range(n_trials):
        counts, _ = np.histogram(all_spike_times[trial], bins=bin_edges)
        binned_counts[:, trial] = counts

    binned_rates = binned_counts / bin_width
    trial_avg_rate = np.mean(binned_rates, axis=1)

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), height_ratios=[2, 1, 1])

    # Raster plot
    plot_spike_raster(all_spike_times, ax=axes[0])
    axes[0].set_ylabel('Trial Number')
    axes[0].set_title(f'Spike Raster ({n_trials} trials, FF across trials = {trial_FF:.3f})', fontsize=12)
    axes[0].set_ylim(-0.5, n_trials - 0.5)
    axes[0].grid(True, alpha=0.3)

    # Single trial rate
    axes[1].bar(bin_centers, binned_rates[:, trial_number], width=bin_width * 0.9,
                alpha=0.7, color='steelblue', edgecolor='black')
    axes[1].set_ylabel('Rate (Hz)')
    axes[1].set_title(f'Single Trial Rate (Trial {trial_number}, Spikes: {int(spike_counts[trial_number])})',
                      fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Trial-averaged rate
    axes[2].bar(bin_centers, trial_avg_rate, width=bin_width * 0.9,
                alpha=0.7, color='darkorange', edgecolor='black')
    axes[2].set_ylabel('Rate (Hz)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Trial-Averaged Rate', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\nTrial Statistics:")
    print(f"  Number of trials: {n_trials}")
    print(f"  Mean spike count per trial: {mean_count:.2f}")
    print(f"  Variance of spike counts: {var_count:.2f}")
    print(f"  Fano Factor (across trials): {trial_FF:.3f}  [Poisson = 1.0]")
    print(f"  Min/Max spikes in a trial: {int(np.min(spike_counts))}/{int(np.max(spike_counts))}")

    return {
        'spike_counts': spike_counts,
        'trial_FF': trial_FF,
        'binned_rates': binned_rates,
        'trial_avg_rate': trial_avg_rate,
        'all_spike_times': all_spike_times
    }
