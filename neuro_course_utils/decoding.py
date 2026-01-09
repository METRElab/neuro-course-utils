"""
Decoding Lab - Interactive Visualization Module

This module provides functions for exploring neural decoding and information theory
for the computational neuroscience course.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_spike_counts(rate, T, dt, n_trials):
    """
    Generate spike counts for multiple trials of a Poisson process.

    Parameters
    ----------
    rate : float
        Firing rate in Hz
    T : float
        Duration of each trial in seconds
    dt : float
        Time step in seconds
    n_trials : int
        Number of trials to simulate

    Returns
    -------
    spike_counts : ndarray
        Array of spike counts for each trial
    """
    n_time_points = int(T / dt)
    spike_counts = np.zeros(n_trials)

    for i in range(n_trials):
        spikes = np.random.uniform(0, 1, n_time_points) < rate * dt
        spike_counts[i] = np.sum(spikes)

    return spike_counts


def calculate_ml_threshold(r_apple, r_banana, T):
    """
    Calculate the Maximum Likelihood threshold for Poisson spike counts.

    The ML threshold is where P(N|Apple) = P(N|Banana).
    For Poisson distributions, this is: N* = T * (r_A - r_B) / ln(r_A / r_B)

    Parameters
    ----------
    r_apple : float
        Apple firing rate in Hz
    r_banana : float
        Banana firing rate in Hz
    T : float
        Integration time in seconds

    Returns
    -------
    threshold : float
        ML decision threshold (spike count)
    """
    if r_apple == r_banana:
        return r_apple * T
    return T * (r_apple - r_banana) / np.log(r_apple / r_banana)


def calculate_map_threshold(r_apple, r_banana, T, P_apple):
    """
    Calculate the Maximum A Posteriori threshold for Poisson spike counts.

    The MAP threshold incorporates prior probabilities.

    Parameters
    ----------
    r_apple : float
        Apple firing rate in Hz
    r_banana : float
        Banana firing rate in Hz
    T : float
        Integration time in seconds
    P_apple : float
        Prior probability of apple

    Returns
    -------
    threshold : float
        MAP decision threshold (spike count)
    """
    P_banana = 1 - P_apple
    ml_threshold = calculate_ml_threshold(r_apple, r_banana, T)

    if r_apple == r_banana or P_apple == P_banana:
        return ml_threshold

    # MAP adjustment based on priors
    map_threshold = ml_threshold + np.log(P_banana / P_apple) / np.log(r_apple / r_banana)
    return map_threshold


def calculate_classification_accuracy(apple_counts, banana_counts, threshold, P_apple):
    """
    Calculate classification accuracy for a given threshold.

    Parameters
    ----------
    apple_counts : ndarray
        Spike counts from apple trials
    banana_counts : ndarray
        Spike counts from banana trials
    threshold : float
        Decision threshold
    P_apple : float
        Prior probability of apple

    Returns
    -------
    accuracy_apple : float
        Fraction of apple trials correctly classified
    accuracy_banana : float
        Fraction of banana trials correctly classified
    accuracy_total : float
        Overall weighted accuracy
    """
    P_banana = 1 - P_apple

    accuracy_apple = np.mean(apple_counts >= threshold)
    accuracy_banana = np.mean(banana_counts < threshold)
    accuracy_total = P_apple * accuracy_apple + P_banana * accuracy_banana

    return accuracy_apple, accuracy_banana, accuracy_total


def calculate_mutual_information(r_apple, r_banana, P_apple, dt):
    """
    Calculate mutual information between stimulus and neural response.

    Uses the formulation: I(S; N) = H(S) - H(S|N)

    Parameters
    ----------
    r_apple : float
        Apple firing rate in Hz
    r_banana : float
        Banana firing rate in Hz
    P_apple : float
        Prior probability of apple
    dt : float
        Time bin in seconds

    Returns
    -------
    results : dict
        Dictionary containing all information theory metrics
    """
    P_banana = 1 - P_apple

    # Stimulus entropy
    H_stim = -P_apple * np.log2(P_apple) - P_banana * np.log2(P_banana)

    # Probability of spike given stimulus
    P_spike_given_apple = r_apple * dt
    P_spike_given_banana = r_banana * dt

    # Ensure probabilities are valid (not > 1)
    P_spike_given_apple = min(P_spike_given_apple, 1.0)
    P_spike_given_banana = min(P_spike_given_banana, 1.0)

    # Total probability of spike
    P_spike = P_apple * P_spike_given_apple + P_banana * P_spike_given_banana
    P_no_spike = 1 - P_spike

    # Posterior probabilities (Bayes' rule)
    if P_spike > 0:
        P_apple_given_spike = P_spike_given_apple * P_apple / P_spike
        P_banana_given_spike = P_spike_given_banana * P_banana / P_spike
    else:
        P_apple_given_spike = P_apple
        P_banana_given_spike = P_banana

    P_no_spike_given_apple = 1 - P_spike_given_apple
    P_no_spike_given_banana = 1 - P_spike_given_banana

    if P_no_spike > 0:
        P_apple_given_no_spike = P_no_spike_given_apple * P_apple / P_no_spike
        P_banana_given_no_spike = P_no_spike_given_banana * P_banana / P_no_spike
    else:
        P_apple_given_no_spike = P_apple
        P_banana_given_no_spike = P_banana

    # Conditional entropy H(S|spike)
    H_stim_given_spike = 0
    if P_apple_given_spike > 0:
        H_stim_given_spike -= P_apple_given_spike * np.log2(P_apple_given_spike)
    if P_banana_given_spike > 0:
        H_stim_given_spike -= P_banana_given_spike * np.log2(P_banana_given_spike)

    # Conditional entropy H(S|no spike)
    H_stim_given_no_spike = 0
    if P_apple_given_no_spike > 0:
        H_stim_given_no_spike -= P_apple_given_no_spike * np.log2(P_apple_given_no_spike)
    if P_banana_given_no_spike > 0:
        H_stim_given_no_spike -= P_banana_given_no_spike * np.log2(P_banana_given_no_spike)

    # Expected conditional entropy H(S|N)
    H_stim_given_N = P_spike * H_stim_given_spike + P_no_spike * H_stim_given_no_spike

    # Mutual information
    MI = H_stim - H_stim_given_N

    # Information rate
    info_rate = MI / dt

    return {
        'H_stim': H_stim,
        'H_stim_given_spike': H_stim_given_spike,
        'H_stim_given_no_spike': H_stim_given_no_spike,
        'H_stim_given_N': H_stim_given_N,
        'MI': MI,
        'info_rate': info_rate,
        'P_spike': P_spike,
        'P_spike_given_apple': P_spike_given_apple,
        'P_spike_given_banana': P_spike_given_banana,
        'P_apple_given_spike': P_apple_given_spike,
        'P_banana_given_spike': P_banana_given_spike
    }


# =============================================================================
# INTERACTIVE PLOTTING FUNCTIONS
# =============================================================================

def plot_spike_trains_comparison(r_apple=30, r_banana=20, duration=0.5):
    """
    Compare spike trains from apple vs. banana stimuli side by side.

    Parameters
    ----------
    r_apple : float
        Apple firing rate in Hz
    r_banana : float
        Banana firing rate in Hz
    duration : float
        Duration of each trial in seconds
    """
    dt = 0.001
    n_trials = 8
    time_points = np.arange(0, duration, dt)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Apple trials
    apple_counts = []
    for trial in range(n_trials):
        spikes = np.random.uniform(0, 1, len(time_points)) < r_apple * dt
        spike_times = time_points[spikes]
        apple_counts.append(np.sum(spikes))
        axes[0].scatter(spike_times, np.full_like(spike_times, trial),
                        marker='|', s=100, c='red', linewidths=1.5)

    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Trial')
    axes[0].set_title(f'Apple Responses (r = {r_apple} Hz)\nCounts: {apple_counts}')
    axes[0].set_ylim(-0.5, n_trials - 0.5)
    axes[0].set_xlim(0, duration)
    axes[0].set_yticks(range(n_trials))
    axes[0].grid(True, alpha=0.3)

    # Add mean count
    axes[0].axvline(x=duration, color='red', linestyle='--', alpha=0)  # invisible, for spacing
    axes[0].text(0.98, 0.98, f'Mean: {np.mean(apple_counts):.1f}',
                 transform=axes[0].transAxes, ha='right', va='top',
                 fontsize=12, fontweight='bold', color='darkred')

    # Banana trials
    banana_counts = []
    for trial in range(n_trials):
        spikes = np.random.uniform(0, 1, len(time_points)) < r_banana * dt
        spike_times = time_points[spikes]
        banana_counts.append(np.sum(spikes))
        axes[1].scatter(spike_times, np.full_like(spike_times, trial),
                        marker='|', s=100, c='goldenrod', linewidths=1.5)

    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Trial')
    axes[1].set_title(f'Banana Responses (r = {r_banana} Hz)\nCounts: {banana_counts}')
    axes[1].set_ylim(-0.5, n_trials - 0.5)
    axes[1].set_xlim(0, duration)
    axes[1].set_yticks(range(n_trials))
    axes[1].grid(True, alpha=0.3)

    axes[1].text(0.98, 0.98, f'Mean: {np.mean(banana_counts):.1f}',
                 transform=axes[1].transAxes, ha='right', va='top',
                 fontsize=12, fontweight='bold', color='darkgoldenrod')

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_spike_count_distributions(r_apple=30, r_banana=20, T=0.5, P_apple=0.5, n_trials=10000):
    """
    Plot spike count distributions with ML/MAP thresholds and theoretical Poisson overlay.

    Parameters
    ----------
    r_apple : float
        Apple firing rate in Hz
    r_banana : float
        Banana firing rate in Hz
    T : float
        Integration time in seconds
    P_apple : float
        Prior probability of apple
    n_trials : int
        Total number of trials
    """
    dt = 0.001
    P_banana = 1 - P_apple

    n_apple_trials = int(P_apple * n_trials)
    n_banana_trials = int(P_banana * n_trials)

    # Generate spike counts
    apple_counts = generate_spike_counts(r_apple, T, dt, n_apple_trials)
    banana_counts = generate_spike_counts(r_banana, T, dt, n_banana_trials)

    # Calculate thresholds
    ml_threshold = calculate_ml_threshold(r_apple, r_banana, T)
    map_threshold = calculate_map_threshold(r_apple, r_banana, T, P_apple)

    # Calculate accuracies
    acc_apple_ml, acc_banana_ml, acc_total_ml = calculate_classification_accuracy(
        apple_counts, banana_counts, ml_threshold, 0.5)
    acc_apple_map, acc_banana_map, acc_total_map = calculate_classification_accuracy(
        apple_counts, banana_counts, map_threshold, P_apple)

    # Histogram setup
    bin_min = 0
    bin_max = int(max(np.max(apple_counts), np.max(banana_counts)) + 5)
    bin_centers = np.arange(bin_min, bin_max)
    bin_edges = np.arange(bin_min - 0.5, bin_max + 0.5)

    apple_hist, _ = np.histogram(apple_counts, bins=bin_edges, density=True)
    banana_hist, _ = np.histogram(banana_counts, bins=bin_edges, density=True)

    # Theoretical Poisson distributions
    lambda_apple = r_apple * T
    lambda_banana = r_banana * T
    poisson_apple = stats.poisson.pmf(bin_centers, lambda_apple)
    poisson_banana = stats.poisson.pmf(bin_centers, lambda_banana)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histograms with thresholds
    axes[0].bar(bin_centers - 0.2, apple_hist, width=0.4, alpha=0.6,
                label='Apple (simulated)', color='red', edgecolor='darkred')
    axes[0].bar(bin_centers + 0.2, banana_hist, width=0.4, alpha=0.6,
                label='Banana (simulated)', color='gold', edgecolor='darkgoldenrod')

    # Theoretical distributions
    axes[0].plot(bin_centers, poisson_apple, 'r--', linewidth=2,
                 label=f'Poisson(λ={lambda_apple:.1f})')
    axes[0].plot(bin_centers, poisson_banana, 'y--', linewidth=2,
                 label=f'Poisson(λ={lambda_banana:.1f})')

    # Thresholds
    axes[0].axvline(x=ml_threshold, color='blue', linestyle='-', linewidth=2,
                    label=f'ML Threshold: {ml_threshold:.1f}')
    if abs(map_threshold - ml_threshold) > 0.1:
        axes[0].axvline(x=map_threshold, color='green', linestyle='-', linewidth=2,
                        label=f'MAP Threshold: {map_threshold:.1f}')

    axes[0].set_xlabel('Spike Count (N)', fontsize=12)
    axes[0].set_ylabel('Probability', fontsize=12)
    axes[0].set_title(f'Spike Count Distributions (T={T}s)', fontsize=12)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(bin_min - 1, bin_max)

    # Right: Accuracy summary
    categories = ['Apple\nCorrect', 'Banana\nCorrect', 'Overall']
    ml_accs = [acc_apple_ml * 100, acc_banana_ml * 100, acc_total_ml * 100]
    map_accs = [acc_apple_map * 100, acc_banana_map * 100, acc_total_map * 100]

    x = np.arange(len(categories))
    width = 0.35

    axes[1].bar(x - width / 2, ml_accs, width, label='ML Decoder', color='blue', alpha=0.7)
    axes[1].bar(x + width / 2, map_accs, width, label='MAP Decoder', color='green', alpha=0.7)

    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'Classification Accuracy\nP(Apple)={P_apple:.2f}', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].legend()
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add accuracy values as text
    for i, (ml, mp) in enumerate(zip(ml_accs, map_accs)):
        axes[1].text(i - width / 2, ml + 2, f'{ml:.1f}%', ha='center', fontsize=9)
        axes[1].text(i + width / 2, mp + 2, f'{mp:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()
    plt.close()

    print(f"Expected counts: Apple = {lambda_apple:.1f}, Banana = {lambda_banana:.1f}")
    print(f"ML Threshold: {ml_threshold:.2f}, MAP Threshold: {map_threshold:.2f}")


def plot_threshold_explorer(r_apple=30, r_banana=20, T=0.5, P_apple=0.5, threshold=12):
    """
    Interactive threshold explorer showing misclassification regions.

    Parameters
    ----------
    r_apple : float
        Apple firing rate in Hz
    r_banana : float
        Banana firing rate in Hz
    T : float
        Integration time in seconds
    P_apple : float
        Prior probability of apple
    threshold : float
        Decision threshold to explore
    """
    dt = 0.001
    P_banana = 1 - P_apple
    n_trials = 10000

    n_apple_trials = int(P_apple * n_trials)
    n_banana_trials = int(P_banana * n_trials)

    # Generate spike counts
    apple_counts = generate_spike_counts(r_apple, T, dt, n_apple_trials)
    banana_counts = generate_spike_counts(r_banana, T, dt, n_banana_trials)

    # Calculate optimal thresholds for reference
    ml_threshold = calculate_ml_threshold(r_apple, r_banana, T)
    map_threshold = calculate_map_threshold(r_apple, r_banana, T, P_apple)

    # Calculate accuracy for current threshold
    acc_apple, acc_banana, acc_total = calculate_classification_accuracy(
        apple_counts, banana_counts, threshold, P_apple)

    # Calculate accuracy for optimal threshold
    _, _, acc_optimal = calculate_classification_accuracy(
        apple_counts, banana_counts, map_threshold, P_apple)

    # Histogram setup
    bin_min = 0
    bin_max = int(max(np.max(apple_counts), np.max(banana_counts)) + 5)
    bin_centers = np.arange(bin_min, bin_max)
    bin_edges = np.arange(bin_min - 0.5, bin_max + 0.5)

    # Theoretical Poisson (scaled by prior for visualization)
    lambda_apple = r_apple * T
    lambda_banana = r_banana * T
    poisson_apple = stats.poisson.pmf(bin_centers, lambda_apple) * P_apple
    poisson_banana = stats.poisson.pmf(bin_centers, lambda_banana) * P_banana

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Distributions with error shading
    axes[0].fill_between(bin_centers, 0, poisson_apple,
                         where=bin_centers < threshold,
                         alpha=0.3, color='red', label='Apple errors (false negative)')
    axes[0].fill_between(bin_centers, 0, poisson_banana,
                         where=bin_centers >= threshold,
                         alpha=0.3, color='orange', label='Banana errors (false positive)')

    axes[0].plot(bin_centers, poisson_apple, 'r-', linewidth=2, label='P(N|Apple)·P(Apple)')
    axes[0].plot(bin_centers, poisson_banana, 'y-', linewidth=2, label='P(N|Banana)·P(Banana)')

    axes[0].axvline(x=threshold, color='black', linestyle='-', linewidth=3,
                    label=f'Your Threshold: {threshold:.1f}')
    axes[0].axvline(x=map_threshold, color='green', linestyle='--', linewidth=2,
                    label=f'Optimal (MAP): {map_threshold:.1f}')

    axes[0].set_xlabel('Spike Count (N)', fontsize=12)
    axes[0].set_ylabel('Probability × Prior', fontsize=12)
    axes[0].set_title('Threshold Explorer\nShaded regions = classification errors', fontsize=12)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(bin_min - 1, bin_max)

    # Right: Accuracy gauge
    ax2 = axes[1]

    # Create accuracy comparison
    labels = ['Your\nThreshold', 'Optimal\n(MAP)']
    accuracies = [acc_total * 100, acc_optimal * 100]
    colors = ['steelblue', 'green']

    bars = ax2.bar(labels, accuracies, color=colors, alpha=0.7, edgecolor='black')

    ax2.set_ylabel('Overall Accuracy (%)', fontsize=12)
    ax2.set_title('Classification Performance', fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add text labels
    for bar, acc in zip(bars, accuracies):
        ax2.text(bar.get_x() + bar.get_width() / 2, acc + 2, f'{acc:.1f}%',
                 ha='center', fontsize=14, fontweight='bold')

    # Add breakdown
    ax2.text(0.02, 0.98,
             f'Apple correct: {acc_apple * 100:.1f}%\nBanana correct: {acc_banana * 100:.1f}%',
             transform=ax2.transAxes, ha='left', va='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_time_integration(r_apple=30, r_banana=20, P_apple=0.5, max_T=2.0):
    """
    Show how classification accuracy improves with integration time.

    Parameters
    ----------
    r_apple : float
        Apple firing rate in Hz
    r_banana : float
        Banana firing rate in Hz
    P_apple : float
        Prior probability of apple
    max_T : float
        Maximum integration time to plot
    """
    dt = 0.001
    n_trials = 5000

    T_values = np.linspace(0.1, max_T, 20)
    accuracies_ml = []
    accuracies_map = []

    for T in T_values:
        n_apple_trials = int(P_apple * n_trials)
        n_banana_trials = int((1 - P_apple) * n_trials)

        apple_counts = generate_spike_counts(r_apple, T, dt, n_apple_trials)
        banana_counts = generate_spike_counts(r_banana, T, dt, n_banana_trials)

        ml_thresh = calculate_ml_threshold(r_apple, r_banana, T)
        map_thresh = calculate_map_threshold(r_apple, r_banana, T, P_apple)

        _, _, acc_ml = calculate_classification_accuracy(apple_counts, banana_counts, ml_thresh, 0.5)
        _, _, acc_map = calculate_classification_accuracy(apple_counts, banana_counts, map_thresh, P_apple)

        accuracies_ml.append(acc_ml * 100)
        accuracies_map.append(acc_map * 100)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Accuracy vs T
    axes[0].plot(T_values, accuracies_ml, 'b-o', linewidth=2, markersize=6, label='ML Decoder')
    axes[0].plot(T_values, accuracies_map, 'g-s', linewidth=2, markersize=6, label='MAP Decoder')
    axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')
    axes[0].axhline(y=100, color='gray', linestyle=':', alpha=0.5)

    axes[0].set_xlabel('Integration Time T (s)', fontsize=12)
    axes[0].set_ylabel('Classification Accuracy (%)', fontsize=12)
    axes[0].set_title('Accuracy vs. Integration Time', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(45, 105)

    # Right: Distribution separation visualization
    T_examples = [0.2, 0.5, 1.0]
    colors = ['lightblue', 'steelblue', 'darkblue']

    for T_ex, color in zip(T_examples, colors):
        lambda_a = r_apple * T_ex
        lambda_b = r_banana * T_ex
        x = np.arange(0, max(lambda_a, lambda_b) * 2)

        axes[1].plot(x, stats.poisson.pmf(x, lambda_a), '-', color=color,
                     linewidth=2, alpha=0.7, label=f'T={T_ex}s')
        axes[1].plot(x, stats.poisson.pmf(x, lambda_b), '--', color=color,
                     linewidth=2, alpha=0.7)

    axes[1].set_xlabel('Spike Count (N)', fontsize=12)
    axes[1].set_ylabel('Probability', fontsize=12)
    axes[1].set_title('Distribution Separation\n(solid=Apple, dashed=Banana)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()

    print(f"At T=0.5s: ML={accuracies_ml[np.argmin(np.abs(T_values - 0.5))]:.1f}%, "
          f"MAP={accuracies_map[np.argmin(np.abs(T_values - 0.5))]:.1f}%")
    print(f"At T=1.0s: ML={accuracies_ml[np.argmin(np.abs(T_values - 1.0))]:.1f}%, "
          f"MAP={accuracies_map[np.argmin(np.abs(T_values - 1.0))]:.1f}%")


def plot_information_accumulation(r_apple=30, r_banana=20, P_apple=0.75):
    """
    Show how mutual information accumulates over time.

    Parameters
    ----------
    r_apple : float
        Apple firing rate in Hz
    r_banana : float
        Banana firing rate in Hz
    P_apple : float
        Prior probability of apple
    """
    # Calculate information for different time bins
    dt_values = np.logspace(-4, -1, 50)  # 0.1ms to 100ms
    MI_values = []
    info_rates = []

    for dt in dt_values:
        results = calculate_mutual_information(r_apple, r_banana, P_apple, dt)
        MI_values.append(results['MI'])
        info_rates.append(results['info_rate'])

    # Calculate cumulative information over time (approximation)
    time_points = np.linspace(0.01, 2.0, 100)
    dt_fixed = 0.001
    results_fixed = calculate_mutual_information(r_apple, r_banana, P_apple, dt_fixed)
    info_rate_fixed = results_fixed['info_rate']
    cumulative_info = info_rate_fixed * time_points

    # Stimulus entropy (upper bound)
    P_banana = 1 - P_apple
    H_stim = -P_apple * np.log2(P_apple) - P_banana * np.log2(P_banana)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Information rate vs dt
    axes[0].semilogx(dt_values * 1000, info_rates, 'b-', linewidth=2)
    axes[0].set_xlabel('Time Bin dt (ms)', fontsize=12)
    axes[0].set_ylabel('Information Rate (bits/s)', fontsize=12)
    axes[0].set_title('Information Rate vs. Time Bin Size', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=info_rate_fixed, color='red', linestyle='--',
                    label=f'Rate at dt=1ms: {info_rate_fixed:.2f} bits/s')
    axes[0].legend()

    # Right: Cumulative information
    axes[1].plot(time_points, cumulative_info, 'g-', linewidth=2,
                 label='Cumulative MI')
    axes[1].axhline(y=H_stim, color='red', linestyle='--', linewidth=2,
                    label=f'Stimulus entropy H(S) = {H_stim:.3f} bits')
    axes[1].fill_between(time_points, 0, np.minimum(cumulative_info, H_stim),
                         alpha=0.3, color='green')

    axes[1].set_xlabel('Observation Time (s)', fontsize=12)
    axes[1].set_ylabel('Cumulative Information (bits)', fontsize=12)
    axes[1].set_title('Information Accumulation Over Time', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 2)

    plt.tight_layout()
    plt.show()
    plt.close()

    print(f"Information rate: {info_rate_fixed:.2f} bits/second")
    print(f"Stimulus entropy (upper bound): {H_stim:.4f} bits")
    print(f"Time to accumulate 0.5 bits: {0.5 / info_rate_fixed * 1000:.1f} ms")


def plot_information_components(r_apple=30, r_banana=20, P_apple=0.75):
    """
    Visualize the components of mutual information calculation.

    Parameters
    ----------
    r_apple : float
        Apple firing rate in Hz
    r_banana : float
        Banana firing rate in Hz
    P_apple : float
        Prior probability of apple
    """
    dt = 0.001
    results = calculate_mutual_information(r_apple, r_banana, P_apple, dt)

    P_banana = 1 - P_apple

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: Entropy breakdown
    labels = ['H(S)\nStimulus\nEntropy', 'H(S|N)\nConditional\nEntropy', 'I(S;N)\nMutual\nInformation']
    values = [results['H_stim'], results['H_stim_given_N'], results['MI']]
    colors = ['steelblue', 'coral', 'green']

    bars = axes[0].bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Bits', fontsize=12)
    axes[0].set_title('Information Theory Components\n(per time bin dt=1ms)', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                     f'{val:.4f}', ha='center', fontsize=10)

    # Middle: Probability tree
    ax1 = axes[1]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Probability Flow', fontsize=12)

    # Draw tree
    # Prior
    ax1.text(1, 8, f'P(A)={P_apple:.2f}', fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    ax1.text(1, 2, f'P(B)={P_banana:.2f}', fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    # Likelihood
    ax1.annotate('', xy=(4, 7.5), xytext=(2, 8),
                 arrowprops=dict(arrowstyle='->', color='black'))
    ax1.annotate('', xy=(4, 2.5), xytext=(2, 2),
                 arrowprops=dict(arrowstyle='->', color='black'))

    ax1.text(5, 7.5, f'P(spike|A)={results["P_spike_given_apple"]:.4f}', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax1.text(5, 2.5, f'P(spike|B)={results["P_spike_given_banana"]:.4f}', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Posterior
    ax1.text(5, 5, f'P(spike)={results["P_spike"]:.4f}', fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Right: Conditional entropy breakdown
    ax2 = axes[2]

    x = np.arange(2)
    width = 0.6

    cond_entropies = [results['H_stim_given_spike'], results['H_stim_given_no_spike']]
    weights = [results['P_spike'], 1 - results['P_spike']]
    weighted = [ce * w for ce, w in zip(cond_entropies, weights)]

    bars1 = ax2.bar(x - width / 4, cond_entropies, width / 2, label='H(S|observation)',
                    color='coral', alpha=0.7)
    bars2 = ax2.bar(x + width / 4, weighted, width / 2, label='Weighted contribution',
                    color='purple', alpha=0.7)

    ax2.set_xticks(x)
    ax2.set_xticklabels(['Given\nSpike', 'Given\nNo Spike'])
    ax2.set_ylabel('Bits', fontsize=12)
    ax2.set_title('Conditional Entropy Components', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add text
    for bar, val in zip(bars1, cond_entropies):
        if val > 0.001:
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                     f'{val:.4f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()
    plt.close()

    print(f"\nSummary:")
    print(f"  Stimulus entropy H(S) = {results['H_stim']:.6f} bits")
    print(f"  Conditional entropy H(S|N) = {results['H_stim_given_N']:.6f} bits")
    print(f"  Mutual information I(S;N) = {results['MI']:.6f} bits")
    print(f"  Information rate = {results['info_rate']:.2f} bits/second")
