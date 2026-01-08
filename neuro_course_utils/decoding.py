import numpy as np
import matplotlib.pyplot as plt

def plot_poisson_examples(rate, duration, dt):
    """Visualize example Poisson spike trains"""
    time_points = np.arange(0, duration, dt)
    n_time_points = len(time_points)

    n_examples = 5
    spikes = np.zeros((n_examples, n_time_points))

    for i in range(n_examples):
        spikes[i] = (np.random.uniform(0, 1, n_time_points) < rate * dt)

    plt.figure(figsize=(12, 6))
    for i in range(n_examples):
        plt.plot(time_points, spikes[i] * (i + 1), '|', markersize=10,
                 label=f'Trial {i + 1}')

    plt.title(f'Example Poisson Spike Trains (Rate: {rate} Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Trial #')
    plt.ylim(0.5, n_examples + 0.5)
    plt.grid(True)
    plt.show()


def generate_spike_counts(rate, T, dt, n_trials):
    """Generate spike counts for multiple trials"""
    n_time_points = int(T / dt)
    spike_counts = np.zeros(n_trials)

    for i in range(n_trials):
        spikes = (np.random.uniform(0, 1, n_time_points) < rate * dt)
        spike_counts[i] = np.sum(spikes)

    return spike_counts


def plot_spike_count_distributions(r_apple, r_banana, T, dt, P_apple, n_trials):
    """Plot spike count distributions for apple and banana stimuli"""
    n_apple_trials = int(P_apple * n_trials)
    n_banana_trials = int((1 - P_apple) * n_trials)
    P_banana = 1 - P_apple

    # Generate spike counts
    apple_counts = generate_spike_counts(r_apple, T, dt, n_apple_trials)
    banana_counts = generate_spike_counts(r_banana, T, dt, n_banana_trials)

    # Calculate theoretical thresholds
    ml_threshold = T * (r_apple - r_banana) / np.log(r_apple / r_banana)
    map_threshold = ml_threshold + np.log(P_banana / P_apple) / np.log(r_apple / r_banana)

    # Plotting
    bin_size = 1
    bin_min = 0
    bin_max = max(np.max(apple_counts), np.max(banana_counts)) + 5
    bin_centers = np.arange(bin_min, bin_max, bin_size)
    bin_edges = np.arange(bin_min - bin_size / 2, bin_max + bin_size / 2, bin_size)

    apple_hist, _ = np.histogram(apple_counts, bins=bin_edges, density=True)
    banana_hist, _ = np.histogram(banana_counts, bins=bin_edges, density=True)

    plt.figure(figsize=(12, 6))
    plt.bar(bin_centers, apple_hist, alpha=0.5, label='Apple', color='red')
    plt.bar(bin_centers, banana_hist, alpha=0.5, label='Banana', color='yellow')

    plt.axvline(x=ml_threshold, color='k', linestyle='--',
                label=f'ML Threshold: {ml_threshold:.1f}')
    plt.axvline(x=map_threshold, color='r', linestyle='--',
                label=f'MAP Threshold: {map_threshold:.1f}')

    plt.title(f'Spike Count Distributions (T={T}s, P(apple)={P_apple:.2f})')
    plt.xlabel('Spike Count')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

    correct_apple = np.mean(apple_counts > map_threshold)
    correct_banana = np.mean(banana_counts < map_threshold)
    total_accuracy = P_apple * correct_apple + P_banana * correct_banana

    print(f"\nClassification Results (using MAP threshold):")
    print(f"Correct Apple Classification: {correct_apple * 100:.1f}%")
    print(f"Correct Banana Classification: {correct_banana * 100:.1f}%")
    print(f"Overall Accuracy: {total_accuracy * 100:.1f}%")

    return map_threshold, total_accuracy


def calculate_information_metrics(r_apple, r_banana, P_apple, dt):
    """Calculate various information theory metrics"""
    P_banana = 1 - P_apple
    r_avg = P_apple * r_apple + P_banana * r_banana

    p_spike_apple = r_apple * dt
    p_spike_banana = r_banana * dt
    p_spike_avg = r_avg * dt

    H_code = -p_spike_avg * np.log2(p_spike_avg) - (1 - p_spike_avg) * np.log2(1 - p_spike_avg)

    H_noise_apple = -p_spike_apple * np.log2(p_spike_apple) - (1 - p_spike_apple) * np.log2(1 - p_spike_apple)
    H_noise_banana = -p_spike_banana * np.log2(p_spike_banana) - (1 - p_spike_banana) * np.log2(1 - p_spike_banana)

    H_noise = P_apple * H_noise_apple + P_banana * H_noise_banana
    I = H_code - H_noise
    info_rate = I / dt

    return H_code, H_noise_apple, H_noise_banana, H_noise, I, info_rate


def plot_information_analysis(r_apple, r_banana, P_apple):
    """Visualize information theory metrics"""
    dt = 0.001

    H_code, H_noise_apple, H_noise_banana, H_noise, I, info_rate = \
        calculate_information_metrics(r_apple, r_banana, P_apple, dt)

    plt.figure(figsize=(10, 6))
    metrics = ['Code Entropy', 'Noise Entropy\n(Apple)', 'Noise Entropy\n(Banana)',
               'Overall\nNoise Entropy', 'Mutual\nInformation']
    values = [H_code, H_noise_apple, H_noise_banana, H_noise, I]

    plt.bar(metrics, values)
    plt.title('Information Theory Metrics')
    plt.ylabel('Bits')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\nInformation Rate: {info_rate:.2f} bits/second")
