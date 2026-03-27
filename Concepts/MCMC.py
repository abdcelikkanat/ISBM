import numpy as np


def target_distribution(x):
    """
    The 'true' distribution we want to sample from.
    In the IRM model, this was your massive probability equation.
    Here, it's just a simple bimodal curve (two peaks at -3 and 3).
    """
    return np.exp(-0.5 * (x - 3) ** 2) + np.exp(-0.5 * (x + 3) ** 2)


def simple_mcmc(num_iterations=10000, burn_in=2000):
    samples = []  # Equivalent to your self.z1_samples_
    ll_trace = []  # Equivalent to your self.ll_trace_

    # 1. Start at an arbitrary, random initial state
    current_x = 0.0

    for i in range(num_iterations):
        # 2. Propose a new state (take a random step nearby)
        # This is the "Markov Chain" part - the next step depends only on the current step
        proposed_x = np.random.normal(loc=current_x, scale=2.0)

        # 3. Calculate how "good" the states are
        p_current = target_distribution(current_x)
        p_proposed = target_distribution(proposed_x)

        # 4. Calculate the Acceptance Ratio
        # If proposed > current, ratio is > 1.
        # If proposed < current, ratio is a fraction between 0 and 1.
        acceptance_ratio = p_proposed / p_current

        # 5. The Accept/Reject Logic
        # If the ratio is > 1 (a better state), np.random.rand() is always < ratio, so we accept.
        # If the ratio is < 1 (a worse state), we only accept it SOME of the time.
        if np.random.rand() < acceptance_ratio:
            current_x = proposed_x  # Move to the new state
            p_current = p_proposed  # Update our current probability

        # 6. Track the Log-Likelihood (Score)
        # We take the log just to mimic how real models prevent underflow
        ll_trace.append(np.log(p_current))

        # 7. Save the valid samples (ignoring the initial noisy guesses)
        if i >= burn_in:
            samples.append(current_x)

    return samples, ll_trace


# --- Run the algorithm ---
valid_samples, log_likelihood_history = simple_mcmc()

print(f"Collected {len(valid_samples)} valid samples after burn-in.")
print(
    f"Mean of samples (should be close to 0 due to symmetry): {np.mean(valid_samples):.2f}"
)
