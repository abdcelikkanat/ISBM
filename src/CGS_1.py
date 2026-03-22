import numpy as np

# --- 0. HYPERPARAMETERS & MOCK DATA ---
N = 20  # Number of nodes
X = np.random.randint(0, 2, size=(N, N))  # Mock network graph
np.fill_diagonal(X, 0)  # No self-loops

alpha = 1.0  # CRP prior hyperparameter (for new clusters)
a = 1.0  # Beta prior hyperparameter (successes)
b = 1.0  # Beta prior hyperparameter (failures)
num_iterations = 100  # How many times to run the Gibbs sampler

# --- 1. RANDOM INITIALIZATION ---
K = 3  # Starting guess for number of clusters
Z = np.random.randint(0, K, size=N).tolist()

# Pre-allocate tally arrays to their maximum possible size (N)
m_counts = np.zeros(N)
n_counts = np.zeros((N, N))
N_counts = np.zeros((N, N))

# Populate initial tallies based on our random guess
for i in range(N):
    m_counts[Z[i]] += 1
    for j in range(N):
        if i == j:
            continue
        if X[i, j] == 1:
            n_counts[Z[i], Z[j]] += 1
        else:
            N_counts[Z[i], Z[j]] += 1

# --- 2. THE GIBBS SAMPLING LOOP ---
for iteration in range(num_iterations):
    for i in range(N):
        # A. Remove node i's current influence
        current_cluster = Z[i]
        m_counts[current_cluster] -= 1

        for j in range(N):
            if i == j:
                continue
            if X[i, j] == 1:
                n_counts[current_cluster, Z[j]] -= 1
            else:
                N_counts[current_cluster, Z[j]] -= 1

        # B. Calculate log probabilities for all options (K existing + 1 new)
        log_scores = np.zeros(K + 1)

        for k in range(K + 1):
            # Prior
            if k < K:
                prior = m_counts[k]  # Existing table
                if prior == 0:  # Skip completely empty tables
                    log_scores[k] = -np.inf
                    continue
            else:
                prior = alpha  # Brand new table

            log_total_score = np.log(prior)

            # Likelihood
            log_likelihood = 0.0
            for j in range(N):
                if i == j:
                    continue
                l = Z[j]

                denominator = n_counts[k, l] + N_counts[k, l] + a + b
                if X[i, j] == 1:
                    prob = (n_counts[k, l] + a) / denominator
                else:
                    prob = (N_counts[k, l] + b) / denominator

                log_likelihood += np.log(prob)

            log_scores[k] = log_total_score + log_likelihood

        # C. The Log-Sum-Exp Trick to safely convert back to standard probabilities
        max_log_score = np.max(log_scores)
        safe_scores = log_scores - max_log_score
        probabilities = np.exp(safe_scores)
        probabilities = probabilities / np.sum(probabilities)  # Normalize

        # D. Choose the new cluster
        cluster_options = np.arange(K + 1)
        new_cluster = np.random.choice(cluster_options, p=probabilities)
        Z[i] = new_cluster

        # E. Update K if a new cluster was chosen
        if new_cluster == K:
            K += 1

        # F. Add node i's statistics to its new home
        m_counts[new_cluster] += 1
        for j in range(N):
            if i == j:
                continue
            if X[i, j] == 1:
                n_counts[new_cluster, Z[j]] += 1
            else:
                N_counts[new_cluster, Z[j]] += 1

    # Optional: Print progress
    if iteration % 10 == 0:
        print(f"Iteration {iteration} | Active Clusters (K): {K}")
