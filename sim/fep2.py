import numpy as np

# Define generative model components
P_o_given_s = {
  "apple": np.array([0.8, 0.2, 0.9]),  # Features: [red, round, shiny]
  "tomato": np.array([0.7, 0.6, 0.8])
}
initial_P_s = {"apple": 0.6, "tomato": 0.4}  # Initial prior belief P(s)

# Number of timesteps
T = 10

# Observations over time (could be the same or different)
observations = [
  np.array([0.0, 0.5, 1.0]) for _ in range(T)
]

# Initialize posterior belief q(s) at t=0
q_s = initial_P_s.copy()

# Learning rate
eta = 0.1

# Variance of the Gaussian likelihood
variance = 1.0

# Small constants to prevent division by zero and log(0)
epsilon = 1e-10
min_prob = 1e-6  # Minimum probability to enforce

# Store beliefs over time
beliefs_over_time = []

for t in range(T):
  o = observations[t]

  # Set prior for current timestep to previous posterior
  P_s = q_s.copy()

  # Initialize free energy minimization variables
  max_iterations = 100
  tolerance = 1e-6
  previous_F = np.inf

  # Free energy minimization loop
  for iteration in range(max_iterations):
    # Generate predictions
    predicted_o = sum(q_s[state] * P_o_given_s[state] for state in P_s)

    # Compute Free Energy
    prediction_error_term = sum(
      q_s[state] * np.sum((o - P_o_given_s[state])**2) / (2 * variance)
        for state in P_s
    )
    complexity_term = sum(
      q_s[state] * np.log((q_s[state] + epsilon) / (P_s[state] + epsilon))
        for state in P_s
    )
    F = prediction_error_term + complexity_term

    # Gradient of F with respect to q(s)
    gradients = {}
    for state in q_s:
      likelihood = P_o_given_s[state]
      log_likelihood = -np.sum((o - likelihood)**2) / (2 * variance)
      gradients[state] = np.log((q_s[state] + epsilon) / (P_s[state] + epsilon)) - log_likelihood + 1

    # Update q(s) using gradient descent
    for state in q_s:
      q_s[state] -= eta * gradients[state]

    # Project q(s) onto the probability simplex
    probs = np.array(list(q_s.values()))
    probs = np.maximum(probs, min_prob)  # Enforce minimum probability
    total = probs.sum()
    q_s = {state: prob / total for state, prob in zip(q_s.keys(), probs)}

    # Check for convergence
    if np.abs(F - previous_F) < tolerance:
      break
    previous_F = F

  # Store the updated beliefs
  beliefs_over_time.append(q_s.copy())

  # Optionally, print the results at each timestep
  print(f"Timestep {t + 1}:")
  print(f"  Observation: {o}")
  print(f"  Predicted o: {predicted_o}")
  print(f"  Free Energy: {F}")
  print(f"  Updated Beliefs: {q_s}")
  print()
