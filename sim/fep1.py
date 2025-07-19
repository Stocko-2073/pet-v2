import numpy as np

# Define generative model components
P_o_given_s = {
  "apple": np.array([0.8, 0.2, 0.9]),
  "tomato": np.array([0.7, 0.6, 0.8])
}
P_s = {"apple": 0.6, "tomato": 0.4}

# Observation
o = np.array([1.0, 0.5, 1.0])

# Initialize posterior belief q(s)
q_s = P_s.copy()

# Learning rate
eta = 0.1

# Variance of the Gaussian likelihood
variance = 1.0

# Convergence parameters
max_iterations = 100
tolerance = 1e-6
previous_F = np.inf

# Predictive coding loop
for iteration in range(max_iterations):
  # Generate predictions
  predicted_o = sum(q_s[state] * P_o_given_s[state] for state in P_s)

  # Compute Free Energy
  prediction_error_term = sum(
    q_s[state] * np.sum((o - P_o_given_s[state])**2) / (2 * variance)
      for state in P_s
  )
  complexity_term = sum(
    q_s[state] * np.log(q_s[state] / P_s[state])
      for state in P_s
  )
  F = prediction_error_term + complexity_term

  # Gradient of F with respect to q(s)
  gradients = {}
  for state in q_s:
    likelihood = P_o_given_s[state]
    log_likelihood = -np.sum((o - likelihood)**2) / (2 * variance)
    gradients[state] = np.log(q_s[state] / P_s[state]) - log_likelihood + 1

  # Update q(s) using gradient descent
  for state in q_s:
    q_s[state] -= eta * gradients[state]

  # Project q(s) onto the probability simplex
  probs = np.array(list(q_s.values()))
  probs = np.maximum(probs, 0)  # Enforce non-negativity
  total = probs.sum()
  q_s = {state: prob / total for state, prob in zip(q_s.keys(), probs)}

  # Check for convergence
  if np.abs(F - previous_F) < tolerance:
    break
  previous_F = F

  print(f"Iteration {iteration + 1}:")
  print(f"  Predicted o: {predicted_o}")
  print(f"  Free Energy: {F}")
  print(f"  Updated Beliefs: {q_s}")
