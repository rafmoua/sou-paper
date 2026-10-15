data {
  int<lower=1> n;                 // Number of groups (i = 1, ..., n)
  int<lower=1> p;                 // Number of replicates per group
  matrix[n, p] Y;                 // Observations: Y[i, j] is replicate j in group i

  real<lower=0> tau_scale;        // Global shrinkage scale (half-Cauchy)
  real<lower=0> lambda_scale;     // Local shrinkage scale (half-Cauchy)
  real<lower=0> sigma_alpha;      // Inverse-Gamma shape for sigma^2
  real<lower=0> sigma_beta;       // Inverse-Gamma scale for sigma^2
}

parameters {
  vector[n] beta;                 // Group-specific means
  real<lower=0> tau;              // Global shrinkage parameter
  vector<lower=0>[n] lambda;      // Local shrinkage parameters
  real<lower=0> sigma_sq;         // Noise variance (sigma^2)
}

transformed parameters {
  real<lower=0> sigma = sqrt(sigma_sq);  // Noise standard deviation
}

model {
  // Priors
  tau ~ cauchy(0, tau_scale);                 // half-Cauchy due to lower bound
  lambda ~ cauchy(0, lambda_scale);           // half-Cauchy due to lower bound
  sigma_sq ~ inv_gamma(sigma_alpha, sigma_beta);

  // Horseshoe prior (centered)
  beta ~ normal(0, tau * lambda);

  // Likelihood: p replicates per group
  for (i in 1:n) {
    Y[i] ~ normal(beta[i], sigma);
  }
}

generated quantities {
  matrix[n, p] Y_rep;             // Posterior predictive replicated data
  vector[n] kappa;                // Shrinkage weights (one per group)

  for (i in 1:n) {
    kappa[i] = 1 / (1 + (square(lambda[i]) * square(tau)) / sigma_sq);
    for (j in 1:p) {
      Y_rep[i, j] = normal_rng(beta[i], sigma);
    }
  }
}
