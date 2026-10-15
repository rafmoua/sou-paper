data {
  int<lower=1> n;                 // Number of observations
  int<lower=1> p;                 // Number of predictors
  matrix[n, p] X;                 // Design matrix
  vector[n] y;                    // Response vector

  real<lower=0> tau_scale;        // Global shrinkage scale (half-Cauchy)
  real<lower=0> lambda_scale;     // Local shrinkage scale (half-Cauchy)
  real<lower=0> sigma_alpha;      // Inverse-Gamma shape for sigma^2
  real<lower=0> sigma_beta;       // Inverse-Gamma scale for sigma^2
}

parameters {
  real<lower=0> tau;              // Global shrinkage parameter
  vector<lower=0>[p] lambda;      // Local shrinkage parameters
  vector[p] beta_raw;             // Standardized coefficients (non-centered)
  real<lower=0> sigma_sq;         // Noise variance (sigma^2)
}

transformed parameters {
  vector[p] beta;                 // Regression coefficients
  real<lower=0> sigma;            // Noise standard deviation

  beta = beta_raw .* lambda * tau;
  sigma = sqrt(sigma_sq);
}

model {
  // Priors
  tau ~ cauchy(0, tau_scale);                 // half-Cauchy due to lower bound
  lambda ~ cauchy(0, lambda_scale);           // half-Cauchy due to lower bound
  beta_raw ~ normal(0, 1);                    // non-centered parameterization
  sigma_sq ~ inv_gamma(sigma_alpha, sigma_beta);

  // Likelihood
  y ~ normal(X * beta, sigma);
}

generated quantities {
  vector[n] y_rep;                // Replicated data (posterior predictive)

  for (i in 1:n) {
    y_rep[i] = normal_rng(dot_product(X[i], beta), sigma);
  }
}
