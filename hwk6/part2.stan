
data {
  int<lower=0> N;
  vector[N] y_n;
  vector[N] k_n;
  vector[N] e_n;
  vector[N] a_n;
}

parameters {
  // * gamma, rho, rho_1, and nu cannot be negative. Therefore, p_gamma, p_rho, p_rho_1, and p_nu must
  // be more than 0.5. This restriction does not have to be imposed, but if we don't include it many
  // of the draws will be rejected - reducing ESS and thus requiring more computing time. 
  // * delta and delta_1 must be between 0 and 1. From the way they are calculated in the transformed
  // parameters block, this implies 0.5+(p_delta)*0.5 > 0 -> p_delta > 0.16 approx. and 
  // 0.5+(p_delta)*0.5 < 1 -> p_delta<0.84 approx. The same reasoning applies to p_delta_1/
  // * rho and rho_1 must be higher than 0.1, and therefore p_rho and p_rho1 must be higher than
  // inv_Phi(p_rho)*10>-1 -> p_rho > 0.46 approx.
  real<lower=0.5, upper=1> p_gamma;
  real<lower=0.15, upper=0.85> p_delta;
  real<lower=0.15, upper=0.85> p_delta_1;
  real<lower=0.45, upper=1> p_rho;
  real<lower=0.45, upper=1> p_rho_1;
  real<lower=0.5, upper=1> p_nu;
  
  //real<lower=0> gamma;
  //real<lower=0, upper=1> delta;
  //real<lower=0, upper=1> delta_1;
  //real<lower=-1> rho;
  //real<lower=-1> rho_1;
  //real<lower=0> nu;
  //real<lower=0> sigma;
  
}

transformed parameters{
  real<lower=0> gamma = inv_Phi(p_gamma)*10;
  real<lower=0, upper=1> delta = 0.5 + inv_Phi(p_delta)*0.5;
  real<lower=0, upper=1> delta_1 = 0.5 + inv_Phi(p_delta_1)*0.5;
  real<lower=-1> rho = inv_Phi(p_rho)*10;
  real<lower=-1> rho_1 = inv_Phi(p_rho_1)*10;
  real<lower=0> nu = inv_Phi(p_nu)*10;
}

model {
  vector[N] z;
  
  // Except for the deltas that have clearly defined bounds, it seems like all the other parameters
  // can take a wide range of values. Consequently, I decided for wide priors. However, I am sure 
  // a better understanding of the model could lead to better priors.
  // The priors below and the transformer parameters calculated in the transformed parameters block are 
  // equivalent to the priors shown below in a more intuitive way
  //gamma ~ normal(0, 10);
  //delta ~ normal(0.5, 0.5);
  //delta_1 ~ normal(0.5, 0.5);
  //rho ~ normal(0, 10);
  //rho_1 ~ normal(0, 10);
  //nu ~ normal(0, 10);
  sigma ~ normal(0, 10);
  
  // Draw from uniform and then from cumulative normal
  p_gamma ~ uniform(0, 1);
  p_delta ~ uniform(0, 1);
  p_delta_1 ~ uniform(0, 1);
  p_rho ~ uniform(0, 1);
  p_rho_1 ~ uniform(0, 1);
  p_nu ~ uniform(0, 1);
  
  // In a loop because I haven't found a way to do element-wise exponentiation in Stan
  for(n in 1:N){
    z[n] = gamma * (delta*((delta_1*k_n[n]^(-rho_1)+(1-delta_1)*e_n[n]^(-rho_1))^(rho/rho_1)) + (1-delta)*a_n[n]^(-rho))^(-nu/rho);
  }
  
  y_n ~ normal(z, sigma);
}
generated quantities {
  real b_1;
  real theta_star;
  real theta;
  real theta_1;
  real theta_2;
  real theta_3;
  real au_12;
  real au_13;
  real au_23;
  real hm_12;
  real hm_13;
  real hm_23;


  b_1 = delta_1*mean(k_n)^(-rho_1) + (1-delta_1)*mean(e_n)^(-rho_1);
  au_12 = (1-rho)^(-1) + ((1-rho_1)^(-1)-(1-rho)^(-1))/(delta*(mean(y_n)/b_1^(-1/rho_1))^(1+rho));
  au_13 = (1-rho)^-1;
  au_23 = (1-rho)^-1;
  
  theta_star = delta*b_1^(rho/rho_1)*mean(y_n)^rho;
  theta = (1-delta)*mean(a_n)^(-rho)*mean(y_n)^rho;
  theta_1 = delta*delta_1*mean(k_n)^(-rho_1)*b_1^(-(rho_1-rho)/rho_1)*mean(y_n)^(rho);
  theta_2 = delta*(1-delta_1)*mean(e_n)^(-rho_1)*b_1^(-(rho_1-rho)/rho_1)*mean(y_n)^(rho);
  theta_3 = theta;
  hm_12 = (1-rho_1)^-1;
  hm_13 = ((1/theta_1)+(1/theta_3))/((1-rho_1)*((1/theta_1)-(1/theta_star)) + (1-rho)*((1/theta_3)-(1/theta)) + (1-rho)*((1/theta_star)-(1/theta)));
  hm_23 = ((1/theta_2)+(1/theta_3))/((1-rho_1)*((1/theta_2)-(1/theta_star)) + (1-rho)*((1/theta_3)-(1/theta)) + (1-rho)*((1/theta_star)-(1/theta)));
}


