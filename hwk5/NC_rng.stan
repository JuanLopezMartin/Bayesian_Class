//
// Stan file for assignment 5 (Juan Lopez Martin; jl5522)

functions {
  int[ , , ] NC_rng(int D, int R, int[] Asian, int[] Black, int[] Hispanic, int[] White) {
    //matrix[D, R] data_matrix =  append_col(White, Black)
    int draws[D, R, 2] = rep_array(0, D, R, 2); // initialize with zeros
    
    vector[R] phi_r;
    vector[R] lambda_r;
    vector[D] phi_d;
    vector[D] lambda_d;
    
    real mu_phi_r = normal_rng(0, 2);
    real sigma_phi_r = fabs(normal_rng(0, 2));
    
    real mu_phi_d = normal_rng(0, 2);
    real sigma_phi_d = fabs(normal_rng(0, 2));
    
    real mu_lambda_r = normal_rng(0, 2);
    real sigma_lambda_r = fabs(normal_rng(0, 2));
    
    real mu_lambda_d = normal_rng(0, 2);
    real sigma_lambda_d = fabs(normal_rng(0, 2));
    
    real mu_t_rd = normal_rng(0, 2);
    real sigma_t_rd = fabs(normal_rng(0, 2));
    
    for (r in 1:R){
      phi_r[r] = normal_rng(mu_phi_r, sigma_phi_r);
      lambda_r[r] = normal_rng(mu_lambda_r, sigma_lambda_r);
    }

    for (d in 1:D){
      phi_d[d] = normal_rng(mu_phi_d, sigma_phi_d);
      lambda_d[d] = normal_rng(mu_lambda_d, sigma_lambda_d);
    }
    
    phi_d[1] = 0;
    lambda_d[1] = 0;
    
    for (r in 1:R){
      for (d in 1:D){
        real phi_rd = inv_logit(phi_r[r] + phi_d[d]);
        real lambda_rd = exp(phi_r[r] + phi_d[d]);
        real t_rd = inv_logit(normal_rng(mu_t_rd, sigma_t_rd));
        
        real p_rd = 1 - beta_cdf(t_rd, phi_rd*lambda_rd, (1-phi_rd*lambda_rd));
        real q_rd;
        
        int n_rd;
        if (r==1)
            n_rd = Asian[d];
        else if (r==2)
            n_rd = Black[d];
        else if (r==3)
            n_rd = Hispanic[d];
        else if (r==4)
            n_rd = White[d];

        draws[d, r, 1] = binomial_rng(n_rd, p_rd);
        
        q_rd = phi_rd * (1 - beta_cdf(t_rd, phi_rd*(lambda_rd+1), (1-phi_rd*(lambda_rd+1)))) / beta_cdf(t_rd, phi_rd*lambda_rd, (1-phi_rd*lambda_rd));
        draws[d, r, 2] = binomial_rng(draws[d, r, 1], q_rd);
        
      }
    }
    
    return draws;
  }
}
