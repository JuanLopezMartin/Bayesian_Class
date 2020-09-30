functions {
  matrix[] coronavirus_PPD_rng(int time_all, int num_country, int[] cases,
                             int[] tests, int[,] time_outbreak,
                             matrix ortho_time, vector suppress, vector count_outbreak,
                             real phi_scale, int[] country_pop) {
      matrix[num_country, time_all] infections;

      // Variable declaration
      // -- vars in the original code
      matrix[num_country,time_all] time_outbreak_trans1;
      matrix[num_country,time_all] time_outbreak_trans2;
      matrix[num_country,time_all] time_outbreak_trans3;
      matrix[num_country,time_all] count_tests;
      matrix[num_country,time_all] count_cases;
      matrix[num_country,time_all] num_infected_high;
      matrix[num_country,3] time_array[time_all]; 

      // -- parameters in the model with priors
      real beta_a = exponential_rng(.1);
      real sigma_q = exponential_rng(.1);
      real beta_S1 = normal_rng(0, 2);
      real beta_S2 = normal_rng(0, 2);
      real beta_I1 = normal_rng(0, 2);
      real beta_I2 = normal_rng(0, 2);
      real beta_I3 = normal_rng(0, 2);
      real alpha_1 = normal_rng(0, 0.1);
      real alpha_2 = normal_rng(0, 0.1);
      real alpha_3 = normal_rng(0, 0.1);
      vector[num_country] alpha_c;
      vector[num_country] beta_cq;
      
      // The ratio for which we assign a prior
      real ratio_requirement = fabs(normal_rng(0.15, 0.5));
      real ratio_data;
      int accept = 0;
      
      // -- other parameters in the model for which priors are not defined in the blog
      real beta_O1 = fabs(normal_rng(0, 1));
      real phi = exponential_rng(phi_scale);
      real phi_q = exponential_rng(phi_scale);
      real phi_a = exponential_rng(phi_scale);
      
      // Quantities we are going to calculate
      matrix[num_country, time_all] I_ct_mu;
      matrix[num_country, time_all] I_ct;
      matrix[num_country, time_all] q_ct_mu;
      int q_ct[num_country, time_all];
      matrix[num_country, time_all] q_ct_reals;
      matrix[num_country, time_all] a_ct_mu;
      int a_ct[num_country, time_all];
      matrix[num_country, time_all] a_ct_reals;
      matrix[num_country, time_all] output[3];
      
      // Data processing
      for(t in 1:time_all) {
        for(n in 1:num_country) {
          if(time_outbreak[n,t]>0) {
            time_outbreak_trans1[n,t] = ortho_time[time_outbreak[n,t],1];
            time_outbreak_trans2[n,t] = ortho_time[time_outbreak[n,t],2];
            time_outbreak_trans3[n,t] = ortho_time[time_outbreak[n,t],3];
          } else {
            time_outbreak_trans1[n,t] = 0;
            time_outbreak_trans2[n,t] = 0;
            time_outbreak_trans3[n,t] = 0;
          }
        }
      }
      
      for(t in 1:time_all) {
        time_array[t] = append_col(time_outbreak_trans1[,t], 
          append_col(time_outbreak_trans2[,t], 
          time_outbreak_trans3[,t]));
      }
      
      for(c in 1:num_country) {
        alpha_c[c] = normal_rng(0, 0.1);
        beta_cq[c] = exponential_rng(sigma_q);
      }
      
      
      // Model
      for(t in 2:time_all) {
        for(c in 1:num_country) {
          accept=0;
          while(accept==0){
            // We calculate everything based on the formulas
            I_ct_mu[c,t] = inv_logit(alpha_1 + alpha_c[c] + beta_O1 * count_outbreak[t] + (beta_S1*suppress[c]) + beta_I1*time_array[t][c,1] + beta_I2*time_array[t][c,2] + beta_I3*time_array[t][c,3] + ((beta_S2*suppress[c]) * time_outbreak_trans1[c,t]));
            I_ct[c,t] = beta_rng(I_ct_mu[c,t]*phi, (1-I_ct_mu[c,t])*phi);
            q_ct_mu[,t] = inv_logit(alpha_2 + beta_cq * I_ct[c,t]);
            q_ct[c,t] = beta_binomial_rng(country_pop[c], q_ct_mu[c,t]*phi_q, (1-q_ct_mu[c,t])*phi_q);
            a_ct_mu[,t] = inv_logit(alpha_3 + beta_a * I_ct[,t]);
            a_ct[c,t] = beta_binomial_rng(q_ct[c,t], a_ct_mu[c, t]*phi_a, (1-a_ct_mu[c,t])*phi_a);
            q_ct_reals[c,t] = q_ct[c,t];
            a_ct_reals[c,t] = a_ct[c,t];
            
            // the .1 is a dirty trick to convert the int to a real
            q_ct_reals[c,t] = (q_ct[c,t]-.1)/country_pop[c];
            a_ct_reals[c,t] = (a_ct[c,t]-.1)/country_pop[c];
            
            // The ratio of the data
            ratio_data = q_ct_reals[c,t]/I_ct[c,t];
            
            // To put the prior we do it in two steps: (1) verify that the ratio is between the valid range (0.1 to 0.3, with 
            // the 0.3 being three SD way from the expected value of 0.15) and (2) accept the value with probability p where
            // is |ratio_requirement - ratio_data|, with ratio_requirement being defined previously as ratio_requirement = 
            // fabs(normal_rng(0.15, 0.5))
            if(ratio_data>0.1 && ratio_data<0.3)
              accept = bernoulli_rng(1-fabs(ratio_requirement-ratio_data));
            }
          }
          //print("Done")
        }
      
      output[1] = I_ct;
      output[2] = q_ct_reals;
      output[3] = a_ct_reals;
      return output;
  }
}

