/**
 * The 3-body dynamic causal model class within the dcEmb package
 *
 * Copyright (C) 2022 Embecosm Limited
 *
 * Contributor William Jones <william.jones@embecosm.com>
 * Contributor Elliot Stein <E.Stein@soton.ac.uk>
 *
 * This file is part of the dcEmb package
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "DEM_weather.hh"
#include <stdio.h>
#include <iostream>
#include <list>
#include <random>
#include <vector>
#include "Eigen/Dense"
#include "country_data.hh"
#include "dcEmb/bmr_model.hh"
#include "dynamic_weather_model.hh"
#include "species_struct.hh"
#include "utility.hh"
#define DEBUG(x) std::cout << #x << "= " << '\n' << x << std::endl;
/**
 * Run the weather example
 */
int run_weather_test() {
  dynamic_weather_model model;

  // Set the dates between which we want to pull weather data
  int start_date = 1750;
  int end_date = 2100;
  int sz = end_date - start_date + 1;
  model.num_samples = sz;

  // This particular example has many free variables to fit, and will require
  // more than the default 128 iterations
  model.max_invert_it = 16;

  // Turn on outputting intermediate calculations from dcEmb, and set filenames.
  // We'll use these to make graphs later.
  model.intermediate_outputs_to_file = 1;
  // model.intermediate_expectations_filename =
  //     "param_expectations.csv";
  // model.intermediate_covariances_filename =
  //     "param_covariances.csv";

  // Define which species of gasses will drive our model, and populate the
  // properties of these gasses with default values from a config file.
  std::vector<std::string> species_names(
      {"CO2 FFI", "CO2 AFOLU", "CO2", "CH4", "N2O"});
  model.species_list = simple_species_struct(species_names);

  // Define the Emissions, Concentrations and Forcings for the ground truth,
  // based on the ssp585 scenario, a high emissions scenario
  std::vector<Eigen::MatrixXd> ecf =
      simple_ecf(model.species_list, "ssp585", start_date, end_date);

  // Set model emissions
  model.emissions = ecf.at(0);
  // Convert model emissions to standard units
  model.emissions.row(0) = model.emissions.row(0) / 1000;
  model.emissions.row(1) = model.emissions.row(1) / 1000;
  model.emissions.row(4) = model.emissions.row(4) / 1000;

  // CO2 emissions are the sum of CO2 FFI and CO2 AFOLU emissions
  model.emissions.row(2) = model.emissions.row(0) + model.emissions.row(1);

  // Set model concentrations & tweak initial concentrations
  model.concentrations = ecf.at(1);
  model.concentrations(Eigen::all, 0) =
      model.species_list.baseline_concentration;
  model.concentrations(0, 0) = 0;
  model.concentrations(1, 0) = 0;

  // Set model forcings
  model.forcings = Eigen::MatrixXd::Zero(model.species_list.name.size(), sz);

  // Specify which outputs of the climate model we will fit our DCM with. 5, 6,
  // 7, 8, 9 correspond to concentrations of each of the our gas species
  model.num_response_vars = 5;
  model.select_response_vars = (Eigen::VectorXi(5) << 5, 6, 7, 8, 9).finished();

  // Define prior expectations for parameters. In this example, we set the
  // parameters as a mix of global parameters, and CO2 FFI emissions between
  // 2000 and 2100
  std::vector<Eigen::MatrixXd> ecf_prior =
      simple_ecf(model.species_list, "ssp126", start_date, end_date);
  model.prior_parameter_expectations = default_prior_expectations(
      ecf_prior.at(0)(0, Eigen::seq(250, Eigen::last)));

  // Set other parameter and hyperparameter expectations/covariances to
  // default values
  model.prior_parameter_covariances = default_prior_covariances(sz - 250);
  model.prior_hyper_expectations = default_hyper_expectations();
  model.prior_hyper_covariances = default_hyper_covariances();
  model.parameter_locations = default_parameter_locations();

  // Initialise temperatures, airborne emissions and cumulative emissions to 0
  model.temperature = Eigen::MatrixXd::Zero(3, sz);
  model.airborne_emissions =
      Eigen::MatrixXd::Zero(model.species_list.name.size(), sz);
  model.cumulative_emissions =
      Eigen::MatrixXd::Zero(model.species_list.name.size(), sz);

  // Create ground truth climate model from known values for all parameters
  Eigen::MatrixXd true_out = model.eval_generative(
      true_prior_expectations(model.emissions(0, Eigen::seq(250, Eigen::last))),
      default_parameter_locations(), sz);
  model.response_vars = true_out(Eigen::all, model.select_response_vars);

  // Invert the model to create posterior estimates of model parameters, based
  // on the priors we set previously and on the ground truth data we selected
  // previously (the concentrations of our gas species)
  model.invert_model();

  // Set the number of samples of our posteriors to record
  int n = 10;

  // Record the output of our climate model based on our prior expectations
  Eigen::MatrixXd prior_e_out = model.eval_generative(
      model.prior_parameter_expectations, default_parameter_locations(), sz);

  // Record the output of our climate model based on our posterior expectations
  Eigen::MatrixXd posterior_final_out =
      model.eval_generative(model.conditional_parameter_expectations,
                            default_parameter_locations(), sz);

  // Initialize random number generation
  std::default_random_engine rd;
  std::mt19937 gen(rd());

  // Create and populate an array of sample model outputs based on random
  // samples from our joint prior probability distribution across parameters
  Eigen::MatrixXd prior_rand_out =
      Eigen::MatrixXd(prior_e_out.rows() * n, prior_e_out.cols());
  for (int i = 0; i < n; i++) {
    Eigen::MatrixXd prior_tmp =
        random_generative(model, model.prior_parameter_expectations,
                          model.prior_parameter_covariances, sz, gen);
    prior_rand_out(Eigen::seqN(i * sz, sz), Eigen::all) = prior_tmp;
  }

  // Open files recording intermediate parameter expectaions and covariances
  // from each iteration of the DCM model inversion
  std::ifstream param_expectations_file;
  param_expectations_file.open("param_expecations.csv");
  std::string param_expectations_line;
  std::ifstream param_covariances_file;
  param_covariances_file.open("param_covariances.csv");
  std::string param_covariances_line;

  // Record the output of our climate model based on our posterior expectations,
  // for each iteration of DCM model inversion
  Eigen::MatrixXd posterior_e_out =
      Eigen::MatrixXd(sz * model.performed_it, posterior_final_out.cols());

  // Initialize arrays to record samples of model outputs based on random
  // samples from the joint posterior probability distribution across parameters
  // for iteration of the DCM model inversion
  // Warning, if n is set high, this will be BIG. For ~350 timesteps,
  // n=100 = ~4.5gb, n=1000 ~45gb.
  Eigen::MatrixXd posterior_rand_out =
      Eigen::MatrixXd(sz * n * model.performed_it, posterior_final_out.cols());

  // Fill sample of model outputs from posterior samples array
  int i = 0;
  while (std::getline(param_expectations_file, param_expectations_line)) {
    std::vector<double> values_e;
    std::stringstream lineStream_e(param_expectations_line);
    std::string cell_e;
    while (std::getline(lineStream_e, cell_e, ',')) {
      values_e.push_back(std::stod(cell_e));
    }
    std::vector<double> values_c;
    for (int k = 0; k < values_e.size(); k++) {
      std::getline(param_covariances_file, param_covariances_line);

      std::stringstream lineStream_c(param_covariances_line);
      std::string cell_c;
      while (std::getline(lineStream_c, cell_c, ',')) {
        values_c.push_back(std::stod(cell_c));
      }
    }
    Eigen::VectorXd param_expectations =
        Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(values_e.data(),
                                                      values_e.size());
    Eigen::MatrixXd param_covariances =
        Eigen::Map<Eigen::MatrixXd, Eigen::Unaligned>(
            values_c.data(), values_e.size(), values_e.size());
    posterior_e_out(Eigen::seqN(i * sz, sz), Eigen::all) =
        model.eval_generative(param_expectations, default_parameter_locations(),
                              sz);
    for (int j = 0; j < n; j++) {
      posterior_rand_out(Eigen::seqN(i * sz * n + (j * sz), sz), Eigen::all) =
          random_generative(model, param_expectations, param_covariances, sz,
                            gen);
    }
    i++;
  }

  // close file handles
  param_expectations_file.close();
  param_covariances_file.close();

  // Output recorded values to folder containing visualization python scripts
  utility::print_matrix("../visualisation/weather/true_generative.csv",
                        true_out);
  utility::print_matrix("../visualisation/weather/prior_generative.csv",
                        prior_e_out);
  utility::print_matrix("../visualisation/weather/prior_generative_rand.csv",
                        prior_rand_out);
  utility::print_matrix("../visualisation/weather/pos_generative.csv",
                        posterior_e_out);
  utility::print_matrix("../visualisation/weather/pos_generative_rand.csv",
                        posterior_rand_out);
  return 0;
}

/**
 * Run the "forward" generative model of the given DCM, with parameters sampled
 * from the given mean and covariance matrix.
 *
 * The method uses the scheme based on the Cholesky decomposition described
 * here:
 * https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values
 * _from_the_distribution
 */
Eigen::MatrixXd random_generative(dynamic_weather_model& model,
                                  Eigen::VectorXd& mean, Eigen::MatrixXd& var,
                                  int& sz, std::mt19937& gen) {
  std::normal_distribution<double> dis(0, 1);

  // Independent samples from the N(0,1) distribution
  Eigen::VectorXd rand_param_prior =
      Eigen::VectorXd::Zero(mean.size()).unaryExpr([&](double dummy) {
        return dis(gen);
      });

  // Cholesky Decomposition
  Eigen::LLT<Eigen::MatrixXd> lltOfA(var);
  Eigen::MatrixXd L = lltOfA.matrixL();

  // x = L * Normal(0^N,1^N)^T + mu
  rand_param_prior = ((L * rand_param_prior).array() + mean.array()).eval();

  // Run model
  return model.eval_generative(rand_param_prior, default_parameter_locations(),
                               sz);
}

/**
 * Populate emissions (e), concentrations(c), and forcings(f) matricies with
 * values from the given scenario between the two dates from rcmip data files
 */
std::vector<Eigen::MatrixXd> simple_ecf(const species_struct& species,
                                        const std::string& scenario,
                                        const int& start_date,
                                        const int& end_date) {
  // Initialize e, c and f matricies
  Eigen::MatrixXd e_matrix =
      Eigen::MatrixXd::Zero(species.name.size(), end_date - start_date + 1);
  Eigen::MatrixXd c_matrix =
      Eigen::MatrixXd::Zero(species.name.size(), end_date - start_date + 1);
  Eigen::MatrixXd f_matrix =
      Eigen::MatrixXd::Zero(species.name.size(), end_date - start_date + 1);
  // e, c and f filenames
  std::string emissions_filename =
      "../src/weather/data/rcmip-emissions-annual-means-v5-1-0.csv";
  std::string concentrations_filename =
      "../src/weather/data/rcmip-concentrations-annual-means-v5-1-0.csv";
  std::string forcings_filename =
      "../src/weather/data/rcmip-radiative-forcing-annual-means-v5-1-0.csv";

  // Get the names of the species in the species struct, fix some names to be
  // consistent with ones in the rcmip file
  std::vector<std::string> species_names_rcmip = species.name;

  std::replace(species_names_rcmip.begin(), species_names_rcmip.end(),
               (std::string) "CO2 FFI",
               (std::string) "CO2|MAGICC Fossil and Industrial");
  std::replace(species_names_rcmip.begin(), species_names_rcmip.end(),
               (std::string) "CO2 AFOLU", (std::string) "CO2|MAGICC AFOLU");

  // Iterate through each line of the rcmip emissions file, finding the "World"
  // emissions for the given scenario that match any of our specified species
  std::string emissions_line;
  std::ifstream emissions_file;
  emissions_file.open(emissions_filename);
  while (std::getline(emissions_file, emissions_line)) {
    std::vector<std::string> emissions_split;
    std::vector<std::string> variable_split;
    utility::splitstr(emissions_split, emissions_line, ',');
    utility::splitstr(variable_split, emissions_split.at(3), '|');
    if (emissions_split.at(2) != "World") {
      continue;
    }
    if (emissions_split.at(1) != scenario) {
      continue;
    }
    int name_idx = std::find(species_names_rcmip.begin(),
                             species_names_rcmip.end(), variable_split.back()) -
                   species_names_rcmip.begin();
    if (name_idx == species_names_rcmip.size()) {
      if (variable_split.size() == 1) {
        continue;
      }
      name_idx =
          std::find(species_names_rcmip.begin(), species_names_rcmip.end(),
                    (variable_split.at(variable_split.size() - 2) + "|" +
                     variable_split.back())) -
          species_names_rcmip.begin();
      if (name_idx == species_names_rcmip.size()) {
        continue;
      }
    }
    // Fill blank entries in the rcmip file with interpolated values
    // There's probably a way to do this elegantly with Eigen::Map to reuse
    // the memory
    for (int i = 0; i < emissions_split.size(); i++) {
      if (emissions_split.at(i).empty()) {
        for (int j = i; j < emissions_split.size(); j++) {
          if (!emissions_split.at(j).empty()) {
            Eigen::VectorXd vec = Eigen::VectorXd::LinSpaced(
                j - i + 2, std::stod(emissions_split.at(i - 1)),
                std::stod(emissions_split.at(j)));
            for (int k = 0; k < vec.size() - 2; k++) {
              emissions_split.at(i + k) = std::to_string(vec(k + 1));
            }
            break;
          }
        }
      }
    }

    if (species.input_mode.at(name_idx) == "emissions") {
      for (int i = 0; i < (end_date - start_date + 1); i++) {
        int pos = start_date - 1750 + 7 + i;
        if (pos < emissions_split.size()) {
          std::string s = emissions_split.at(pos);
          e_matrix(name_idx, i) =
              (s == "") ? 0 : std::stod(emissions_split.at(pos));
        }
      }
    }
  }

  // Iterate through each line of the rcmip concentrations file, finding the
  // "World" concentrations for the given scenario that match any of our
  // specified species
  std::string concentrations_line;
  std::ifstream concentrations_file;
  concentrations_file.open(concentrations_filename);
  while (std::getline(concentrations_file, concentrations_line)) {
    std::vector<std::string> concentrations_split;
    std::vector<std::string> variable_split;
    utility::splitstr(concentrations_split, concentrations_line, ',');
    utility::splitstr(variable_split, concentrations_split.at(3), '|');
    if (concentrations_split.at(2) != "World") {
      continue;
    }
    if (concentrations_split.at(1) != scenario) {
      continue;
    }
    int name_idx = std::find(species_names_rcmip.begin(),
                             species_names_rcmip.end(), variable_split.back()) -
                   species_names_rcmip.begin();
    if (name_idx == species_names_rcmip.size()) {
      if (variable_split.size() == 1) {
        continue;
      }
      name_idx =
          std::find(species_names_rcmip.begin(), species_names_rcmip.end(),
                    (variable_split.at(variable_split.size() - 2) + "|" +
                     variable_split.back())) -
          species_names_rcmip.begin();
      if (name_idx == species_names_rcmip.size()) {
        continue;
      }
    }
    // Fill blank entries in the rcmip file with interpolated values
    // There's probably a way to do this elegantly with Eigen::Map to reuse
    // the memory
    if (species.input_mode.at(name_idx) == "concentrations") {
      for (int i = 0; i < (end_date - start_date + 1); i++) {
        int pos = start_date - 1700 + 7 + i;
        if (pos < concentrations_split.size()) {
          std::string s = concentrations_split.at(pos);
          c_matrix(name_idx, i) =
              (s == "") ? 0 : std::stod(concentrations_split.at(pos));
        }
      }
    }
  }

  // Iterate through each line of the rcmip forcings file, finding the
  // "World" forcings for the given scenario that match any of our
  // specified species
  std::string forcings_line;
  std::ifstream forcings_file;
  forcings_file.open(forcings_filename);
  while (std::getline(forcings_file, forcings_line)) {
    std::vector<std::string> forcings_split;
    std::vector<std::string> variable_split;
    utility::splitstr(forcings_split, forcings_line, ',');
    utility::splitstr(variable_split, forcings_split.at(3), '|');
    if (forcings_split.at(2) != "World") {
      continue;
    }
    if (forcings_split.at(1) != scenario) {
      continue;
    }
    int name_idx = std::find(species_names_rcmip.begin(),
                             species_names_rcmip.end(), variable_split.back()) -
                   species_names_rcmip.begin();
    if (name_idx == species_names_rcmip.size()) {
      if (variable_split.size() == 1) {
        continue;
      }
      name_idx =
          std::find(species_names_rcmip.begin(), species_names_rcmip.end(),
                    (variable_split.at(variable_split.size() - 2) + "|" +
                     variable_split.back())) -
          species_names_rcmip.begin();
      if (name_idx == species_names_rcmip.size()) {
        continue;
      }
    }
    // Fill blank entries in the rcmip file with interpolated values
    // There's probably a way to do this elegantly with Eigen::Map to reuse
    // the memory
    if (species.input_mode.at(name_idx) == "forcings") {
      for (int i = 0; i < (end_date - start_date + 1); i++) {
        int pos = start_date - 1750 + 7 + i;
        if (pos < forcings_split.size()) {
          std::string s = forcings_split.at(pos);
          f_matrix(name_idx, i) =
              (s == "") ? 0 : std::stod(forcings_split.at(pos));
        }
      }
    }
  }
  std::vector<Eigen::MatrixXd> out;
  out.push_back(e_matrix);
  out.push_back(c_matrix);
  out.push_back(f_matrix);
  return out;
}

/**
 * Return a species struct from a list of gas species names,
 */
species_struct simple_species_struct(
    const std::vector<std::string>& species_names) {
  std::string filename = "../src/weather/data/species_configs_properties.csv";
  species_struct species = utility::species_from_file(filename, species_names);

  // Change CO2 to calculated
  species.input_mode.at(2) = "calculated";

  // Adjust unpeturbed lifetimes for CH4
  species.unperturbed_lifetime(Eigen::all, 3) =
      Eigen::VectorXd::Ones(species.unperturbed_lifetime.rows()) * 10.8537568;
  // Adjust baseline emissions for CH4 and NO2
  species.baseline_emissions(3) = 19.01978312;
  species.baseline_emissions(4) = 0.08602230754;

  // Adjust aerosol-cloud interactions. ACI shape. 0, except for CO2
  species.aci_shape = Eigen::VectorXd::Zero(6);
  species.aci_shape(2) = 1 / 260.34644166;

  // Cascade any downstream changes from the updates values
  utility::update_species_list_indicies(species);
  return species;
}

/**
 * Some sensible "true" values for model parameters, plus CO2 FFI true values
 * as additional parameters
 */
Eigen::VectorXd true_prior_expectations(Eigen::VectorXd em) {
  Eigen::VectorXd default_prior_expectation = Eigen::VectorXd::Zero(11);
  default_prior_expectation << 1.876, 5.154, 0.6435, 2.632, 9.262, 52.93, 1.285,
      2.691, 0.4395, 28.24, 8;

  Eigen::VectorXd out_vec = Eigen::VectorXd(em.size() + 11);
  out_vec << default_prior_expectation, em;
  return out_vec;
}

/**
 * A parameter locations object, to allow a mapping between parameters and
 * locations in the parameter array
 */
parameter_location_weather default_parameter_locations() {
  parameter_location_weather parameter_locations;
  parameter_locations.ocean_heat_transfer = Eigen::VectorXi(3);
  parameter_locations.ocean_heat_transfer << 0, 1, 2;
  parameter_locations.ocean_heat_capacity = Eigen::VectorXi(3);
  parameter_locations.ocean_heat_capacity << 3, 4, 5;
  parameter_locations.deep_ocean_efficacy = Eigen::VectorXi(1);
  parameter_locations.deep_ocean_efficacy << 6;
  parameter_locations.sigma_eta = Eigen::VectorXi(1);
  parameter_locations.sigma_eta << 7;
  parameter_locations.sigma_xi = Eigen::VectorXi(1);
  parameter_locations.sigma_xi << 8;
  parameter_locations.gamma_autocorrelation = Eigen::VectorXi(1);
  parameter_locations.gamma_autocorrelation << 9;
  parameter_locations.forcing_4co2 = Eigen::VectorXi(1);
  parameter_locations.forcing_4co2 << 10;

  return parameter_locations;
}

/**
 * A set of prior expectations from the parameters that deviate from the true
 * values, plus input CO2 FFI values as additional parameters
 */
Eigen::VectorXd default_prior_expectations(Eigen::VectorXd em) {
  Eigen::VectorXd default_prior_expectation = Eigen::VectorXd::Zero(11);
  double x = 1;
  default_prior_expectation << 2, 5, 1, 3, 9, 50, 1, 3, 0, 30, 5;

  Eigen::VectorXd out_vec = Eigen::VectorXd(em.size() + 11);
  out_vec << default_prior_expectation, em * x;
  return out_vec;
}

/**
 * Prior covariance matrix
 */
Eigen::MatrixXd default_prior_covariances(int sz) {
  double flat = 1.0;                    // flat priors
  double informative = 1 / (double)16;  // informative priors
  double precise = 1 / (double)256;     // precise priors
  double fixed = 1 / (double)2048;      // fixed priors
  Eigen::VectorXd default_prior_covariance = Eigen::VectorXd::Ones(sz + 11);
  default_prior_covariance = default_prior_covariance * informative;

  Eigen::MatrixXd return_default_prior_covariance =
      Eigen::MatrixXd::Zero(sz + 11, sz + 11);
  return_default_prior_covariance.diagonal() = default_prior_covariance;
  return return_default_prior_covariance;
}

/**
 * Prior hyperparameter expectation vector
 */
Eigen::VectorXd default_hyper_expectations() {
  Eigen::VectorXd default_hyper_expectation = Eigen::VectorXd::Zero(5);
  return default_hyper_expectation;
}
/**
 * Prior hyperparameter covariance matrix
 */
Eigen::MatrixXd default_hyper_covariances() {
  double flat = 1.0;                    // flat priors
  double informative = 1 / (double)16;  // informative priors
  double precise = 1 / (double)256;     // precise priors
  double fixed = 1 / (double)2048;      // fixed priors
  Eigen::MatrixXd default_hyper_covariance = Eigen::MatrixXd::Zero(5, 5);
  default_hyper_covariance.diagonal() << precise, precise, precise, precise,
      precise;
  return default_hyper_covariance;
}
