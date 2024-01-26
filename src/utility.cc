/**
 * A set of utility functions for the dcEmb package
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

#include "utility.hh"
#include <bitset>
#include <functional>
#include <iostream>
#include <sstream>
#include <unsupported/Eigen/MatrixFunctions>
#include "Eigen/Core"

/**
 * Split a string around a token
 */
void utility::splitstr(std::vector<std::string>& vec, const std::string& str,
                       const char& delim) {
  std::stringstream ss(str);
  std::string line;
  while (std::getline(ss, line, delim)) {
    vec.push_back(line);
  }
  return;
}

species utility::species_from_string(const std::string& string) {
  species species;
  std::vector<std::string> properties;
  utility::splitstr(properties, string, ',');

  species.name = properties[0];
  species.type = properties[1];
  species.input_mode = properties[2];
  species.greenhouse_gas = stoi(properties[3]);
  species.aerosol_chemistry_from_emissions = stoi(properties[4]);
  species.aerosol_chemistry_from_concentration = stoi(properties[5]);
  species.partition_fraction << stod(properties[6]), stod(properties[7]),
      stod(properties[8]), stod(properties[9]);
  species.unperturbed_lifetime << stod(properties[10]), stod(properties[11]),
      stod(properties[12]), stod(properties[13]);
  species.tropospheric_adjustment = stod(properties[14]);
  species.forcing_efficacy = stod(properties[15]);
  species.forcing_temperature_feedback = stod(properties[16]);
  species.forcing_scale = stod(properties[17]);
  species.molecular_weight = stod(properties[18]);
  species.baseline_concentration = stod(properties[19]);
  species.forcing_reference_concentration = stod(properties[20]);
  species.forcing_reference_emissions = stod(properties[21]);
  species.iirf_0 = stod(properties[22]);
  species.iirf_airborne = stod(properties[23]);
  species.iirf_uptake = stod(properties[24]);
  species.iirf_temperature = stod(properties[25]);
  species.baseline_emissions = stod(properties[26]);
  species.g1 = stod(properties[27]);
  species.g0 = stod(properties[28]);
  species.greenhouse_gas_radiative_efficiency = stod(properties[29]);
  species.contrails_radiative_efficiency = stod(properties[30]);
  species.erfari_radiative_efficiency = stod(properties[31]);
  species.h2o_stratospheric_factor = stod(properties[32]);
  species.lapsi_radiative_efficiency = stod(properties[33]);
  species.land_use_cumulative_emissions_to_forcing = stod(properties[34]);
  species.ozone_radiative_efficiency = stod(properties[35]);
  species.cl_atoms = stod(properties[36]);
  species.br_atoms = stod(properties[37]);
  species.fractional_release = stod(properties[38]);
  species.aci_shape = stod(properties[39]);
  species.aci_scale = stod(properties[40]);
  species.ch4_lifetime_chemical_sensitivity = stod(properties[41]);
  species.lifetime_temperature_sensitivity = stod(properties[42]);
  return species;
}

species_struct utility::species_from_file(
    const std::string& filename, const std::vector<std::string>& names) {
  std::vector<species> species_list(names.size());
  std::string species_line;
  std::ifstream species_file;
  species_file.open(filename);
  while (std::getline(species_file, species_line)) {
    std::vector<std::string> species_split;
    utility::splitstr(species_split, species_line, ',');
    int idx =
        std::find(names.begin(), names.end(), species_split[0]) - names.begin();
    if (idx != names.size()) {
      species_list.at(idx) = utility::species_from_string(species_line);
    }
  }
  return utility::species_list_to_struct(species_list);
}

species_struct utility::species_from_file(const std::string& filename) {
  std::vector<species> species_list;
  std::string species_line;
  std::ifstream species_file;
  species_file.open(filename);
  while (std::getline(species_file, species_line)) {
    std::vector<std::string> species_split;
    utility::splitstr(species_split, species_line, ',');
    species_list.push_back(utility::species_from_string(species_line));
  }
  return utility::species_list_to_struct(species_list);
}

void utility::update_species_list_indicies(species_struct& species_struct) {
  std::vector<int> co2_indices_tmp;
  std::vector<int> co2_ffi_indices_tmp;
  std::vector<int> co2_afolu_indices_tmp;
  std::vector<int> ch4_indices_tmp;
  std::vector<int> n2o_indices_tmp;
  std::vector<int> other_indices_tmp;
  std::vector<int> ghg_indices_tmp;
  std::vector<int> ghg_forward_indices_tmp;
  std::vector<int> ghg_inverse_indices_tmp;
  std::vector<int> aerosol_chemistry_from_emissions_indices_tmp;
  std::vector<int> aerosol_chemistry_from_concentration_indices_tmp;
  std::vector<int> ari_indices_tmp;
  std::vector<int> aci_indices_tmp;
  for (int i = 0; i < species_struct.name.size(); i++) {
    if (species_struct.type.at(i) == "co2") {
      co2_indices_tmp.push_back(i);
    } else if (species_struct.type.at(i) == "co2 ffi") {
      co2_ffi_indices_tmp.push_back(i);
    } else if (species_struct.type.at(i) == "co2 afolu") {
      co2_afolu_indices_tmp.push_back(i);
    } else if (species_struct.type.at(i) == "ch4") {
      ch4_indices_tmp.push_back(i);
    } else if (species_struct.type.at(i) == "n2o") {
      n2o_indices_tmp.push_back(i);
    } else {
      other_indices_tmp.push_back(i);
    }
    if (species_struct.greenhouse_gas(i)) {
      ghg_indices_tmp.push_back(i);
    }
    if (species_struct.type.at(i) == "ari") {
      ari_indices_tmp.push_back(i);
    }
    if (species_struct.type.at(i) == "aci") {
      aci_indices_tmp.push_back(i);
    }
    if (species_struct.aerosol_chemistry_from_emissions(i)) {
      aerosol_chemistry_from_emissions_indices_tmp.push_back(i);
    }
    if (species_struct.aerosol_chemistry_from_concentration(i)) {
      aerosol_chemistry_from_concentration_indices_tmp.push_back(i);
    }
    if ((species_struct.input_mode[i] == "emissions" |
         species_struct.input_mode[i] == "calculated") &
        species_struct.greenhouse_gas(i)) {
      ghg_forward_indices_tmp.push_back(i);
    }
    if ((species_struct.input_mode[i] == "concentrations") &
        species_struct.greenhouse_gas(i)) {
      ghg_inverse_indices_tmp.push_back(i);
    }
  }
  species_struct.co2_indices = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(
      co2_indices_tmp.data(), co2_indices_tmp.size());
  species_struct.co2_ffi_indices =
      Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(co2_ffi_indices_tmp.data(),
                                                    co2_ffi_indices_tmp.size());
  species_struct.co2_afolu_indices =
      Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(
          co2_afolu_indices_tmp.data(), co2_afolu_indices_tmp.size());
  species_struct.ch4_indices = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(
      ch4_indices_tmp.data(), ch4_indices_tmp.size());
  species_struct.n2o_indices = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(
      n2o_indices_tmp.data(), n2o_indices_tmp.size());
  species_struct.other_gh_indices =
      Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(other_indices_tmp.data(),
                                                    other_indices_tmp.size());
  species_struct.ghg_indices = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(
      ghg_indices_tmp.data(), ghg_indices_tmp.size());
  species_struct.ari_indices = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(
      ari_indices_tmp.data(), ari_indices_tmp.size());
  species_struct.aci_indices = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(
      aci_indices_tmp.data(), aci_indices_tmp.size());
  species_struct.aerosol_chemistry_from_concentration_indices =
      Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(
          aerosol_chemistry_from_concentration_indices_tmp.data(),
          aerosol_chemistry_from_concentration_indices_tmp.size());
  species_struct.aerosol_chemistry_from_emissions_indices =
      Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(
          aerosol_chemistry_from_emissions_indices_tmp.data(),
          aerosol_chemistry_from_emissions_indices_tmp.size());
  species_struct.ghg_forward_indices =
      Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(
          ghg_forward_indices_tmp.data(), ghg_forward_indices_tmp.size());
  species_struct.ghg_inverse_indices =
      Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(
          ghg_inverse_indices_tmp.data(), ghg_inverse_indices_tmp.size());
}

species_struct utility::species_list_to_struct(
    const std::vector<species>& species_list) {
  species_struct species_struct(species_list.size());
  for (int i = 0; i < species_list.size(); i++) {
    species_struct.name[i] = species_list[i].name;
    species_struct.type[i] = species_list[i].type;
    species_struct.input_mode[i] = species_list[i].input_mode;
    species_struct.greenhouse_gas(i) = species_list[i].greenhouse_gas;
    species_struct.aerosol_chemistry_from_emissions(i) =
        species_list[i].aerosol_chemistry_from_emissions;
    species_struct.aerosol_chemistry_from_concentration(i) =
        species_list[i].aerosol_chemistry_from_concentration;
    species_struct.partition_fraction.col(i) =
        species_list[i].partition_fraction;
    species_struct.unperturbed_lifetime.col(i) =
        species_list[i].unperturbed_lifetime;
    species_struct.tropospheric_adjustment(i) =
        species_list[i].tropospheric_adjustment;
    species_struct.forcing_efficacy(i) = species_list[i].forcing_efficacy;
    species_struct.forcing_temperature_feedback(i) =
        species_list[i].forcing_temperature_feedback;
    species_struct.forcing_scale(i) = species_list[i].forcing_scale;
    species_struct.molecular_weight(i) = species_list[i].molecular_weight;
    species_struct.baseline_concentration(i) =
        species_list[i].baseline_concentration;
    species_struct.forcing_reference_concentration(i) =
        species_list[i].forcing_reference_concentration;
    species_struct.forcing_reference_emissions(i) =
        species_list[i].forcing_reference_emissions;
    species_struct.iirf_0(i) = species_list[i].iirf_0;
    species_struct.iirf_airborne(i) = species_list[i].iirf_airborne;
    species_struct.iirf_uptake(i) = species_list[i].iirf_uptake;
    species_struct.iirf_temperature(i) = species_list[i].iirf_temperature;
    species_struct.baseline_emissions(i) = species_list[i].baseline_emissions;
    species_struct.g1(i) = species_list[i].g1;
    species_struct.g0(i) = species_list[i].g0;
    species_struct.greenhouse_gas_radiative_efficiency(i) =
        species_list[i].greenhouse_gas_radiative_efficiency;
    species_struct.contrails_radiative_efficiency(i) =
        species_list[i].contrails_radiative_efficiency;
    species_struct.erfari_radiative_efficiency(i) =
        species_list[i].erfari_radiative_efficiency;
    species_struct.h2o_stratospheric_factor(i) =
        species_list[i].h2o_stratospheric_factor;
    species_struct.lapsi_radiative_efficiency(i) =
        species_list[i].lapsi_radiative_efficiency;
    species_struct.land_use_cumulative_emissions_to_forcing(i) =
        species_list[i].land_use_cumulative_emissions_to_forcing;
    species_struct.ozone_radiative_efficiency(i) =
        species_list[i].ozone_radiative_efficiency;
    species_struct.cl_atoms(i) = species_list[i].cl_atoms;
    species_struct.br_atoms(i) = species_list[i].br_atoms;
    species_struct.fractional_release(i) = species_list[i].fractional_release;
    species_struct.ch4_lifetime_chemical_sensitivity(i) =
        species_list[i].ch4_lifetime_chemical_sensitivity;
    species_struct.aci_shape(i) = species_list[i].aci_shape;
    species_struct.aci_scale(i) = species_list[i].aci_scale;
    species_struct.lifetime_temperature_sensitivity(i) =
        species_list[i].lifetime_temperature_sensitivity;
    species_struct.concentration_per_emission(i) =
        species_list[i].concentration_per_emission;
  }
  utility::update_species_list_indicies(species_struct);
  utility::calculate_concentration_per_emission(species_struct);
  return species_struct;
}

void utility::calculate_concentration_per_emission(
    species_struct& species_struct) {
  species_struct.concentration_per_emission =
      1 / (5.1352e18 /  // Mass atmosphere
           1e18 * species_struct.molecular_weight.array() /
           28.97);  // Molecular_weight_air

  return;
}
