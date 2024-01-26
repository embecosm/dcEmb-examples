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

#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <dcEmb/utility.hh>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include "species_struct.hh"

namespace utility {

void splitstr(std::vector<std::string>& vec, const std::string& str,
              const char& delim);

species species_from_string(const std::string& string);
species_struct species_from_file(const std::string& filename,
                                 const std::vector<std::string>& names);
species_struct species_from_file(const std::string& filename);
void update_species_list_indicies(species_struct& species_struct);
species_struct species_list_to_struct(const std::vector<species>& species_list);
void calculate_concentration_per_emission(species_struct& species_struct);

}  // namespace utility
