// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: vramasub

#include "GridForceCompute.h"

using namespace std;
namespace py = pybind11;

/*! \file GridForceCompute.cc
    \brief Contains code for the GridForceCompute class
*/

namespace solvent
{

void export_GridForceCompute(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<GridForceCompute, std::shared_ptr<GridForceCompute> >(m, name.c_str(), pybind11::base<ForceCompute>())
        .def(pybind11::init< std::shared_ptr<SystemDefinition> >())
        .def("compute", &GridForceCompute::compute)
        .def("setGrid", &GridForceCompute::setGrid)
    ;
    }

} // end namespace solvent
