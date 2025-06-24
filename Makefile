EXAMPLE = fluidizedbed

OLB_ROOT := /Users/varghese/VARGHESE/ice_crystallization_project/lattice_boltzmann/openlb/olb-lammps
include $(OLB_ROOT)/default.mk
CXXFLAGS += -DLAMMPS_SMALLBIG -I/Users/varghese/.local/include
LDFLAGS  += -L/Users/varghese/.local/lib -llammps
