
REFNX ?= 16384
REFNY ?= 16384
sizes ?= 8 16 32 64 128 256 512 1024 2048 4096 8192
simulators ?= LxF FORCE HLL HLL2 KP07 KP07_dimsplit WAF
kind_data ?= space_data time_data
ics ?= constant dambreak bump
sizes_sizes := $(foreach size, $(sizes), $(size)_$(size))
sizes+= $(REFNX)

# simulations are in kind_data/ics/simulators_sizes.npz
simulation_targets = $(foreach kd, $(kind_data), \
					$(foreach ic, $(ics), \
						$(foreach simulator, $(simulators), \
							$(foreach size, $(sizes), \
								$(kd)/$(ic)/$(simulator)_$(size)_$(size).npz))))
#
# calculated values are in kind_data/results/ics/simulators.npz
result_targets = $(foreach kd, $(kind_data), \
				 $(foreach ic, $(ics), \
				 $(foreach simulator, $(simulators), \
				 $(kd)/results/$(ic)/$(simulator).npz)))

# Phony targets are those that do not refer to actual files, only other actions.
# This way make runs the recipe for clean even if there happens to be a file called clean
.PHONY: clean plots help all
all: plots
plots: plots_bump.ipynb plots_dambreak.ipynb
clean:
	@echo "No"

$(simulation_targets): simulate.py
$(result_targets): calculator_simulator.py
help:
	@echo "	Usage:"
	@echo "		make [ics=ICS] [simulators=SIMULATORS] [sizes=SIZES] [kind_data=KIND_DATA]"
	@echo
	@echo "To run only 1 simulation and update the data, run the following:"
	@echo "		make ic=constant simulator=WAF sizes=1024 kind_data=time_data"

plots_%.ipynb: plotter_simulator.ipynb space_data/results/%/*.npz time_data/results/%/*.npz
	papermill plotter_simulator.ipynb $@ -p ic $* 

####################
#
# python simulate.py space dambreak --nx 8 --ny 8
# 	--ref-nx 16384 --ref-ny 16384 --tf 6.0
# USING --force-rerun because make's logic is better than the script's
#
define empty_template = 
endef

define simulation_template =

$(eval 
ifeq ($(1).$(2),space_data.dambreak)
ENDFLAG=--tf 6.0
else ifeq ($(1).$(2),space_data.bump)
ENDFLAG=--tf 1.0
else ifeq ($(1).$(2),space_data.constant)
ENDFLAG=--tf 1.0
else ifeq ($(1),time_data)
ENDFLAG=--nt 1000
endif

ifeq ($(3),LxF)
# 	ENDFLAG+=--cfl 0.6
endif
)

$(1)/$(2)/$(3)_$(4)_$(4).npz:
	python simulate.py $(2) $(3) --nx $(4) --ny $(4) \
--ref-nx $(REFNX) --ref-ny $(REFNY) \
$(ENDFLAG) --force-rerun

endef

# This creates the rule for each of our files!
$(foreach kd,$(kind_data),$(foreach ic,$(ics),$(foreach simulator,$(simulators),$(foreach size,$(sizes),$(eval $(call simulation_template,$(kd),$(ic),$(simulator),$(size)))))))

#####################
#
#	python calculator_simulator.py space dambreak HLL \
#		--ref space_data/dambreak/HLL_16384_16384.npz \
#		--sizes $(SIZES)
#
define result_template =
# The reference file for error calculation. Ignored during time calculations
REFFLAG=--ref $(1)/$(2)/$(3)_$(REFNX)_$(REFNY).npz
# The solution files that this result file depends upon
# All the solution files, basically

$(1)/results/$(2)/$(3).npz: $(foreach size, $(sizes), $(1)/$(2)/$(3)_$(size)_$(size).npz)
	python calculator_simulator.py $(subst _data,,$(1)) $(2) $(3) \
		--ref $(1)/$(2)/$(3)_$(REFNX)_$(REFNY).npz \
		--sizes $(filter-out $(REFNX)_$(REFNY), $(sizes_sizes))

endef

$(foreach kd, $(kind_data), $(foreach ic, $(ics), $(foreach simulator, $(simulators), $(eval $(call result_template,$(kd),$(ic),$(simulator))))))

.NOTPARALLEL: $(simulation_targets)
