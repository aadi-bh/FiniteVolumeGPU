# 
# Makefile for running simulations, calculations, and generating plots.
# Complicated because it generate rules dynamically for each single solution, result, and plots file.
# Run `make help` to get usage.
# Created 13th Feb 2025

# ?= only sets if empty, so can be overriden command line to override
REFNX ?= 16384
REFNY ?= 16384
# Even these can be changed on the cli
sizes ?= 8 16 32 64 128 256 512 1024 2048 4096 8192
simulators ?= LxF FORCE HLL HLL2 KP07 KP07_dimsplit WAF
kind_data ?= space_data time_data
ics ?= constant dambreak bump

# puts the sizes list into "n_n" format
sizes_sizes := $(foreach size, $(sizes), $(size)_$(size))
sizes+= $(REFNX)

# Product of the sets of space/time, ics, simulators, and sizes 
# (with the right path name and extension: simulations are in kind_data/ics/simulators_sizes.npz)
simulation_targets = $(foreach kd, $(kind_data), \
					$(foreach ic, $(ics), \
						$(foreach simulator, $(simulators), \
							$(foreach size, $(sizes), \
								$(kd)/$(ic)/$(simulator)_$(size)_$(size).npz))))

# Simulator .py files
simulator_classes = $(foreach simulator, $(simulators), GPUSimulators/$(simulator).py)
# Another product of sets to generate the calculation files
# calculated values are in kind_data/results/ics/simulators.npz
result_targets = $(foreach kd, $(kind_data), \
				 $(foreach ic, $(ics), \
				 $(foreach simulator, $(simulators), \
				 $(kd)/results/$(ic)/$(simulator).npz)))

# Phony targets are those that do not refer to actual files, only other actions.
# This way make runs the recipe for clean even if there happens to be a file called clean
.PHONY: clean plots help all ref

# First target is the default target
# plots: $(foreach ic,$(ics), plots_$(ic).ipynb)
plots: benchmark_plots_bump.ipynb benchmark_plots_dambreak.ipynb
all: plots

help:
	@echo "	Usage:"
	@echo "		make [plots | all | FILENAME(S) | ref] [ics=ICS] [simulators=SIMULATORS] [sizes=SIZES] [kind_data=KIND_DATA] [REFNX=REFNX] [REFNY=REFNY]"
	@echo
	@echo "To run only 1 simulation and update the data, run the following:"
	@echo "		make ic=constant simulator=WAF sizes=1024 kind_data=time_data"

# To use the matched pattern (% or $*) in the prerequisites we need to use second expansion, so double dollar signs
.SECONDEXPANSION:
benchmark_plots_bump.ipynb benchmark_plots_dambreak.ipynb : plots_%.ipynb: benchmark_plotter.ipynb misc_plotting.py $$(foreach kd,$$(kind_data),$$(foreach simulator, $$(simulators), $$(kd)/results/$$*/$$(simulator).npz))
	papermill benchmark_plotter.ipynb $@ -p ic $* 

$(simulator_classes): GPUSimulators/%.py: GPUSimulators/cuda/SWE2D_%.cu

# Shouldn't make it too easy to delete hours of work
# Prepending with a minus tells make to ignore errors
clean:
	@echo "Delete simulation files manually. Only removing plots and calculations."
	-rm benchmark_plots_bump.ipynb benchmark_plots_dambreak.ipynb
	-rm -r space_data/results/* time_data/results/*

# Declares a common dependency here. Individual rules below
$(simulation_targets): benchmark_simulate.py
$(result_targets): benchmark_postprocess.py

#################### SIMULATION RECIPES ####################
# Syntax reference:
# > python benchmark_simulate.py space dambreak --nx 8 --ny 8
# 	--ref-nx 16384 --ref-ny 16384 --tf 6.0
# USING --force-rerun because make's logic is better than the script's
#

# This template is called for each simulation target
# Order of arguments: 
# 1: kind_data
# 2: ic
# 3: simulator
# 4: size
define simulation_template =

# Eval ensures execution at this point. Fixes some phase issues
# Set the end time or maxsteps depending on the type of simulation
$(eval 
ifeq ($(1).$(2),space_data.dambreak)
ENDFLAG=--tf 6.0
else ifeq ($(1).$(2),space_data.bump)
ENDFLAG=--tf 1.0
else ifeq ($(1).$(2),space_data.constant)
ENDFLAG=--tf 1.0
else ifeq ($(1),time_data)
ENDFLAG=--nt 10000
endif

)

$(1)/$(2)/$(3)_$(4)_$(4).npz:
	python benchmark_simulate.py $(2) $(3) --nx $(4) --ny $(4) \
--ref-nx $(REFNX) --ref-ny $(REFNY) \
$(ENDFLAG) --force-rerun

endef
# EMPTY LINE IS NECESSARY BEFORE `endef`, because `eval` needs that to work properly

# Now call that template for every simulation target
# and then run it through eval again
$(foreach kd,$(kind_data),$(foreach ic,$(ics),$(foreach simulator,$(simulators),$(foreach size,$(sizes),$(eval $(call simulation_template,$(kd),$(ic),$(simulator),$(size)))))))

#################### CALCULATION RECIPES ####################
# Syntax reference:
#	python benchmark_postprocess.py space dambreak HLL \
#		--ref space_data/dambreak/HLL_16384_16384.npz \
#		--sizes $(SIZES)
#

# This template is called for each simulation target
# Order of arguments: 
# 1: kind_data
# 2: ic
# 3: simulator
define result_template =

# The reference file for error calculation. Ignored during time calculations
REFFLAG=--ref $(1)/$(2)/$(3)_$(REFNX)_$(REFNY).npz

# The solution files that this result file depends upon are also generated with loops
$(1)/results/$(2)/$(3).npz: $(foreach size, $(sizes), $(1)/$(2)/$(3)_$(size)_$(size).npz)
	python benchmark_postprocess.py $(subst _data,,$(1)) $(2) $(3) \
		--ref $(1)/$(2)/$(3)_$(REFNX)_$(REFNY).npz \
		--sizes $(filter-out $(REFNX)_$(REFNY), $(sizes_sizes))

endef
# EMPTY LINE IS NECESSARY BEFORE `endef`, because `eval` needs that to work properly

# Now call that template again for every calculation target
# and run the result of the template through eval
$(foreach kd, $(kind_data), $(foreach ic, $(ics), $(foreach simulator, $(simulators), $(eval $(call result_template,$(kd),$(ic),$(simulator))))))

# Don't run `benchmark_simulate.py` in parallel, because we are benchmarking performance
# But it affects all the prerequisites as well

##################### CLAWPACK RECIPE ####################
# Syntax reference:
#	python shallow2d_bump_clawpack.py
# Runs the clawpack script to generate the reference solution for bump case
# then moves and compresses it.

# variable to hide the equals
EQUALS= =
ref: reference/clawpack_nx$(EQUALS)1024.csv.gz
reference/clawpack_nx$(EQUALS)1024.csv.gz: %.gz : %
	gzip $^
reference/clawpack_nx$(EQUALS)1024.csv: reference/shallow2d_bump_clawpack.py
	python reference/shallow2d_bump_clawpack.py && mv _output/fort.q0002 reference/clawpack_nx=1024.csv
	conda deactivate
reference/swashes_*:
	@echo "Generating swashes solutions not implemented."

.NOTPARALLEL: $(simulation_targets)
