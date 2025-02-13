
REFNX=16384
REFNY=16384
sizes:= 16 32 64 128 256 512 1024 2048 4096 8192 8
sizes+=$(REFNX)
sizes_sizes=$(foreach size, $(sizes), $(size)_$(size))
simulators=LxF FORCE HLL HLL2 KP07 KP07_dimsplit WAF
kind_data=space_data time_data
ics=bump dambreak constant

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
.PHONY: clean
all: $(result_targets)
$(simulation_targets): simulate.py
clean:
	@echo "No"

define simulation_template =
ifeq ($(1).$(2),space_data.dambreak)
	ENDFLAG=--tf 6.0
else ifeq ($(1).$(2),space_data.bump)
	ENDFLAG=--tf 1.0
else ifeq ($(1).$(2),space_data.constant)
	ENDFLAG=--tf 1.0
else ifeq ($(1),time_data)
	ENDFLAG=--nt 1000
endif

$(info, $(1).$(2))

$(1)/$(2)/$(3)_$(4)_$(4).npz: simulate.py
	python simulate.py $(2) $(3) --nx $(4) --ny $(4) \
	--ref-nx $(REFNX) --ref-ny $(REFNY) \
	$(ENDFLAG)
endef

# This creates the rule for each of our files!
$(foreach kd, $(kind_data),	$(foreach ic, $(ics), $(foreach simulator, $(simulators), $(foreach size, $(sizes), $(eval $(call simulation_template,$(kd),$(ic),$(simulator),$(size)))))))

#####################
#
#	python calculator_simulator.py space dambreak HLL \
#		--ref space_data/dambreak/HLL_16384_16384.npz \
#		--sizes $(SIZES)
#
define result_template =
ifeq ($(1),space_data)
	REFFLAG=--ref $(1)/$(2)/$(3)_$(REFNX)_$(REFNY).npz
else
	undefine REFFLAG
endif

deps = $(foreach size_size, $(sizes_sizes), $(1)/$(2)/$(3)_$(size_size).npz)

$(1)/results/$(2)/$(3).npz: 
	python calculator_simulator.py $(subst _data,,$(1)) $(2) $(3) \
		$(REFFLAG) \
		--sizes $(sizes_sizes)
endef
$(foreach kd, $(kind_data), $(foreach ic, $(ics), $(foreach simulator, $(simulators), $(eval $(call result_template,$(kd),$(ic),$(simulator))))))