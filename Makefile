CC = g++
CFLAGS = -std=c++11 -O3 -MMD 

HILLVALLEA_DIR := ./hv_based_MO_optimization/HillVallEA
HILLVALLEA_SRC_FILES := $(wildcard $(HILLVALLEA_DIR)/*.cpp)
HILLVALLEA_OBJ_FILES := $(patsubst $(HILLVALLEA_DIR)/%.cpp,$(HILLVALLEA_DIR)/%.o,$(HILLVALLEA_SRC_FILES))
HILLVALLEA_DEP_FILES := $(patsubst $(HILLVALLEA_DIR)/%.cpp,$(HILLVALLEA_DIR)/%.d,$(HILLVALLEA_SRC_FILES))

MOHILLVALLEA_DIR := ./domination_based_MO_optimization/mohillvallea
MOHILLVALLEA_SRC_FILES := $(wildcard $(MOHILLVALLEA_DIR)/*.cpp)
MOHILLVALLEA_OBJ_FILES := $(patsubst $(MOHILLVALLEA_DIR)/%.cpp,$(MOHILLVALLEA_DIR)/%.o,$(MOHILLVALLEA_SRC_FILES))
MOHILLVALLEA_DEP_FILES := $(patsubst $(MOHILLVALLEA_DIR)/%.cpp,$(MOHILLVALLEA_DIR)/%.d,$(MOHILLVALLEA_SRC_FILES))

GOMEA_DIR := ./domination_based_MO_optimization/gomea
GOMEA_SRC_FILES := $(wildcard $(GOMEA_DIR)/*.cpp)
GOMEA_OBJ_FILES := $(patsubst $(GOMEA_DIR)/%.cpp,$(GOMEA_DIR)/%.o,$(GOMEA_SRC_FILES))
GOMEA_DEP_FILES := $(patsubst $(GOMEA_DIR)/%.cpp,$(GOMEA_DIR)/%.d,$(GOMEA_SRC_FILES))

BENCHMARK_DIR :=./benchmark_functions
BENCHMARK_SRC_FILES := $(wildcard $(BENCHMARK_DIR)/*.cpp)
BENCHMARK_OBJ_FILES := $(patsubst $(BENCHMARK_DIR)/%.cpp,$(BENCHMARK_DIR)/%.o,$(BENCHMARK_SRC_FILES))
BENCHMARK_DEP_FILES := $(patsubst $(BENCHMARK_DIR)/%.cpp,$(BENCHMARK_DIR)/%.d,$(BENCHMARK_SRC_FILES))

WFG_BENCHMARK_DIR :=./benchmark_functions/wfg_Toolkit
WFG_BENCHMARK_SRC_FILES := $(wildcard $(WFG_BENCHMARK_DIR)/*.cpp)
WFG_BENCHMARK_OBJ_FILES := $(patsubst $(WFG_BENCHMARK_DIR)/%.cpp,$(WFG_BENCHMARK_DIR)/%.o,$(WFG_BENCHMARK_SRC_FILES))
WFG_BENCHMARK_DEP_FILES := $(patsubst $(WFG_BENCHMARK_DIR)/%.cpp,$(WFG_BENCHMARK_DIR)/%.d,$(WFG_BENCHMARK_SRC_FILES))

all: uhv_gomea bezea sofomore_gomea uhv_grad mogomea mamalgam

uhv_gomea: ./hv_based_MO_optimization/main_uhv_gomea.o ./hv_based_MO_optimization/bezier.o ./hv_based_MO_optimization/UHV.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES)  $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 
	$(CC) $(CFLAGS) -o $@ ./hv_based_MO_optimization/main_uhv_gomea.o ./hv_based_MO_optimization/UHV.o ./hv_based_MO_optimization/bezier.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 

sofomore_gomea: ./hv_based_MO_optimization/main_sofomore_gomea.o ./hv_based_MO_optimization/bezier.o ./hv_based_MO_optimization/UHV.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES)  $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 
	$(CC) $(CFLAGS) -o $@ ./hv_based_MO_optimization/main_sofomore_gomea.o ./hv_based_MO_optimization/UHV.o ./hv_based_MO_optimization/bezier.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 

uhv_grad: ./hv_based_MO_optimization/main_uhv_grad.o ./hv_based_MO_optimization/bezier.o ./hv_based_MO_optimization/UHV.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES)  $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 
	$(CC) $(CFLAGS) -o $@ ./hv_based_MO_optimization/main_uhv_grad.o ./hv_based_MO_optimization/UHV.o ./hv_based_MO_optimization/bezier.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 

bezea: ./hv_based_MO_optimization/main_bezea.o ./hv_based_MO_optimization/bezier.o ./hv_based_MO_optimization/UHV.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 
	$(CC) $(CFLAGS) -o $@ ./hv_based_MO_optimization/main_bezea.o ./hv_based_MO_optimization/UHV.o ./hv_based_MO_optimization/bezier.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 

mogomea: ./domination_based_MO_optimization/main_mogomea.o  $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 
	$(CC) $(CFLAGS) -o $@ ./domination_based_MO_optimization/main_mogomea.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 

mamalgam: ./domination_based_MO_optimization/main_mamalgam.o  $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 
	$(CC) $(CFLAGS) -o $@ ./domination_based_MO_optimization/main_mamalgam.o $(HILLVALLEA_OBJ_FILES) $(MOHILLVALLEA_OBJ_FILES) $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_OBJ_FILES) 


%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(HILLVALLEA_OBJ_FILES) $(HILLVALLEA_DEP_FILES) $(MOHILLVALLEA_OBJ_FILES)  $(GOMEA_OBJ_FILES) $(BENCHMARK_OBJ_FILES) $(MOHILLVALLEA_DEP_FILES) $(GOMEA_DEP_FILES) $(BENCHMARK_DEP_FILES) $(WFG_BENCHMARK_OBJ_FILES) $(WFG_BENCHMARK_dep_FILES) ./hv_based_MO_optimization/*.d ./hv_based_MO_optimization/*.o

clean_runlogs:
	rm -f *.dat