# Uncrowded-hypervolume based multi-objective optimization
This repository contains implementations (C++) of different uncrowded-hypervolume based methods for (gradient-free) multi-objective optimization. Using the (uncrowded) hypervolume, multi-objective optimization problems can be formulated as (high-dimensional) single-objective optimization problems, in the hypervolume of a solution sets is optimized. For details, see the corresponding publications/preprints. 

The main publication corresponding to this work is

> Uncrowded Hypervolume-based Multi-objective Optimization with Gene-pool Optimal Mixing
> by S.C. Maree, T. Alderliesten, P.A.N. Bosman, 2020 (https://arxiv.org/abs/2004.05068)

This repository is built upon the HillVallEA framework (https://github.com/scmaree/HillVallEA), which is a framework for black-box (gradient-free) multi-modal optimization. It is combined with MO-HillVallEA  (https://github.com/scmaree/MOHillVallEA), a black-box optimization algorithm for multi-objective optimization. A collection of different optimization approaches is included in this repository. A number of benchmark sets/problems has also been included, such as the Walking Fish Group (WFG) Toolkit, the ZDT problems, and a number of simple benchmark problems. Current implementations supports bi-objective optimization problems. 

# Getting started 

Start by making a clone of the repository,

``` git clone https://github.com/SCMaree/uncrowded-hypervolume```

Call `make` in the root directory to build the C++ project. This builds the following optimizers: `uhv_gomea`, `sofomore_gomea`, `uvh_grad` and `bezea`. Call `make clean` to clean all build files, and call `make clean_runlogs` to clean all runlogs (\*.dat) in the root directory that are generated during a run of an optimizer.

## A simple example

Each optimizer has a minimal description of its input parameters, which is shown by calling it without any parameters, e.g., `./uhv_gomea`. To print a list of installed problems, run the executable with the flag `-P`. In order to add your own problems, have a look at `./benchmark_functions/mo_benchmarks.h`. 

We provide here a simple example with parameters for each algorithm to solve the bi-sphere problem (problem id 26) with 10 decision variables, initialized in the range [-10,10]. 

The aim of optimization is to obtain a solution set of 9 non-dominated solutions with maximal hypervolume value. In this example, the maximally achievable HV is 120.78767307497085. Note that this is specifically for solution set of 9 solutions. 

We use the flag `-r` to enable the value-to-reach, which is determined in terms of the hypervolume, and is set to 120.7876730749. Furthermore, the flag `-v` (verbose) prints run details to the screen, and `-s` enables writing of generational statistics to a file of the name `statistics_ ... .dat` in the provided directory. In the statistics file, the most important details of a run are shown. Note that many of these optimizers are single-objective optimizers, and a single evaluation for these optimizers is the evaluation of an entire set of 9 Multi-Objective solutions. In the statistics file, there is a column MO-evals, that keeps track of the MO-evaluations, which is generally the quantity of interest to measure performance.

### UHV-GOMEA
Uncrowded hypervolume gene-pool optimal mixing evolutionary algorithm (UHV-GOMEA), introduced in,
> Uncrowded Hypervolume-based Multi-objective Optimization with Gene-pool Optimal Mixing
> by S.C. Maree, T. Alderliesten, P.A.N. Bosman, 2020 (https://arxiv.org/abs/2004.05068)

`./uhv_gomea -s -v -r 0 26 10 9 -10 10 31 1000 1000000 60 120.7876730749 1234 "./"`

The first input gives the linkage model (0 = marginal linkage, 1 = linkage tree, 2 = full linkage). Essentially, use marginal linkage (0) when the solution set size is small (e.g. <16), and use the linkage tree (1) else. It is almost never beneficial to use full linkage (2). 

The population size (here: 31) can be set to its default by setting it to 0. This hyperparameter is most problem-dependent, and can be tuned to great extend for the problem at hand. An elitist archive, in which all solutions obtained during the entire run are collected. To enable it, use the flag `-e`. Note: checking for non-dominance in the elitist archive slows down the implementation significantly, so set the target size (here: 1000) not too large.

The maximum number of evaluations is set to 1 million evaluations in this example, runtime is limited to 60 seconds, and the value-to-reach is activated (`-r` flag). The random seed is set to 1234, and result files are written in the current directory.

### Sofomore-GOMEA
The Sofomore framework is introduced in *Uncrowded Hypervolume Improvement: COMO-CMA-ES and the Sofomore framework* by C. Toure et al., 2019. Here, we present a version in which GOMEA is used to perform the internal optimization, as described in the above mentioned publication. 

`./sofomore_gomea -s -v -r 26 10 9 -10 10 31 1000 1000000 60 120.7876730749 1234 "./"`

Sofomore-GOMEA does not require a linkage model to be specified (when the multi-objective problem is considered to be a black-box), and the remainder of the inputs is the same as for `uhv_gomea`.

*Note: In Sofomore-GOMEA, re-evaluation of all solutions is performed at the beginning of each generation. The current implementation is lazy in the sense that it also re-evaluates the multi-objective fitness values, which is not required. These evaluations are not counted in all counters, but when using this code in any other benchmarking setting, this needs to be fixed.*

### UHV-Grad (gradient-based multi-objective optimization)
Uncrowed hypervolume indicator can be used for efficient gradient-based multi-objective optimization, as described in,

> Multi-objective Optimization by Uncrowded Hypervolume Gradient Ascent
> by T. M. Deist, S.C. Maree, T. Alderliesten, P.A.N. Bosman, PPSN 2020 

To run UHV-Grad on the example problem with ADAM as optimizer,

`./uhv_grad -s -v -r 0 26 10 9 -10 10 1000 1000000 60 120.7876730749 1234 "./"`

The first input parameter specifies the gradient descent methods. Currently implemented are the well-known ADAM (0), which is recommended, and the GAMO (1) scheme, as described in the corresponding publication. If the flag `-f`, a finite difference approximation of the gradient is used. This flag is automatically activated when the provided problem has no gradients implemented. 


### BezEA (smoothly navigable approximation sets)
Bezier GOMEA (BezEA) is introduced in,

> Ensuring smoothly navigable approximation sets by Bezier curve parameterizations in evolutionary bi-objective optimization
> by S.C. Maree, T. Alderliesten, P.A.N. Bosman, PPSN 2020 

In BezEA, approximation sets are parameterized as a set of solutions that lie on a smooth Bezier curve in decision space. This allows for an intuitive navigation of the resulting trade-off curve, as all decision variables change in a smooth and continuous fashion. Note that due to this guaranteed smoothness, BezEA cannot generally obtain the same maximal hypervolume VTR as the previously mentioned methods.

`./bezea -s -v -r 26 10 2 9 -10 10 200 1000 100000 60 120.7863059455 1234 "./"`

BezEA had one additional input, which is the number of control points of the Bezier curve. When using 2 points (as shown here), the approximation set is a straight line segment in decision space. BezEA works well for a small number of control points. It is always suggested to set the number of points on the front (here: 9) larger than the number of control points.

### MO-GOMEA
Multi-objective (real-valued) Gene-pool optimal mixing (MO-GOMEA), as introduced in,

> A. Bouter, N.H. Luong, C. Witteveen, T. Alderliesten, and P.A.N. Bosman. The multi-objective real-valued 
> gene-pool optimal mixing evolutionary algorithm, GECCO 2017

MO-GOMEA can be run with

`./mogomea -s -v 26 10 -10 10 200 1000 9 100000 60 0 1234 "./"`

MO-GOMEA is a domination-based algorithm that does not explitly optimize the hypervolume. Especially, it uses a population during optimization, and additionally, it collects all non-dominated solutions in an elitist archive. A small number of the solutions in this archive are placed back into the population every generation. It therefore helps to set the elitist size not too small. A recommended size is 1000. Additionally, after optimization, an approximation set is formed. This can be set very small, e.g. 9, to be able to compare performance of MO-GOMEA to the hypervolume-based methods (here: 9). As MO-GOMEA does not explicitly optimize the hypervolume of the approximation set (or elitist archive/population), it cannot obtain the same value-to-reach that the UHV-based algorithms obtain. However, it can obtain better hypervolume values when the approximation set size is larger, which does not influence the required number of function evaluations (but it does affect runtime).

*Note: This is not the officiel implementation of MO-GOMEA. The ability to exploit partial evaluations is not implemented in this version. For the implementation of the original authors, have a look at https://homepages.cwi.nl/~bosman/source_code.php*

### MAMaLGaM (and MO-HillVallEA)
The domination-based multi-objective AMaLGaM (MAMaLGaM) is an implementation of the algorithm described in,

> S. Rodrigues, P. Bauer, and P.A.N. Bosman. A Novel Population-based Multi-Objective CMA-ES and the 
> Impact of Different Constraint Handling Techniques, GECCO 2014.

MAMaLGaM can be run by,

` ./mamalgam -s -v 26 10 -10 10 200 1000 9 100000 60 0 1234 "./"`

By enabling niching `-n`, MAMaLGaM performs multi-modal optimization, and thereby aims to obtain all solutions in the Pareto set, by explicitly clustering the elitist archive and the population into clusters that each reside in a single mode. For this, hill-valley clustering is used, and the resulting algorithm is called the multi-objective evolutionary algorithm (MO-HillVallEA), and is published in,

> S.C. Maree, T. Alderliesten, and P.A.N. Bosman. Real-valued Evolutionary Multi-modal
> Multi-objective Optimization by Hill-valley Clustering, GECCO 2019

Both algorithms use the same input parameters as MO-GOMEA, and are also domination-based, so do not converge to the optimal hypervolume (see explanation for MO-GOMEA above). 

*Note: Several adaptations and corrections have been implemented in MAMaLGaM compared to the algorithm described by Rodrigues et al.*
















