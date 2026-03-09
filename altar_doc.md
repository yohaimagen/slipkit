# Complete Documentation Scraped from https://altar.readthedocs.io/en/cuda/index.html

## Source: https://altar.readthedocs.io/en/cuda/README.html

# AlTar 2.0 Release[](https://altar.readthedocs.io/en/cuda/README.html#altar-2-0-release "Link to this heading")
## About AlTar[](https://altar.readthedocs.io/en/cuda/README.html#about-altar "Link to this heading")
AlTar(Al-Tar) is a software package to solve inverse problems with Bayesian inference. It is named after the Spanish-French physicist 
AlTar uses primarily the 
## Guides[](https://altar.readthedocs.io/en/cuda/README.html#guides "Link to this heading")
(work in progress, currently for AlTar 2.0 CUDA version only)
Readthedocs ([html](https://altar.readthedocs.io) | [pdf](https://altar.readthedocs.io/_/downloads/en/cuda/pdf/) | [epub](https://altar.readthedocs.io/_/downloads/en/cuda/epub)):
  * [Installation Guide](https://altar.readthedocs.io/en/cuda/cuda/Installation.html)
  * [User Guide](https://altar.readthedocs.io/en/cuda/cuda/Manual.html)
  * [Programming Guide](https://altar.readthedocs.io/en/cuda/cuda/Programming.html)
  * [API Reference](https://altar.readthedocs.io/en/cuda/api/index.html)


## Tutorials[](https://altar.readthedocs.io/en/cuda/README.html#tutorials "Link to this heading")
Tutorials presented with jupyter notebooks:
  * Static slip inversion: a toy model with epistemic uncertainties


## Support[](https://altar.readthedocs.io/en/cuda/README.html#support "Link to this heading")
  * [Common Issues](https://altar.readthedocs.io/en/cuda/cuda/Issues.html)


## Copyright[](https://altar.readthedocs.io/en/cuda/README.html#copyright "Link to this heading")
```
    Copyright (c) 2013-2021 ParaSim Inc.
    Copyright (c) 2010-2021 California Institute of Technology
    All Rights Reserved
    
    This software is subject to the provisions of its LICENSE, a copy of
    which should accompany all distributions, in both source and binary
    form. If you received this software without a copy of the LICENSE,
    please contact the author at michael.aivazis@para-sim.com.
    
    THIS SOFTWARE IS PROVIDED "AS IS" AND ANY AND ALL EXPRESS OR IMPLIED
    WARRANTIES ARE DISCLAIMED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF TITLE, MERCHANTABILITY, AGAINST INFRINGEMENT, AND FITNESS
    FOR A PARTICULAR PURPOSE.

```



---

## Source: https://altar.readthedocs.io/en/cuda/index.html

# AlTar 2.0
##  A Bayesian framework for inverse problems 
Contents
  * [Preface](https://altar.readthedocs.io/en/cuda/cuda/Preface.html)
    * [AlTar 2.0 Release](https://altar.readthedocs.io/en/cuda/README.html)
      * [About AlTar](https://altar.readthedocs.io/en/cuda/README.html#about-altar)
      * [Guides](https://altar.readthedocs.io/en/cuda/README.html#guides)
      * [Tutorials](https://altar.readthedocs.io/en/cuda/README.html#tutorials)
      * [Support](https://altar.readthedocs.io/en/cuda/README.html#support)
      * [Copyright](https://altar.readthedocs.io/en/cuda/README.html#copyright)
    * [Bayesian Inference for Inverse Problems](https://altar.readthedocs.io/en/cuda/cuda/Background.html)
      * [Inverse problem](https://altar.readthedocs.io/en/cuda/cuda/Background.html#inverse-problem)
      * [Bayesian approach](https://altar.readthedocs.io/en/cuda/cuda/Background.html#bayesian-approach)
      * [CATPMIP algorithm](https://altar.readthedocs.io/en/cuda/cuda/Background.html#catpmip-algorithm)
      * [References](https://altar.readthedocs.io/en/cuda/cuda/Background.html#references)
  * [Installation Guide](https://altar.readthedocs.io/en/cuda/cuda/Installation.html)
    * [Overview](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#overview)
    * [Supported Platforms](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#supported-platforms)
      * [Hardware](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#hardware)
      * [Operation systems](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#operation-systems)
      * [Prerequisites](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#prerequisites)
    * [Downloads](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#downloads)
    * [Install with CMake](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#install-with-cmake)
      * [General steps](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#general-steps)
      * [CMake Options](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#cmake-options)
        * [Installation path](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#installation-path)
        * [Enable/disable CUDA](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#enable-disable-cuda)
        * [Target GPU architecture(s)](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#target-gpu-architecture-s)
        * [C++ Compiler](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#c-compiler)
        * [CUDA Compiler](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#cuda-compiler)
        * [BLAS Library](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#blas-library)
        * [Library search path](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#library-search-path)
        * [Build type](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#build-type)
        * [Show compiling details](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#show-compiling-details)
        * [More options](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#more-options)
    * [Conda method (Linux/MacOSX)](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#conda-method-linux-macosx)
      * [Install Anaconda/Miniconda](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#install-anaconda-miniconda)
      * [Install Conda packages](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#install-conda-packages)
      * [C++ Compiler](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#id7)
      * [CUDA compiler (nvcc)](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#cuda-compiler-nvcc)
      * [Download pyre and AlTar](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#download-pyre-and-altar)
      * [Install pyre](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#install-pyre)
      * [Install AlTar](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#install-altar)
      * [MPI setup](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#mpi-setup)
    * [Linux Systems](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#linux-systems)
      * [Ubuntu 18.04/20.04](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#ubuntu-18-04-20-04)
        * [Install prerequisites](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#install-prerequisites)
        * [Install pyre/AlTar](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#install-pyre-altar)
      * [RHEL/CentOS 7](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#rhel-centos-7)
        * [Install prerequisites](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#id9)
        * [Install pyre/AlTar](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#id10)
      * [Linux with software modules](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#linux-with-software-modules)
    * [Docker container](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#docker-container)
    * [Install with the mm build tool](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#install-with-the-mm-build-tool)
      * [Download `mm` build tool](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#download-mm-build-tool)
      * [Prepare a `config.mm` file](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#prepare-a-config-mm-file)
      * [Install pyre](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#id11)
      * [Install AlTar](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#id12)
    * [Tests and Examples](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#tests-and-examples)
  * [User Guide](https://altar.readthedocs.io/en/cuda/cuda/Manual.html)
    * [Overview](https://altar.readthedocs.io/en/cuda/cuda/Overview.html)
    * [QuickStart](https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html)
      * [Prepare the configuration file](https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html#prepare-the-configuration-file)
      * [Prepare input files](https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html#prepare-input-files)
      * [Run an AlTar application](https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html#run-an-altar-application)
      * [Collect and analyze results](https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html#collect-and-analyze-results)
    * [Pyre Basics](https://altar.readthedocs.io/en/cuda/cuda/Pyre.html)
      * [Protocols and Components](https://altar.readthedocs.io/en/cuda/cuda/Pyre.html#protocols-and-components)
      * [Pyre Config Format (`.pfg`)](https://altar.readthedocs.io/en/cuda/cuda/Pyre.html#pyre-config-format-pfg)
    * [AlTar Framework](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html)
      * [Application](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#application)
        * [`Application`](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#Application)
      * [Controller/Annealer](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#controller-annealer)
        * [Worker/AnnealingMethod](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#worker-annealingmethod)
        * [Sampler](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#sampler)
        * [Scheduler](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#scheduler)
        * [Archiver (Output)](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#archiver-output)
      * [Job](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#job)
        * [Configurable Attributes](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#configurable-attributes)
        * [Simulation Size](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#simulation-size)
        * [Single Thread Configuration](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#single-thread-configuration)
        * [Multiple Threads on One Computer](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#multiple-threads-on-one-computer)
        * [Multiple Threads Across Several Computers](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#multiple-threads-across-several-computers)
        * [GPU Configurations](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#gpu-configurations)
      * [Model](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#model)
      * [(Prior) Distributions](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#prior-distributions)
        * [Uniform](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#uniform)
        * [Gaussian](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#gaussian)
        * [Truncated Gaussian](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#truncated-gaussian)
        * [Preset](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#preset)
      * [Other Distributions](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#other-distributions)
    * [Models](https://altar.readthedocs.io/en/cuda/cuda/Models.html)
      * [Static Slip Inversion](https://altar.readthedocs.io/en/cuda/cuda/Static.html)
        * [Static Source model](https://altar.readthedocs.io/en/cuda/cuda/Static.html#static-source-model)
        * [Input](https://altar.readthedocs.io/en/cuda/cuda/Static.html#input)
        * [Configurations](https://altar.readthedocs.io/en/cuda/cuda/Static.html#configurations)
        * [Output](https://altar.readthedocs.io/en/cuda/cuda/Static.html#output)
        * [Moment Distribution](https://altar.readthedocs.io/en/cuda/cuda/Static.html#moment-distribution)
        * [Forward Model Application](https://altar.readthedocs.io/en/cuda/cuda/Static.html#forward-model-application)
        * [Utilities](https://altar.readthedocs.io/en/cuda/cuda/Static.html#utilities)
      * [Static Slip Inversion with Cp: Epistemic Uncertainties](https://altar.readthedocs.io/en/cuda/cuda/cp.html)
      * [Kinematic Slip Inversion](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html)
        * [Kinematic Source Model](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#kinematic-source-model)
        * [Joint Kinematic-Static Inversion](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#joint-kinematic-static-inversion)
        * [Configurations (Kinematic Model only)](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#configurations-kinematic-model-only)
        * [Configurations (Joint inversion)](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#configurations-joint-inversion)
        * [Examples](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#examples)
        * [Forward Model Application (new version)](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#forward-model-application-new-version)
        * [Forward Model Application (old version)](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#forward-model-application-old-version)
  * [Programming Guide](https://altar.readthedocs.io/en/cuda/cuda/Programming.html)
    * [Introduction](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#introduction)
    * [Code Organization](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#code-organization)
    * [Bayesian Model](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#bayesian-model)
    * [Model with the BayesianL2 template](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#model-with-the-bayesianl2-template)
      * [Parametersets(psets)](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#parametersets-psets)
      * [Data observations(dataobs) with L2-norm](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#data-observations-dataobs-with-l2-norm)
      * [Forward modeling](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#forward-modeling)
    * [C/C++/CUDA extension modules](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#c-c-cuda-extension-modules)
    * [CUDA Models](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#cuda-models)
    * [Data Types and Structures](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#data-types-and-structures)
      * [Configurable properties](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#configurable-properties)
      * [Matrix/Vector (GSL)](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#matrix-vector-gsl)
        * [Convert altar array to gsl_vector](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#convert-altar-array-to-gsl-vector)
        * [Basic matrix/vector operations](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#basic-matrix-vector-operations)
        * [Interfacing as numpy arrays](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#interfacing-as-numpy-arrays)
  * [Common Issues](https://altar.readthedocs.io/en/cuda/cuda/Issues.html)
    * [Installation Issues](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#installation-issues)
      * [Cannot find `gmake`](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#cannot-find-gmake)
      * [Cannot find `cublas_v2.h`](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#cannot-find-cublas-v2-h)
    * [Run-time Issues](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#run-time-issues)
      * [Locales](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#locales)
      * [Base case name](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#base-case-name)
      * [Configuration Parser Error](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#configuration-parser-error)
      * [MPI launcher error](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#mpi-launcher-error)
      * [Intel MKL Library](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#intel-mkl-library)


References
  * [API Reference](https://altar.readthedocs.io/en/cuda/api/index.html)


# Indices and tables[](https://altar.readthedocs.io/en/cuda/index.html#indices-and-tables "Link to this heading")
  * [Index](https://altar.readthedocs.io/en/cuda/genindex.html)
  * [Module Index](https://altar.readthedocs.io/en/cuda/py-modindex.html)
  * [Search Page](https://altar.readthedocs.io/en/cuda/search.html)




---

## Source: https://altar.readthedocs.io/en/cuda/cuda/Background.html

# Bayesian Inference for Inverse Problems[](https://altar.readthedocs.io/en/cuda/cuda/Background.html#bayesian-inference-for-inverse-problems "Link to this heading")
## Inverse problem[](https://altar.readthedocs.io/en/cuda/cuda/Background.html#inverse-problem "Link to this heading")
An inverse problem in science is to infer a set of unknown parameters, θ={θ1,θ2,…,θm}, from data observations d={d1,d2,…,dn}. For example, in Seismology, we deduce the earthquake rupture information (parameterized as θ) from ground motions (d) measured by seismometers, GPS, InSAR and other geodetic surveys. In most cases, the forward problem d=G(θ) is well defined but the inverse θ=G−1(d) is not. This happens to linear systems G(θ)=Gθ when the observation matrix G is ill-posed, and nonlinear systems where the inverse is inherently difficult.
## Bayesian approach[](https://altar.readthedocs.io/en/cuda/cuda/Background.html#bayesian-approach "Link to this heading")
Bayesian inference offers a statistical solution to inverse problems by modeling unknown parameters θ as random variables subject to a conditional probability P(θ|d). According to Bayes’ theorem, the probability is given by
P(θ|d)=P(θ)P(d|θ)P(d),P(θ|d):posterior, the probability of observing θ given that d is true,P(θ):prior, the probability of observing θ without regard to d,P(d|θ):likelihood, the probability of observing d given that θ is true,P(d):the probability of observing d independent of θ.
Since P(d) is the same for all possible θ and we treat it as a constant 1. The likelihood P(d|θ) can be determined from the forward modeling, for example, assuming a form of Gaussian probability density,
P(d|θ)∝e−12[d−G(θ)]TCχ−1[d−G(θ)],
where the covariance matrix Cχ captures noises/errors in observations d, as well as uncertainties in the forward function G(θ).
The set of model parameters θ with the maximum posterior probability (MAP), or the mean or median model, provides an estimate solution to the inverse problem. Compared to other point estimators, the Bayesian approach has several advantages. By fully evaluating the posterior probability densities, the Bayesian approach can also address situations when several sets of θ have equal or comparable probabilities, or the posterior distribution is not unimodal. In addition, the Bayesian approach accommodates uncertainty quantification, by including various types of uncertainties in the calculation.
The Bayesian approach is extremely intensive computationally, since it requires repeating the forward modeling for all possible θ. With the improved sampling algorithms and computing powers, e.g., GPU, it now becomes feasible for many inverse problems with high dimensional parameter space and/or complex forward models.
## CATPMIP algorithm[](https://altar.readthedocs.io/en/cuda/cuda/Background.html#catpmip-algorithm "Link to this heading")
The posterior distribution P(θ|d) can be determined, for example, by evaluating P(θ)P(d|θ) for all possible θ. Instead of uniformly sampling over the entire θ-space, we rely on Markov-Chain Monte-Carlo (MCMC) methods, which draw efficiently _weighted_ samples pursuant to a prescribed probability distribution.
The CATMIP (Cascading Adaptive Transitional Metropolis in Parallel) algorithm belongs to a class of MCMC methods which use transitioning: samples are initially drawn from the prior distribution and subsequently _annealed_ to the posterior through a series of transient distributions,
Pm(θ|d)=P(θ)P(d|θ)βm,
where βm (in analogy to the inverse of temperature in statistical physics) is gradually increased from β0=0 to βM=1 in M steps.
The procedure of CATMIP is described as follows:
  1. For the m=0 β-step, with β0=0, generate Ns random samples from the distribution P0(θ|d)=P(θ), as seeds of Ns parallel chains.
  2. Determine βm+1 for the next β-step, e.g., from the statistics of the samples from the βm-step. CATMIP uses an annealing schedule based on the coefficient of variance (COV) of the importance weights wi=P(d|θi)βm+1−βm. The targeted COV is typically set as 1, or the effective sample size (ESS) equals to 50%.
  3. Perform a resampling of samples based on their importance weights {wi}: some samples with small weights may be discarded while some samples with large weights are duplicated. The total number of samples is kept the same, as seeds for Ns chains in the βm+1-step.
  4. With the new βm+1, a burn-in MCMC process is preformed with the Metropolis-Hastings algorithm. New samples are proposed from the distribution Pm(θ|d), assuming a multivariate Gaussian form, and accepted/rejected subject to the probability density Pm+1(θ|d). The acceptance ratio is also recorded, which is used to scale the jump distances for proposals in the next β-step.
  5. Repeat Steps 2-4 until βM=1 is reached, or the desired equilibrium distribution P(θ|d) is achieved.


Step 4 presents the majority of the computational load. Since each chain is updated independently, the computation is embarrassingly parallel, which makes CATMIP an ideal algorithm to be implemented on parallel computers, including GPUs.
## References[](https://altar.readthedocs.io/en/cuda/cuda/Background.html#references "Link to this heading")
  1. Albert Tarantola, _Inverse Problem Theory and Methods for Model Parameter Estimation_ , SIAM, 2005. ISBN: 978-0-89871-572-9.
  2. Sarah E. Minson, Mark Simons, and James L. Beck, _Bayesian inversion for finite fault earthquake source models I — theory and algorithm_ , Geophysical Journal International, Vol. 194, 1701 (2013).




---

## Source: https://altar.readthedocs.io/en/cuda/cuda/Programming.html

# Programming Guide[](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#programming-guide "Link to this heading")
## Introduction[](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#introduction "Link to this heading")
This guide aims to help developers to write their own models for AlTar.
  * AlTar’s framework is developed in Python while the compute-intensive routines are developed as C/C++ or CUDA(for GPUs) extension modules, to offer a user-friendly interface without scarifying the efficiency.
  * AlTar follows the `components`-based programming model of _at runtime_ :
>     * to choose between different implementations of a prescribed functionality (protocol), e.g., to choose different Metropolis sampling algorithms;
>     * to turn features on or off, e.g., the debugger, profiler;
>     * parameters and other attributes are also implemented as configurable properties (traits).
  * AlTar utilizes the job management system from 


## Code Organization[](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#code-organization "Link to this heading")
New models can be added to the `altar/models` directory. We recommend the AlTar/pyre convention to organize the source code inside the directory.
The simplest model can be constructed all by Python. The minimum set of files includes, e.g., for a new model named `regression`,
```
altar/models/regression # root
├── CMakeLists.txt # cmake script
├── bin  # binary commands
│   └── regression # command to invoke AlTar simulation for this model
├── regression # python scripts
│   ├── __init__.py # export modules at the package level
│   ├── meta.py.in # metadata includes version/copyright/authors ...
│   └── Regression.py # the main Python program
└── examples # (optional) provide examples to users
    ├── regression.pfg # an example of the configuration file
    └── synthetic # a directory includes data files for the example

```

If C/C++/Fortran routines are needed, which are wrapped as extension modules in Python, these additional files can be arranged as
```
altar/models/regression # root
├── lib # C/C++/Fortran routines compiled as shared libraries
│   └── libregression # compiled to INSTALL_DIR/lib/libregression.so(dylib)
│      ├── version.h # version header file
│      ├── version.cc  # version code
│      ├── Source.h  # C++ source code header file
│      ├── Source.icc  # C++ source code header include file for static methods/variables
│      └── Source.cc # # C++ source code
├── ext  # CPython extension modules
│   └── regression # compiled to regression.cpython-{}.so
│      ├── capsules.h # Python Capsule (pass pointers of C++ objects) definitions
│      ├── regression.cc  # extension module definition
│      ├── source.h  # header file for wrappers
│      ├── source.cc  # CPython wrappers for C/C++/Fortran routines/classes
│      ├── metadata.h  # metadata header file includes version ...
│      └── metadata.cc # # metadata source file includes version ...
└── regression # python scripts
    └── ext  # to host the regression.cpython-{}.so product
       └── __init__.py # export extension modules at the package level

```

GPU/CUDA models can be constructed in the same fashion. The following files will be added,
```
altar/models/regression # root
├── lib # C/C++/Fortran routines compiled as shared libraries
│   └── libcudaregression # compiled to INSTALL_DIR/lib/libcudaregression.so(dylib)
│      ├── version.h # version header file
│      ├── version.cc  # version code
│      ├── Source.h  # CUDA source code header file
│      ├── Source.icc  # CUDA source code header include file for static methods/variables
│      └── Source.cu # # CUDA source code
├── ext  # CPython extension modules
│   └── cudaregression # compiled to cudaregression.cpython-{}.so
│      ├── capsules.h # Python Capsule (pass pointers of C++ objects) definitions
│      ├── regression.cc  # extension module definition
│      ├── source.h  # header file for wrappers
│      ├── source.cc  # CPython wrappers for C/C++/Fortran routines/classes
│      ├── metadata.h  # metadata header file includes version ...
│      └── metadata.cc # # metadata source file includes version ...
└── regression # python scripts
    ├──cuda #
    │  ├── __init__.py # export modules at the package level
    │  ├── meta.py.in # metadata includes version/copyright/authors ...
    │  └── cudaRegression.py # the main Python program
    └── ext  # to host the regression.cpython-{}.so product
       └── __init__.py # export cuda extension modules as well

```

Additional compile/build scripts are also required, for either CMake or (new) MM build tools.
For `CMAKE`, in addition to the `CMakeLists.txt` file under the model directory, these files need to be added to modified,
```
altar # root directory
├── .cmake # for cmake
│   └── altar_regression.cmake # functions to build packages/lib/modules/driver(bin)
└── CMakeLists.txt # to include the new model

```

For (new) MM build tool,
```
altar # root directory
└── .mm # for cmake
   ├── regression.def # new model configurations
   └── altar.mm # to include the new model

```

Details for the above files will be illustrated by specific examples in the following sections.
## Bayesian Model[](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#bayesian-model "Link to this heading")
An AlTar application can be broadly separated into two parts, the MCMC framework for Bayesian statistics and a Bayesian Model who is responsible for calculating the prior/data likelihood/posterior probabilities for a given set of parameters θ.
Specifically, a Model is required to provide the methods as listed in the 
```
# services for the simulation controller
@altar.provides
def initialize(self, application):
    """
    Initialize the state of the model given a {problem} specification
    """

@altar.provides
def initializeSample(self, step):
    """
    Fill {step.theta} with an initial random sample from my prior distribution.
    """

@altar.provides
def priorLikelihood(self, step):
    """
    Fill {step.prior} with the likelihoods of the samples in {step.theta} in the prior
    distribution
    """

@altar.provides
def dataLikelihood(self, step):
    """
    Fill {step.data} with the likelihoods of the samples in {step.theta} given the available
    data. This is what is usually referred to as the "forward model"
    """

@altar.provides
def posteriorLikelihood(self, step):
    """
    Given the {step.prior} and {step.data} likelihoods, compute a generalized posterior using
    {step.beta} and deposit the result in {step.post}
    """

@altar.provides
def likelihoods(self, step):
    """
    Convenience function that computes all three likelihoods at once given the current {step}
    of the problem
    """

@altar.provides
def verify(self, step, mask):
    """
    Check whether the samples in {step.theta} are consistent with the model requirements and
    update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
    """

# notifications
@altar.provides
def top(self, annealer):
    """
    Notification that a β step is about to start
    """

@altar.provides
def bottom(self, annealer):
    """
    Notification that a β step just ended
    """

```

and these methods are called by the AlTar framework at respective places. (`@altar.provides` decorated methods are similar to C++ virtual functions for which the derived classes must declare).
  * `initialize` is to initialize various parameters and settings of the Model, as also being required for other components in AlTar. It takes `application`, the root component, as inputs in order to pull information from other components, e.g. obtaining the number of chains and the processor information (cpu/gpu) from the `job` component.
  * Most other methods take 
    * the parameters being sampled, θ, in 2d array `shape=(samples, parameters)`,
    * the prior, data likelihood, and posterior probability densities for samples, each in 1d vector `shape=samples`.
Note that AlTar processes a batch of samples (number of Markov Chains) in parallel.
  * `initializeSample` is to generate random samples from a prior distribution in the beginning of the simulations. Samples can also be loaded from pre-calculated values using the `preset` prior.
  * `priorLikelihood` computes the prior probability densities from the given prior distribution(s). During the simulation, when new proposed samples fall outside of the support of certain ranged distributions, the density is 0 and therefore the proposals are invalid. Because AlTar uses the logarithmic value of the densities, we need an extra `verify` method to check the ranged priors.
  * `dataLikelihood` computes the data likelihood. It performs
    * the forward modeling, calculating the data predictions from θ,
    * computes the residual between data predictions and observations,
    * and return the data likelihood with a given Norm function (e.g., L2-Norm).
  * `posteriorLikelihood` computes the posterior probability densities from prior and data likelihood. For transitional posterior distributions in annealing schemes, it is simply prior+β∗data.
  * `top` and `bottom` methods are hooks for developers to insert model-specific procedures before or after each annealing step. For example, for models considering model uncertainties, these methods are the places to invoke computing Cp and updating the covariance matrix (Cd+Cp).


Many models may share the same procedures to compute the prior, data likelihood, and posterior; they may differ in forward modeling. We offer some templates to simplify the model development.
## Model with the BayesianL2 template[](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#model-with-the-bayesianl2-template "Link to this heading")
A `forwardModel` method which performs the forward modeling. This template offers the easiest approach to write a new model.
We use the linear regression model to demonstrate how to construct a Bayesian model with the BayesianL2 template in the following. The linear regression model fits a group of data (xn,yn) with a linear function
y=slope×x+intercept+ϵi
where `slope` and `intercept` are parameters to be sampled while ϵi are random noises. The Python program for this example is available at 
### Parametersets(psets)[](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#parametersets-psets "Link to this heading")
In the linear regression model, θ=[slope,intercept]. In principle, slope and intercept could have different prior distributions. Each set of parameters with the same prior distribution is defined as a (contiguous) parameter set. θ can therefore be described as a parametersets (psets) with each parameter set arranged sequentially and contiguously in 1d vector (or columns in batched samples). In Python, `psets` is a `dict` with a collection of `contiguous` parameter set objects. We further define a list `psets_list` to assure the orders of parameter sets in θ (it is also useful in model ensembles to select model-relevant parameter sets from the global `psets`).
A description of `psets` in the configuration file `linear.pfg` appears as
```
; parameter sets
psets_list = [slope, intercept]
psets:
    slope = contiguous
    intercept = contiguous

    slope:
        count = 1
        prep = uniform
        prep.support = (0, 5)
        prior = uniform
        prior.support = (0, 5)

    intercept:
        count = 1
        prep = uniform
        prep.support = (0, 5)
        prior = uniform
        prior.support = (0, 5)

```

where slope and intercept are each described as a `contiguous` parameter set and each set has its own `prep` (to initialize random samples and `prior` (for verify and prior probability) distributions.
We use the static earthquake inversion as another example, where the sampling parameters are dip-slip (Dd), strike-slip (Ds) displacements for N-patches, and if necessary the inSAR ramping parameters (R). Each is described by a `contiguous` parameter set and assembled as a `psets`. (Each row of) θ is therefore (Dd1,Dd2,…,DdN,Ds1,Ds2,…,DsN,R1,R2,…). The order of sets Dd, Ds and R can be switched as long as it is consistent with the forward modeling, e.g., the Green’s functions. If you want to use different priors for strike slips in different patches, you may separate Ds into several parameter sets.
With `psets`, various methods related to parameters, including `initializeSamples`, `verify` and `priorLikelihood`, are pre-defined in BayesianL2 template. Users only need to specify its included parameter sets in the configuration file.
### Data observations(dataobs) with L2-norm[](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#data-observations-dataobs-with-l2-norm "Link to this heading")
L2-norm is recommended for models which need to incorporate various uncertainties. The data likelihood is computed as
log⁡(DataLikelihood)=−12[dpred−dobs]Cχ−1[dpred−dobs]T+C
where dpred=G(θ) are the data predictions from the forward model, dpred the data observations, Cχ the covariance matrix capturing data (Cd) and/or model(Cp) uncertainties, and C a normalization constant depending on the determinant of Cχ.
In BayesianL2 template, the observed data are described by a `dataobs` object with L2-norm. It includes
  * observed data points (`dataobs.dataobs`) in 1d vector with `shape=observations`, `observations` is the number of observed data points;
  * data covariance matrix (`dataobs.cd`) in 2d array with `shape=(observations, observations)`. (If only constant diagonal elements are available, use `cd_std` instead).


`dataobs` is responsible for
  * loading the data observations (dobs) and the data covariance (Cd),
  * calculating the Cholesky decomposition of the inverse Cd and saving it in `dataobs`,
  * when called by the `model.dataLikelihood`, computing the L2-norm (likelihood) between data predictions and observations with L2-norm,
  * for Cp models, updating the covariance matrix Cχ=Cd+Cp (though still denoted as Cd),
  * when needed, merging the covariance matrix with data in `initialize`, as controlled by a flag `dataobs.merge_cd_with_data=True/False`. This procedure improves performance greatly for models when the covariance matrix can also be merged with model parameters, such as the Green’s functions in the linear model, by avoiding repeating the matrix-vector (or matrix-matrix for batched) multiplication.


For the linear regression model, the data points (xn,yn) don’t fit perfectly into the `dataobs` description. Instead, we treat yn as data observations and xn as model parameters. We need to initialize them with the `initialize` method,
```
@altar.export
def initialize(self, application):
    """
    Initialize the state of the model
    """

    # call the super class initialization
    super().initialize(application=application)
    # super class method loads and initializes dataobs with y_n

    # model specific initialization after superclass
    # grab data
    self.x = self.loadFile(self.x_file)
    self.y = self.dataobs.dataobs

    ... ...

```

so that `x` and `y` are now accessible by the forward modeling.
Their descriptions in the configuration file `linear.pfg` appear as, e.g.,
```
case = synthetic ; input directory
dataobs:
    observations = 200
    data_file = y.txt
    cd_std = 1.e-2
x_file = x.txt

```

where (xn,yn) are separated into two text files (raw binary and H5 input files are also supported by the `loadFile` function).
With L2-norm `dataobs`, the `dataLikelihood` method can be defined straightforwardly: it calls a `forwardModel` defined for a specific model and with the data predictions or residuals it calls dataobs’ norm method to compute the likelihood.
### Forward modeling[](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#forward-modeling "Link to this heading")
As shown above, with `psets` and `dataobs`, the BayesianL2 template only requires developers to write a `forwardModel` or `forwardModelBatched` method to compute the data predictions from a given set of θ.
Developers have the options to
  * Choose whether to implement `forwardModelBatched` or `forwardModel`. The `forwardModelBatched` computes the data predictions for all samples. A pre-defined version iterates over all samples (rows of θ) and calls `forwardModel` which computes the data predictions for one sample. The batched mode sometimes improves the speed, e.g., in the linear model, one may use the matrix-matrix product (a routine commonly optimized for both CPU and GPU) to compute d=Gθ. In this case, a customized `forwardModelBatched` method may be defined to override the one pre-defined in BayesianL2.
  * Choose to simply compute the data predictions or return the residuals between predictions and observations, depending on the performance or convenience; This is controlled by a flag `return_residual = True/False` which can be specified either in `model.initialize` code or in the configuration file `model.return_residual=True/False`.
  * How to incorporate the data covariance (Cd). If the model uncertainties (Cp) are also considered, please refer to the examples such as `models/seismic/staticCp`.


For the linear regression model, the simplest implementation is to use the `forwardModel` method for a single set of parameters, and the code appears as
```
def forwardModel(self, theta, prediction):
    """
    Forward Model
    :param theta: sampling parameters for one sample
    :param prediction: data prediction or residual (prediction - observation)
    :return:
    """

    # grab the parameters from theta

    slope = theta[0]
    intercept = theta[1]

    # calculate the residual between data prediction and observation
    size = self.observations
    for i in range(size):
        prediction[i] = slope * self.x[i] + intercept - self.y[i]

    # all done
    return self

```

we also need to specify the flag `return_residual = True` accordingly.
We can also use the batch method `forwardModelBatched`, where we can construct a 2d array `G=[[x1, x2, ..., xN], [1, 1, ... ,1]]`, and simply use the matrix-matrix product `gemm` to calculate the predictions for all samples dpred=Gθ.
We now complete a new model with the BayesianL2 template. Note that if either parameter sets or L2-`dataobs` descriptions doesn’t fit your model, you may write your own methods following the `Model` protocol.
Please also note that vectors/matrices in AlTar are based on GSL, while they can operate as numpy arrays. But if you would like to use some numpy/scipy functions on numpy arrays, on you may create a numpy ndarray view or copy from GSL vectors/matrices. See [Matrix/Vector (GSL)](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#matrix-vector-gsl) for more details.
## C/C++/CUDA extension modules[](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#c-c-cuda-extension-modules "Link to this heading")
For certain procedures, the Python code might not be efficient and you might want to write them in other high performance programming languages. For example, the Linear Algebra libraries (BLAS, LAPACK) are written in FORTRAN/C while are accessible in Python by extension modules.
While there are many convenient tools to write Python wrappers for C/C++/Fortran/CUDA functions, such as `cython`, `SWIG`, `Boost.Python`, `pybind11`, we recommend the native CPython method.
Let’s use an example of vector copy. Define in `copy.h`:
```
// vector_copy
extern const char * const vector_copy__name__;
extern const char * const vector_copy__doc__;
PyObject * vector_copy(PyObject *, PyObject *);

```

The source code `copy.cc`
```
PyObject *
vector_copy(PyObject *, PyObject * args) {
// the arguments
PyObject * sourceCapsule;
PyObject * destinationCapsule;
// unpack the argument tuple
int status = PyArg_ParseTuple(
                              args, "O!O!:vector_copy",
                              &PyCapsule_Type, &destinationCapsule,
                              &PyCapsule_Type, &sourceCapsule
                              );
// if something went wrong
if (!status) return 0;
// bail out if the source capsule is not valid
if (!PyCapsule_IsValid(sourceCapsule, capsule_t)) {
    PyErr_SetString(PyExc_TypeError, "invalid vector capsule for source");
    return 0;
}
// bail out if the destination capsule is not valid
if (!PyCapsule_IsValid(destinationCapsule, capsule_t)) {
    PyErr_SetString(PyExc_TypeError, "invalid vector capsule for destination");
    return 0;
}

// get the vectors
gsl_vector * source =
    static_cast<gsl_vector *>(PyCapsule_GetPointer(sourceCapsule, capsule_t));
gsl_vector * destination =
    static_cast<gsl_vector *>(PyCapsule_GetPointer(destinationCapsule, capsule_t));
// copy the data
gsl_vector_memcpy(destination, source);

// return None
Py_INCREF(Py_None);
return Py_None;
}

```

## CUDA Models[](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#cuda-models "Link to this heading")
TBD
## Data Types and Structures[](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#data-types-and-structures "Link to this heading")
### Configurable properties[](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#configurable-properties "Link to this heading")
```
altar.properties.array
altar.properties.bool
altar.properties.catalog
altar.properties.complex
altar.properties.converter
altar.properties.date
altar.properties.decimal
altar.properties.dict
altar.properties.dimensional
altar.properties.envpath
altar.properties.envvar
altar.properties.facility
altar.properties.float
altar.properties.identity
altar.properties.inet
altar.properties.int
altar.properties.istream
altar.properties.list
altar.properties.normalizer
altar.properties.object
altar.properties.ostream
altar.properties.path
altar.properties.paths
altar.properties.property
altar.properties.set
altar.properties.str
altar.properties.strings
altar.properties.time
altar.properties.tuple
altar.properties.uri
altar.properties.uris
altar.properties.validator

```

### Matrix/Vector (GSL)[](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#matrix-vector-gsl "Link to this heading")
The Matrix/Vector is based on GSL Matrix/Vector.
GSL matrix shape (size1, size2) -> (rows, cols) and is row-major.
#### Convert altar array to gsl_vector[](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#convert-altar-array-to-gsl-vector "Link to this heading")
```
array = altar.properties.array(default=(0, 0, 0,0))
gvector = altar.vector(shape=4)
for index, value in enumerate(array): gvector[index] = value

```

#### Basic matrix/vector operations[](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#basic-matrix-vector-operations "Link to this heading")
```
mat = altar.matrix(shape=(rows, cols)) # create a new matrix with dimension rows x cols (row-major)
mat.zero() # initialize 0 to each element
mat.fill(number) #
mat_clone = mat.clone() #
mat1.copy(mat2)

```

#### Interfacing as numpy arrays[](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#interfacing-as-numpy-arrays "Link to this heading")
As there are more utilities available for numpy `ndarray`, you may view or copy GSL vectors/matrices are numpy arrays.
```
# create a gsl vector
gslv = altar.vector(shape=10).fill(1)
# create a numpy array view (data changes to npa_view will change gslv)
npa_view = gslv.ndarray()
# create a numpy array view (data changes to npa_view don't affect gslv)
npa_copy = gslv.ndarray(copy=True)

```



---

## Source: https://altar.readthedocs.io/en/cuda/cuda/cp.html

# Static Slip Inversion with Cp: Epistemic Uncertainties[](https://altar.readthedocs.io/en/cuda/cuda/cp.html#static-slip-inversion-with-cp-epistemic-uncertainties "Link to this heading")
In addition to observational errors, one can calculate epistemic uncertainties (Minson et al., 2013), which affect the forward model and are related to our imperfect knowledge of the Earth interior. Epistemic uncertainties will stem from a various set of parameters, but the most prominent uncertainties will derive from elastic heterogeneity (Duputel et al., 2014) and fault geometry (Ragon et al., 2018, 2019).
To estimate epistemic uncertainties, we assume that the real surface displacement follows a Gaussian distribution centred on the predictions with a prediction covariance matrix Cp that depends on the resulting source model. The misfit covariance matrix characterizing the misfit function becomes `Cx = Cd + Cp`.
**First approach: static Cp**
The prediction covariance matrix `Cp` can be included in the inversion following two different approaches. In the first, `Cp` is calculated a priori and included in the inversion process within `Cx`. In this case, the full `Cx` matrix replaces the `Cd ` matrix:
```
    dataobs = altar.cuda.data.datal2
    dataobs:
        observations = 1000
        data_file = d.txt
        cd_file = cx.txt ; Cx contains both Cd and Cp

```

**Second approach: Updated Cp**
The alternative approach, implemented in AlTar, is to update Cp with interim models at each step of the tempered inversion. The covariance matrix `Cp` is calculated from 3 variables: a sensitivity kernel for every investigated parameter, the standard deviation of the a priori distribution of investigated parameters, and an initial model chosen a priori. In this approach, `Cp` is re-calculated at each tempering step by choosing the mean of tested samples as the assumed initial model.
```
    ; sensitivity kernels
    nCmu = 8 ; number of investigated parameters
    cmu_file = Cmu.h5 ; standard deviation matrix
    kmu_file = Kelastic.h5 ; tensor of sensitivity kernels (matrix if only one investigated parameter)
    initial_model_file = inimodel.txt ; initial model vector

```

Additionnally, `Cp` can be incorporated in the inversion at any tempering step. The tempering step will be selected with its corresponding beta value, parameterized by `beta_cp_start`. If `beta_cp_start = 0.01`, then `Cp` will be incorporated from the beginning of the inversion process. If `beta_cp_start = 0.1` or `0.5`, then `Cp` will be incorporated after a few tempering steps, or once beta reaches 0.5, respectively. `beta_use_initial_model` will indicate if the initial model should be used (`1`) or not (`0`). If not, then a unit uniform initial model is used.
```
    beta_cp_start = 0.01 ; beta value to start incorporating Cp
    beta_use_initial_model = 1 ; use provided initial model

```



---

## Source: https://altar.readthedocs.io/en/cuda/cuda/Manual.html

# User Guide[](https://altar.readthedocs.io/en/cuda/cuda/Manual.html#user-guide "Link to this heading")
  * [Overview](https://altar.readthedocs.io/en/cuda/cuda/Overview.html)
  * [QuickStart](https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html)
    * [Prepare the configuration file](https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html#prepare-the-configuration-file)
    * [Prepare input files](https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html#prepare-input-files)
    * [Run an AlTar application](https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html#run-an-altar-application)
    * [Collect and analyze results](https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html#collect-and-analyze-results)
  * [Pyre Basics](https://altar.readthedocs.io/en/cuda/cuda/Pyre.html)
    * [Protocols and Components](https://altar.readthedocs.io/en/cuda/cuda/Pyre.html#protocols-and-components)
    * [Pyre Config Format (`.pfg`)](https://altar.readthedocs.io/en/cuda/cuda/Pyre.html#pyre-config-format-pfg)
  * [AlTar Framework](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html)
    * [Application](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#application)
      * [`Application`](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#Application)
    * [Controller/Annealer](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#controller-annealer)
      * [Worker/AnnealingMethod](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#worker-annealingmethod)
      * [Sampler](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#sampler)
      * [Scheduler](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#scheduler)
      * [Archiver (Output)](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#archiver-output)
    * [Job](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#job)
      * [Configurable Attributes](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#configurable-attributes)
      * [Simulation Size](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#simulation-size)
      * [Single Thread Configuration](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#single-thread-configuration)
      * [Multiple Threads on One Computer](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#multiple-threads-on-one-computer)
      * [Multiple Threads Across Several Computers](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#multiple-threads-across-several-computers)
      * [GPU Configurations](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#gpu-configurations)
    * [Model](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#model)
    * [(Prior) Distributions](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#prior-distributions)
      * [Uniform](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#uniform)
      * [Gaussian](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#gaussian)
      * [Truncated Gaussian](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#truncated-gaussian)
      * [Preset](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#preset)
    * [Other Distributions](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#other-distributions)
  * [Models](https://altar.readthedocs.io/en/cuda/cuda/Models.html)
    * [Static Slip Inversion](https://altar.readthedocs.io/en/cuda/cuda/Static.html)
      * [Static Source model](https://altar.readthedocs.io/en/cuda/cuda/Static.html#static-source-model)
      * [Input](https://altar.readthedocs.io/en/cuda/cuda/Static.html#input)
      * [Configurations](https://altar.readthedocs.io/en/cuda/cuda/Static.html#configurations)
      * [Output](https://altar.readthedocs.io/en/cuda/cuda/Static.html#output)
      * [Moment Distribution](https://altar.readthedocs.io/en/cuda/cuda/Static.html#moment-distribution)
      * [Forward Model Application](https://altar.readthedocs.io/en/cuda/cuda/Static.html#forward-model-application)
      * [Utilities](https://altar.readthedocs.io/en/cuda/cuda/Static.html#utilities)
    * [Static Slip Inversion with Cp: Epistemic Uncertainties](https://altar.readthedocs.io/en/cuda/cuda/cp.html)
    * [Kinematic Slip Inversion](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html)
      * [Kinematic Source Model](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#kinematic-source-model)
      * [Joint Kinematic-Static Inversion](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#joint-kinematic-static-inversion)
      * [Configurations (Kinematic Model only)](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#configurations-kinematic-model-only)
      * [Configurations (Joint inversion)](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#configurations-joint-inversion)
      * [Examples](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#examples)
      * [Forward Model Application (new version)](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#forward-model-application-new-version)
      * [Forward Model Application (old version)](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#forward-model-application-old-version)




---

## Source: https://altar.readthedocs.io/en/cuda/cuda/Preface.html

# Preface[](https://altar.readthedocs.io/en/cuda/cuda/Preface.html#preface "Link to this heading")
  * [AlTar 2.0 Release](https://altar.readthedocs.io/en/cuda/README.html)
    * [About AlTar](https://altar.readthedocs.io/en/cuda/README.html#about-altar)
    * [Guides](https://altar.readthedocs.io/en/cuda/README.html#guides)
    * [Tutorials](https://altar.readthedocs.io/en/cuda/README.html#tutorials)
    * [Support](https://altar.readthedocs.io/en/cuda/README.html#support)
    * [Copyright](https://altar.readthedocs.io/en/cuda/README.html#copyright)
  * [Bayesian Inference for Inverse Problems](https://altar.readthedocs.io/en/cuda/cuda/Background.html)
    * [Inverse problem](https://altar.readthedocs.io/en/cuda/cuda/Background.html#inverse-problem)
    * [Bayesian approach](https://altar.readthedocs.io/en/cuda/cuda/Background.html#bayesian-approach)
    * [CATPMIP algorithm](https://altar.readthedocs.io/en/cuda/cuda/Background.html#catpmip-algorithm)
    * [References](https://altar.readthedocs.io/en/cuda/cuda/Background.html#references)




---

## Source: https://altar.readthedocs.io/en/cuda/cuda/Pyre.html

# Pyre Basics[](https://altar.readthedocs.io/en/cuda/cuda/Pyre.html#pyre-basics "Link to this heading")
AlTar’s architecture is based on the 
## Protocols and Components[](https://altar.readthedocs.io/en/cuda/cuda/Pyre.html#protocols-and-components "Link to this heading")
Python offers a modular programming design with modules and classes while pyre extends the functionalities of python classes to facilitate their integrations and configurations, through _components_.
A prescribed functionality is defined as a _protocol_ (similar to an abstract class). For example, various distributions are used in AlTar, mainly serving as prior distributions. We first define a protocol,
```
# the protocol
class Distribution(altar.protocol, family="altar.distributions"):
    """
    The protocol that all AlTar probability distributions must satisfy
    """

    # required behaviors
    @altar.provides
    def initialize(self, **kwds):
        """
        Initialize with the given random number generator
        """

    # model support
    @altar.provides
    def initializeSample(self, theta):
        """
        Fill my portion of {theta} with initial random values from my distribution.
        """

    @altar.provides
    def priorLikelihood(self, theta, prior):
        """
        Fill my portion of {prior} with the likelihoods of the samples in {theta}
        """

    @altar.provides
    def verify(self, theta, mask):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """

    ... ...

    # framework hooks
    @classmethod
    def pyre_default(cls):
        """
        Supply a default implementation
        """
        # use the uniform distribution
        from .Uniform import Uniform as default
        # and return it
        return default

```

where `@altar.provides` decorator specifies the behaviors (methods) that its implementations must define. An implementation, for example, the Uniform distribution, is defined as a _component_ ,
```
class Uniform(altar.component, family="altar.distributions.uniform"):
    """
    The uniform probability distribution
    """

    # user configurable state
    parameters = altar.properties.int()
    parameters.doc = "the number of model parameters that i take care of"

    offset = altar.properties.int(default=0)
    offset.doc = "the starting point of my parameters in the overall model state"

    # user configurable state
    support = altar.properties.array(default=(0,1))
    support.doc = "the support interval of the prior distribution"


    # protocol obligations
    @altar.export
    def initialize(self, rng):
        """
        Initialize with the given random number generator
        """
        # set up my pdf
        self.pdf = altar.pdf.uniform(rng=rng.rng, support=self.support)
        # all done
        return self

    @altar.export
    def initializeSample(self, theta):
        """
        Fill my portion of {theta} with initial random values from my distribution.
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=theta)
        # fill it with random numbers from my initializer
        self.pdf.matrix(matrix=θ)
        # and return
        return self


    @altar.export
    def verify(self, theta, mask):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """

        ... ...

        # all done; return the rejection map
        return mask

    ... ...

    # private data
    pdf = None # the pdf implementation

```

where the required behaviors specific to the Uniform distribution are defined. Besides behaviors, a component may also include attributes such as
>   * Properties, configurable parameters in terms of basic Python data type, such as an integer [defined as `altar.properties.int()`;
>   * Sub-Components, configurable attributes in terms of components;
>   * Non-configurable attributes, regular Python objects such as static properties or objects determined at runtime, e.g., the _pdf_ function in `Distribution`.
> 

Components are building blocks of AlTar. For example, Distribution can be used as the prior distribution in a Bayesian model,
```
class Bayesian(altar.component, family="altar.models.bayesian", implements=model):
"""
The base class of AlTar models that are compatible with Bayesian explorations
"""

    prior = altar.distributions.distribution()
    prior.doc = "the prior distribution"

    ... ...

```

Here, prior is a configurable component, for which users can specify at runtime by `model.prior=uniform` or any other distributions implementing the Distribution-protocol. Since the protocol defines the uniform distribution as its default implementation, if none is specified at runtime, the uniform distribution is used by default.
Note also that _components_ are abstract methods and can be only be instantiated by an AlTar application instance. If you create a component instance in a Python shell, it will not behave as a regular Python class.
## Pyre Config Format (`.pfg`)[](https://altar.readthedocs.io/en/cuda/cuda/Pyre.html#pyre-config-format-pfg "Link to this heading")
Configurations of properties and components can be passed to the program as command line arguments, or more conveniently, by a configuration file. Three types of configuration files are supported by `.pml` (XML-style), `.cfg` (an INI-style format used in AlTar 1.1), and `.pfg` (YAML/JSON-style). We recommend `.pfg` for its human-readable data serialization format.
An example of `.pfg` file is provided in [QuickStart](https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html#quickstart), for the linear model.
Some basic rules of `.pfg` format are
  * Whitespace indentation is used for denoting structure, or hierarchy; however, tab characters are not allowed.
  * Hierarchy of components can be specified by indentation, or by explicit full path, or by a combination of partial path with indentation. For example, these three configurations are equivalent:
> ```
; method 1: all by indentation
linear:
    job:
        tasks = 1
        gpus = 0
        chains = 2**10

; method 2: all by full path
linear.job.tasks = 1
linear.job.gpus = 0
linear.job.chains = 2**10

; method 3: combination with partial path and indentation
linear:
    job.tasks = 1
    job.gpus = 0
    job.chains = 2**10

```

  * If a component is not specified or listed in the configuration file, a default value/implementation specified in the Python program will be used instead.
  * Strings such as paths, names, don’t need quotation marks.




---

## Source: https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html

# AlTar Framework[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#altar-framework "Link to this heading")
## Application[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#application "Link to this heading")
_See API Reference:_ [`altar.shells.application`](https://altar.readthedocs.io/en/cuda/api/altar/shells/index.html#altar.shells.application "altar.shells.application")
An AlTar application is the _root_ component which integrates all the components in AlTar. The main components of an AlTar application are as follows. 

_class_ Application(_altar.application_ , _family="altar.shells.application"_)[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#Application "Link to this definition") 
     

controller = altar.bayesian.controller() 
     

Value: 
    
altar.bayesian.Annealer (default) 

Description: 
    
the MCMC simulation processor, _Annealer_ for CATMIP algorithm; 

model = altar.models.model() 
     

Value: 
    
altar.models.linear(), altar.models.cuda.static(), … 

Description: 
    
performs the forward modelling and computes the Bayesian probability densities: the prior, the data likelihood and the posterior; 

job = altar.simulations.run() 
     

Description: 
    
manages the simulation size and job deployment; 

rng = altar.simulations.rng() 
     

Description: 
    
the random number generator shared by all processes; 

monitors = altar.properties.dict(schema=altar.simulations.monitor()) 
     

Description: 
    
a collection of event handlers, such as reporter, profiler.
An AlTar application executes the simulation by defining a required `main` entry point:
```
@altar.export
def main(self, *args, **kwds):
    """
    The main entry point
    """

    # initialize various components
    self.job.initialize(application=self)
    self.rng.initialize()
    self.controller.initialize(application=self)
    self.model = self.model.initialize(application=self)
    # sample the posterior distribution
    return self.model.posterior(application=self)

```

which initializes different components and invokes the `model.posterior` to perform the MCMC sampling.
An AlTar application is also the engager of the pyre framework, as inherited from `pyre.application` or `pyre.plexus`, which performs
>   * registering all protocols and components in a database;
>   * reading/loading configuration files;
>   * instantiating all components into _regular_ Python objects;
>   * invoking the proper shell (MPI, SLURM) to deploy the job.
> 

## Controller/Annealer[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#controller-annealer "Link to this heading")
_See API Reference:_ [`altar.bayesian.Annealer`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Annealer/index.html#module-altar.bayesian.Annealer "altar.bayesian.Annealer")
A Bayesian controller uses an annealing schedule and MCMC to approximate the posterior distribution of a model. The current Annealer uses exclusively the CATMIP algorithm (more controllers implementing other algorithms will be added to a future release). It includes the following configurable components
>   * `sampler = altar.bayesian.sampler()`, the MCMC sampler. The default is a `Metropolis` sampler with fixed chain length based on Metropolis-Hastings algorithm. Another sampler implemented is `AdaptiveMetropolis` which targets a fixed acceptance ratio and varies the chain length targeting a fixed effective sample size.
>   * `scheduler = altar.bayesian.scheduler()`, the generator of the annealing schedule. The default and currently only implemented scheduler is based on the Coefficient of Variance (COV) of the data likelihood densities.
>   * `dispatcher = altar.simulations.dispatcher(default=Notifier)`, currently only serves the purpose of profiling.
>   * `archiver = altar.simulations.archiver(default=Recorder)`, the archiver of simulation state. The default recorder saves the simulation state to HDF5 files.
> 

and another component determined at runtime from the job configurations,
>   * `worker` (AnnealingMethod): as AlTar simulations can be performed with either single thread or multiple threads, on CPUs or GPUs, the Controller uses different workers where various deployment-dependence procedures are differentiated. For example, the multiple thread processor, `MPIAnnealing` needs to include additional procedures to collect/distribute samples for all threads.
> 

The Annealer’s behaviors include
>   * `deduceAnnealingMethod` which uses the job configuration to determine the worker.
>   * `posterior` which defines the MCMC procedures with an annealing schedule.
> 

### Worker/AnnealingMethod[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#worker-annealingmethod "Link to this heading")
_See API Reference:_ [`altar.bayesian.AnnealingMethod`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/AnnealingMethod/index.html#module-altar.bayesian.AnnealingMethod "altar.bayesian.AnnealingMethod")
A worker(AnnealingMethod) defines each procedure of annealing which may be platform dependent. There are three types of workers:
>   * `SequentialAnnealing`, a single thread CPU processor
>   * `CUDAAnnealing`, a single thread GPU processor
>   * `MPIAnnealing`, a multiple thread processor which uses SequentialAnnealing(CPU) or CUDAAnnealing(GPU) as slave workers.
> 

> Each worker also keeps a set of simulation state data, such as `beta` (the inverse temperature), `theta` (the random samples), `prior/data/posterior` (the Bayesian densities), in an object `CoolingStep`.
Worker is not directly user configurable; it is determined automatically by the job configuration.
### Sampler[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#sampler "Link to this heading")
_See API Reference:_ [`altar.bayesian.Sampler`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Sampler/index.html#module-altar.bayesian.Sampler "altar.bayesian.Sampler")
Starting with Ns number of chains/samples (processed in parallel), a sampler performs MC updates of the samples pursuant to a given distribution for several steps (length of the chain). For finite β, the target distribution is the transient distribution Pm(θ|d)=P(θ)P(d|θ)βm, while the sampling serves as a burn-in process. When β=1 is reached, the sampler samples the posterior distribution P(θ|d).
The default sampler is a CPU `Metroplis` sampler. To use other samplers, e.g., for CUDA simulations, users need to specify it in the controller block of the configuration file
```
ApplicationInstance:
    controller:
        sampler = altar.cuda.bayesian.metropolis
        sampler:
            ; sampler attributes
            ... ...

```

#### Metropolis[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#metropolis "Link to this heading")
_See API Reference:_ [`altar.bayesian.Metropolis`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Metropolis/index.html#module-altar.bayesian.Metropolis "altar.bayesian.Metropolis")
**Algorithm**
  * New samples are proposed with a Gaussian kernel,
> θ′=θ+αδδ∼N(0,Σ)
> where Σ is the (weighted) covariance matrix of starting samples (from the previous β step), and α is a scaling factor adjusting the jump distance. In CATMIP, α is adjusted by the acceptance rate (from the previous β-step):
> α=acceptanceWeight∗acceptanceRate+rejectionWeight
> Since the acceptance rate varies between 0 and 1, `rejectionWeight` and `rejectionWeight+acceptanceWeight` offer as the lower and upper limits of the scaling factor α.
  * Decide whether to accept the proposed samples with the Metropolis–Hastings algorithm.
  * Repeat the MC updates for a fixed Nc-number of times.


**Configurable attributes** 

scaling: 
    
float, the initial value of α, default=0.1 

acceptanceWeight, rejectionWeight: 
    
float, ratios to adjust the value of α during the run, defaults=8/9, 1/9 

steps: 
    
integer, the MC update steps in each β-step (the length of each chain), configured by `job.steps`.
**Configuration examples**
```
ApplicationInstanceName:
    controller:
        sampler = altar.bayesian.metropolis
        sampler:
            scaling = 0.2
            acceptanceWeight = 0.1
            rejectionWeight = 0.9
    ; the length of chains
    job.steps = 2**12

```

#### Metropolis (CUDA Version)[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#metropolis-cuda-version "Link to this heading")
_See API Reference:_ [`altar.cuda.bayesian.cudaMetropolis`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/bayesian/cudaMetropolis/index.html#module-altar.cuda.bayesian.cudaMetropolis "altar.cuda.bayesian.cudaMetropolis")
The CUDA version follows the same procedure as above, but includes more control on the scaling factor.
**Configurable attributes** 

scaling: 
    
float; the initial value of α; default=0.1 

acceptanceWeight, rejectionWeight: 
    
float; ratios to adjust the value of α during the run, defaults=8/9, 1/9 

useFixedScaling: 
    
bool; if `True`, the initial `scaling` will be used for all β-steps; default= `False` 

scalingMin, scalingMax: 
    
float; the min/max values of the scaling factor; default=0.01, 1 

steps: 
    
integer, the MC update steps in each β-step (the length of each chain), configured by `job.steps`.
**Configuration examples**
```
ApplicationInstanceName:
    controller:
        sampler = altar.cuda.bayesian.metropolis
        sampler:
            scaling = 0.2
            acceptanceWeight = 0.99
            rejectionWeight = 0.01
            useFixedScaling = False
            scalingMin = 0.1
            scalingMax = 0.5
    ; the length of chains
    job.steps = 2**12

```

In this example, `scaling` is set to 0.2 in the beginning. During the run, `scaling = acceptanceWeight*R + rejectionWeight`, where R is the acceptance rate from the previous β-step, or `scaling` ∈[0.01,1]. `scalingMin` and `scalingMax` further adjust the range as [0.1, 0.5].
#### AdaptiveMetropolis[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#adaptivemetropolis "Link to this heading")
_See API Reference:_ [`altar.cuda.bayesian.cudaAdaptiveMetropolis`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/bayesian/cudaAdaptiveMetropolis/index.html#module-altar.cuda.bayesian.cudaAdaptiveMetropolis "altar.cuda.bayesian.cudaAdaptiveMetropolis") (for CUDA only)
**Algorithm**
In an AdaptiveMetropolis sampler, there are a few variations from the Metropolis sampler,
  1. After a certain number of MC updates, `corr_check_steps`, the correlation between the current samples and the initial samples are computed. If the correlation is smaller than a threshold value, `target_correlation`, or the samples become sufficiently de-correlated, we can stop MC updates for the current β-step. A `max_mc_steps` sets the maximum number of MC updates if the correlation threshold value cannot be achieved.
  2. The scaling factor α targets an optimal acceptance rate, `target_acceptance_rate`, with a feedback function
> αj+1=αjexp⁡[−gain∗(acceptanceRatej−target_acceptance_rate)]
> where j labels the β-step. The initial value is set as
> α0=scaling/parameters
> It is shown that an optimal value for α is 2.38/d, where d is the dimension of parameter space, or `parameters`.
  3. (_New in 2.0.2_) Sometimes, it is useful to use more MC steps when β is small, or _vice versa_. We introduce a new `max_mc_steps_stage2` to be used for the maximum MC steps when β > `beta_step2`.


**Configurable Attributes** 

scaling: 
    
float, default=2.38, initial scaling factor for Gaussian proposal, to be normalized by the square root of the number of parameters 

parameters: 
    
integer, default=1, the total number of parameters in simulation: since the controller is initialized before the model, users need to manually provide this information to the sampler (we will try to eliminate this requirement in the next update). 

scaling_min, scaling_max: 
    
float, default=(0.01, 1), the minimum and maximum values allowed for the scaling factor 

target_acceptance_rate: 
    
float, default=0.234, the targeted acceptance rate 

gain: 
    
float, default=None (determined by `target_acceptance_rate`), the feedback gain constant 

max_mc_steps: 
    
integer, default=100000, the maximum Monte-Carlo steps for one beta step 

min_mc_steps: 
    
integer, default=1000, the minimum Monte-Carlo steps for one beta step 

beta_stage2: 
    
float, default=0.1, the start beta value to use another maximum MC steps 

max_mc_steps_stage2: 
    
integer, default=None (to be set the same as `max_mc_steps`), the maximum Monte-Carlo steps for one beta step when `beta` > `beta_stage2` 

corr_check_steps: 
    
integer, default=1000, the Monte-Carlo steps to compute and check the correlation 

target_correlation: 
    
float, default=0.6, the threshold of correlation to stop the chain updates
**Configuration examples**
```
ApplicationInstanceName:
    controller:
        sampler = altar.cuda.bayesian.adaptivemetropolis  ; only for CUDA
        sampler:
            scaling = 2.38
            parameters = 399 ; number of parameters in model
            min_mc_steps = 3000
            max_mc_steps = 10000
            corr_check_steps = 1000
            target_correlation = 0.6
            beta_stage2 = 0.1
            max_mc_steps_stage2 = 5000

```

In this example, the initial scaling factor is set to `scaling` / sqrt(`parameters`) or 2.38/399=0.12 (you can also set `scaling` directly to 0.12, and use the default value 1 for `parameters`). After `min_mc_steps` (3,000), the correlation between current samples and initial samples are computed every `corr_check_steps` (1,000). If the correlation is less than `target_correlation` (0.6), the MC update stops and the simulation proceeds to next β step. If the `target_correlation` cannot be achieved by a certain number of steps, `max_mc_steps` (10,000) for β<= `beta_stage2` (0.1) while `max_mc_steps_stage2` (5,000) for β> `beta_stage2` (0.1), the MC update is forced to stop and the simulation proceeds to next β step.
### Scheduler[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#scheduler "Link to this heading")
A scheduler regulates how β increases between different β-steps. The default (and currently the only option) is COV Scheduler.
#### COV Scheduler[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#cov-scheduler "Link to this heading")
_See API Reference:_ [`altar.bayesian.COV`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/COV/index.html#module-altar.bayesian.COV "altar.bayesian.COV")
**Algorithm**
The COV (Coefficient of Variation) scheduler targets the effective sample size from resampling between different transient distributions,
Pm(θ|d)=P(θ)P(d|θ)βm,
At the m-stage, samples θm,k are generated with the target equilibrium distribution Pm(θ|d), where k=1,2,…,Ns and Ns is the total number of samples (chains). At the m+1-stage, the sampling targets Pm+1(θ|d) as the new equilibrium distribution. To sample a distribution with samples generated from another distribution is called as importance sampling, with the importance weight
w(θm,k)=Pm+1(θm,k|d)Pm(θ|d)=P(d|θm,k)βm+1−βm
while the effective sample size (ESS) from the resampling is associated with the COV of w(θm,k),
ESS=Ns1+COV(w),COV(w)=w¯σw¯=1Ns∑kw(θm,k)σ=1Ns−1∑k[w(θm,k)−w¯]2.
In COV Scheduler, we choose a βm+1 so that COV is of order unity, e.g., COV=1, or ESS=Ns/2.
**Configurable Attributes** 

target: 
    
float, default=1.0, the target value for COV 

solver: 
    
`altar.bayesian.solver()`, values= `grid` (default), `brent`; the δβ solver based on the grid search algorithm (grid) or the Brent minimizer (brent) 

check_positive_definiteness: 
    
bool, values = `True` (default), `False`; whether to check the positive definiteness of Σ matrix and condition it accordingly 

min_eigenvalue_ratio: 
    
float, default=0.001; if checking the positive definiteness, the minimal eigenvalue to be set as a ratio of the maximum eigenvalue
**Configuration examples**
```
ApplicationInstanceName:
    controller:
        scheduler: ; default is COV
            target = 2.0
            solver = brent
            check_positive_definiteness = False

```

### Archiver (Output)[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#archiver-output "Link to this heading")
_See API Reference:_ [`altar.simulations.Archiver`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/Archiver/index.html#module-altar.simulations.Archiver "altar.simulations.Archiver")
The Archiver saves progress information. The default is an H5Recorder which saves the Bayesian statistical data to HDF5 files.
#### H5Recorder[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#h5recorder "Link to this heading")
_See API Reference:_ [`altar.bayesian.H5Recorder`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/H5Recorder/index.html#module-altar.bayesian.H5Recorder "altar.bayesian.H5Recorder")
H5Recorder saves the random samples and their Bayesian probability densities from each β-step to an HDF5 file, `output_dir/step_nnn.h5`.
```
+---------- step_nnn.h5 ------
├── Annealer ; annealing data
|   ├── beta ; the beta value
|   └── covariance ; the covariance matrix for Gaussian proposal
├── Bayesian ; the Bayesian probabilities
|   ├── prior ; (log) prior probability for all samples, vector (samples)
|   ├── likelihood ; (log) data likelihood for all samples
|   └── posterior ; (log) posterior probability for all samples
└── ParameterSets ;  samples
    └── theta ; samples of model parameters, 2d array (samples, parameters)

```

If you use a list of parameter sets, e.g., in Static Inversion, {`strike_slip`, `dip_slip`, `insar_ramp`}, their names will be used for their data sets,
```
└── ParameterSets ;
    └── strike_slip ; samples of strike slips, 2d array (samples, number of patches)
    └── dip_slip ; samples of dip slips, 2d array (samples, number of patches)
    └── insar_ramp ; samples of insar ramp parameters to be fitted, 2d array (samples, number of ramp parameters)

```

H5Recorder also records the MCMC statistics from each β-step to a file `output_dir/BetaStatistics.txt`. An example for the linear model is as follows
```
iteration, beta, scaling, (accepted, invalid, rejected)
0, 0, 0.3, (0, 0, 0)
1, 0.00015000000000000001, 0.5925347222222223, (2773, 0, 2347)
2, 0.00037996549999999997, 0.31666666666666665, (1184, 0, 3936)
3, 0.00069984391104, 0.5828125, (2717, 0, 2403)
4, 0.0011195499765973632, 0.3454861111111111, (1350, 0, 3770)
5, 0.0016789230286104687, 0.5607638888888888, (2590, 0, 2530)
6, 0.0025175127332664362, 0.3590277777777778, (1428, 0, 3692)
7, 0.0037843154920951883, 0.5447916666666667, (2498, 0, 2622)
8, 0.005577503724209416, 0.3751736111111111, (1521, 0, 3599)
9, 0.008163002214526472, 0.5303819444444444, (2415, 0, 2705)
10, 0.012229533905446913, 0.37986111111111115, (1548, 0, 3572)
11, 0.017958602608795324, 0.5154513888888889, (2329, 0, 2791)
12, 0.026207750346881442, 0.38125, (1556, 0, 3564)
13, 0.03808801579264949, 0.5, (2240, 0, 2880)
14, 0.05444051952417445, 0.40243055555555557, (1678, 0, 3442)
15, 0.07760672679583216, 0.4793402777777778, (2121, 0, 2999)
16, 0.11173527790438637, 0.42065972222222225, (1783, 0, 3337)
17, 0.16503116123012318, 0.4588541666666667, (2003, 0, 3117)
18, 0.24518816975203137, 0.41944444444444445, (1776, 0, 3344)
19, 0.3470877668355071, 0.46753472222222225, (2053, 0, 3067)
20, 0.5037867027949854, 0.40625, (1700, 0, 3420)
21, 0.7176546338903467, 0.47465277777777776, (2094, 0, 3026)
22, 1.0, 0.41770833333333335, (1766, 0, 3354)

```

which shows how β evolves from β-step iterations, as well the scaling parameter :math:`alpha`, and the MC acceptance (accepted/rejected = proposals accepted/rejected by Metropolis-Hastings algorithm, invalid = proposals rejected due to being out of range for ranged priors).
**Configurable Attributes** 

output_dir: 
    
path(string), default=”results”; the directory to save results 

output_freq: 
    
integer, default=1; the frequency to write step data to files, e.g., if you only want to save data for every 3 β-steps, you may choose `output_freq=3`. The final β-step, i.e., β=1 will be always saved as `step_final.h5`.
**Configuration examples**
```
ApplicationInstanceName:
    controller:
        archiver: ; default is H5Recorder
            output_dir = results/static_chain_1024
            output_freq = 3

```

## Job[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#job "Link to this heading")
_See API Reference:_ [`altar.simulations.Job`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/Job/index.html#module-altar.simulations.Job "altar.simulations.Job")
The `job` component in an AlTar application controls the size of the simulation as well as its deployment to different platforms.
### Configurable Attributes[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#configurable-attributes "Link to this heading")
```
name = altar.properties.str(default="sample")
name.doc = "the name of the job; used as a stem for making filenames, etc."

mode = altar.properties.str()
mode = "the programming model"

hosts = altar.properties.int(default=1)
hosts.doc = "the number of hosts to run on"

tasks = altar.properties.int(default=1)
tasks.doc = "the number of tasks per host"

gpus = altar.properties.int(default=0)
gpus.doc = "the number of gpus per task"

gpuprecision = altar.properties.str(default="float64")
gpuprecision.doc = "the precision of gpu computations"
gpuprecision.validators = altar.constraints.isMember("float64", "float32")

gpuids = altar.properties.list(schema=altar.properties.int())
gpuids.default = None
gpuids.doc = "the list of gpu ids for parallel jobs"

chains = altar.properties.int(default=1)
chains.doc = "the number of chains per worker"

steps = altar.properties.int(default=20)
steps.doc = 'the length of each Markov chain'

tolerance = altar.properties.float(default=1.0e-3)
tolerance.doc = "convergence tolerance for β->1.0"

```

### Simulation Size[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#simulation-size "Link to this heading")
For a single thread simulation, the job size is determined by the number of `chains` (per thread), which are processed as a batch. More chains offer better phase space exploration. But the number of chains may be limited by the computer memory size (CPU or GPU). Since the memory usage also depends on the number of parameters and the type of model, users are encouraged to try some chain sizes at first (stop the simulation after one or two beta steps) to determine an optimal setting of `chains` for their computer system.
Large-scale simulations can be distributed to multiple threads. Distributing the total number of chains from one thread (sequentially) to multiple threads (in parallel) may also reduce the computation time (wall time). The number of threads are controlled by two parameters `hosts` - the number of hosts (nodes), and `tasks` - the number of threads per host. The total number of chains is therefore `hosts*tasks*chains`.
The multi-threading in AlTar is achieved by MPI. An AlTar application is capable of deploying itself automatically to multiple MPI threads in one or more computers/nodes so that users don’t need to run `mpirun`, `qsub`, or `sbatch` explicitly.
The Metropolis sampler uses `job.steps` to control the number of MC updates in each β-step. (_It might be better to move this setting directly to sampler_). This procedure serves as a burn-in to equilibrate samples from one distribution with βm to another with βm+1. Larger `steps` allow more equilibration but are not required in CATMIP: the β-increment (or the total number of β-steps) will be adjusted.
### Single Thread Configuration[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#single-thread-configuration "Link to this heading")
The default job setting is to run AlTar with one CPU thread; you don’t need to provide any settings in the configuration file. However, if you prefer to keep `hosts` and `tasks` entries to be modified later, set them to be 1 explicitly.
```
ApplicationInstanceName:
    job:
        hosts = 1 ; number of hosts/nodes
        tasks = 1 ; number of threads per host
        chains = 2**12 ; number of Markov chains per thread.
        steps = 2**10 ; length of the Markov chain

```

### Multiple Threads on One Computer[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#multiple-threads-on-one-computer "Link to this heading")
To run a multi-threaded simulation on a single computer, you need to adjust the `tasks` setting, as well as to specify a `mpi.shells.mpirun` shell instead,
From the configuration file
```
ApplicationInstanceName:
    job:
        tasks = 8

    shell = mpi.shells.mpirun

; additional configurations for mpi shell, if needed
mpi.shells.mpirun # altar.plexus.shell:
    extra = -mca btl self,tcp

```

or from the command line
```
$ AlTarApp --job.tasks=8 --shell=mpi.shells.mpirun

```

If you use an MPI package not installed under the system directory, you may need to provide its configuration to AlTar/pyre framework. For example, for OpenMPI installed with Anaconda, the following additional configurations are required
```
mpi.shells.mpirun:
  ; mpi implementation
  mpi = openmpi#mpi_conda

; mpi configuration
pyre.externals.mpi.openmpi # mpi_conda:
  version = 3.0
  launcher = mpirun
  prefix = /opt/anaconda3
  bindir = {mpi_conda.prefix}/bin
  incdir = {mpi_conda.prefix}/include
  libdir = {mpi_conda.prefix}/lib

```

Since the MPI package information is common for running all jobs on a computer, you may choose to save the above configuration to an `mpi.pfg` file, either under `${HOME}/.pyre` directory (searchable by all AlTar/pyre applications), or under the work directory with the AlTar application configurable file, e.g., `linear.pfg`.
Decide the max number of tasks/threads on a computer from the number of physical cores, not from the total (virtual) threads. Hyperthreading may increase the number of available threads, but it might not increase performance for compute-intensive models, e.g., with matrix multiplications.
### Multiple Threads Across Several Computers[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#multiple-threads-across-several-computers "Link to this heading")
If a batch scheduler is not required, you may still use the `mpi.shells.mpirun` shell with the additional `hostfile` configuration. A hostfile is a simple text file with hosts/nodes specified, e.g., `my_hostfile`
```
# This is an example hostfile.  Comments begin with #
192.168.1.101 slots=16
192.168.1.102 slots=16
192.168.1.103 slots=16

```

To use the hostfile with `mpi.shells.mpirun` shell, e.g., with 4 hosts and 8 threads per host,
```
ApplicationInstanceName:
    job:
        hosts = 4
        tasks = 8
    shell = mpi.shells.mpirun
    shell:
        hostfile = my_hostfile

```

Or from the commmand line,
```
$ AlTarApp --job.hosts=4 --job.tasks=8 --shell=mpi.shells.mpirun --shell.hostfile=my_hostfile

```

If a batch scheduler is available, e.g., `mpi.shells.slurm` shell instead. An example configuration is as follows
```
ApplicationInstanceName:
    job:
        hosts = 4
        tasks = 8
    shell = mpi.shells.slurm

; for parallel runs
mpi.shells.slurm :
    submit = True ; if True, submit the job for execution, if not, a slurm script is generated
    queue = gpu ; the name of the queue

```

Or from the command line
```
$ AlTarApp --job.hosts=4 --job.tasks=8 --shell=mpi.shells.slurm --shell.queue=gpu

```

If your Slurm Manager requires additional configurations, you can use `submit=False`, modify the generated Slurm script, and use `sbatch` to submit the job.
### GPU Configurations[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#gpu-configurations "Link to this heading")
The GPU support in AlTar is implemented with 
**To choose GPU or CPU** If you plan to run AlTar simulations on GPU, you may enable it by
```
ApplicationInstanceName:
    job.gpus = 1 ; number of GPU per task,  0 = use gpu

```

AlTar also checks the availability of `cuda` modules (software) and compatible CUDA devices (hardware). If either is unavailable, AlTar enforces `job.gpus = 0`, or using CPU instead.
Currently, the `cuda` modules are not fully integrated with cpu modules. You may need to check whether the model has a cuda implementation, and also select explicitly some cuda components, for example, for the Static Inversion,
```
slipmodel: ; the Application Instance Name

    model = altar.models.seismic.cuda.static
    ; define parametersets
    psets:
        strikeslip = altar.cuda.models.parameterset
        dipslip = altar.cuda.models.parameterset

        strikeslip:
            count = {slipmodel.model.patches}
            prior = altar.cuda.distributions.gaussian
            prior.mean = 0
            prior.sigma = 0.5
        ... ...

    controller:
        sampler = altar.cuda.bayesian.metropolis

    ... ...

```

_In the next release, we will try to merge cuda modules with cpu modules so that a single ``jobs.gpus`` flag can switch between CPU and GPU computations._
**To use multiple GPUs** GPUs (device) rely on CPU (host) for job dispatching and data copies from/to GPU memories. AlTar runs on one GPU per (CPU) thread, therefore, the number of GPUs used for simulation is tuned by the number of threads per host, `job.tasks`, and/or the number of hosts, `job.hosts`. `job.gpus` is always 1 for GPU simulations.
To deploy the simulation to 8 GPUs in one computer/node, the configuration is
```
ApplicationInstanceName:
    job.hosts = 1 ; number of hosts
    job.tasks = 8 ; number of threads per host
    job.gpus = 1 ; number of gpus per thread

```

To deploy the simulation to 4 nodes with 8 GPUs per node, the configuration is
```
ApplicationInstanceName:
    job.hosts = 4 ; number of hosts
    job.tasks = 8 ; number of threads per host
    job.gpus = 1 ; number of gpus per thread

```

For a computer/node with multiple GPUs, the job is distributed sequentially to the first available GPUs (`gpuids` = 0, 1,2, 3 …). If you plan to assign the job to specific GPUs, you can use `job.gpuids` to specify them,
```
ApplicationInstanceName:
    job.hosts = 4 ; number of hosts
    job.tasks = 2 ; number of threads per host
    job.gpus = 1 ; number of gpus per thread
    job.gpuids = [2, 3]

```

Another method is to use the environmental variable `CUDA_VISIBLE_DEVICES`. For example,
```
# bash
$ export CUDA_VISIBLE_DEVICES=2,3
# csh/tcsh
$ setenv CUDA_VISIBLE_DEVICES 2,3

```

which makes only GPU2 and GPU3 visible for applications, appearing as `gpuids=[0, 1]`. With this method, you don’t need to set `gpuids`.
**To select the precision** AlTar supports both single and double precision GPU computations (the CPU computation is always double precision). However, most NVIDIA gaming cards only have single precision processors. In our tests on Tesla cards, single precision computations run twice faster than double precisions. If you want to choose between single and double precisons, you may use the `job.gpuprecison` flag, such as
```
ApplicationInstanceName:
    job:
        hosts = 1 ; number of hosts
        tasks = 2 ; number of threads per host
        gpus = 1 ; number of gpus per thread
        gpuprecision = float32 ; double(float64) or single(float32) precision for gpu computations

```

## Model[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#model "Link to this heading")
The Model component in AlTar Framework defines the forward problem, and computes the data likelihood accordingly. Each model needs an implementation for a given inverse problem. See [Models](https://altar.readthedocs.io/en/cuda/cuda/Models.html#models) for details guides on implemented models.
Users may also develop their own models, following the guide in [Bayesian Model](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#develop-bayesian-model).
## (Prior) Distributions[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#prior-distributions "Link to this heading")
There are several probability distributions defined in AlTar, serving as prior distributions.
For a prior distribution [`altar.distributions.distribution`](https://altar.readthedocs.io/en/cuda/api/altar/distributions/index.html#altar.distributions.distribution "altar.distributions.distribution"), the following methods are required
```
# model support
@altar.provides
def initializeSample(self, theta):
    """
    Fill my portion of {theta} with initial random values from my distribution.
    """

@altar.provides
def priorLikelihood(self, theta, prior):
    """
    Fill my portion of {prior} with the likelihoods of the samples in {theta}
    """

@altar.provides
def verify(self, theta, mask):
    """
    Check whether my portion of the samples in {theta} are consistent with my constraints, and
    update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
    """

```

In AlTar, the logarithmic values of the Bayesian probability densities are processed. Therefore, in addition to `priorLikelihood`, a `verify` method is needed for certain ranged distributions to check whether the proposed samples fall outside the range.
For a parameter to be sampled, the distributions for generating the initial samples and computing the prior probability densities during the simulation may be different. For example, in static inversion of earthquakes, you may want to use a Moment Scale distribution to generate (strike or dip) slips whose sum is consistent with a given moment magnitude scale, while during the simulation, while a simple uniform distribution for `priorLikelihood` and `verify`.
Please note that AlTar processes samples in batch, where θ is a 2d array `shape=(samples, parameters)`. Also, in a specific Model, there may be different parameter sets which observe different prior distributions. Therefore, the methods in a prior distribution are responsible for its own portion of parameters (selected columns of θ) and for a batched samples (rows of θ).
### Uniform[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#uniform "Link to this heading")
The probability density function (PDF) for a uniform distribution is
f(x;a,b)=1b−a,for x∈[a,b]=0,otherwise
where [a,b] is the support or range. 

Example: 

```
prior = uniform
prior:
    support = (0, 1)

```

### Gaussian[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#gaussian "Link to this heading")
The PDF for the Gaussian (normal) distribution is defined as
f(x;μ,σ)=12πσe−(x−μ)22σ2,
where μ and σ2 are mean (center) and variance, respectively. 

Example: 

```
prior = gaussian
prior:
    mean = 0
    sigma = 2

```

### Truncated Gaussian[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#truncated-gaussian "Link to this heading")
The  

Example: 

(Currently only implemented in CUDA).
```
prior = altar.cuda.distributions.tgaussian
prior:
    support = (-1, 1)
    mean = 0
    sigma = 2

```

### Preset[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#preset "Link to this heading")
The `Preset` distribution is used to load initial samples from pre-calculated ones. Therefore, it only serves as a preparation (`prep`) distribution. The currently support input format is HDF5, as the default output for AlTar simulation results. 

Example: 

(Currently only implemented in CUDA).
For example, in the earthquake (seismic) inversion, we have samples of `strikeslip` generated from the static inversion and would like to load them for the kinematic inversion,
```
prep = altar.cuda.distributions.preset ; load preset samples
prep.input_file = theta_cascaded.h5 ; file name
prep.dataset = ParameterSets/strikeslip ; dataset name in h5

```

## Other Distributions[](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#other-distributions "Link to this heading")
More prior distributions can be easily added. You may follow the existing distributions as examples. Or please write to us so that we add them to the package.


---

## Source: https://altar.readthedocs.io/en/cuda/py-modindex.html

# Python Module Index
[**a**](https://altar.readthedocs.io/en/cuda/py-modindex.html#cap-a)
|  |   
---|---|---  
|  **a** |   
![-](https://altar.readthedocs.io/en/cuda/_static/minus.png) |  [`altar`](https://altar.readthedocs.io/en/cuda/api/altar/index.html#module-altar) |   
|  [`altar.actions`](https://altar.readthedocs.io/en/cuda/api/altar/actions/index.html#module-altar.actions) |   
|  [`altar.actions.About`](https://altar.readthedocs.io/en/cuda/api/altar/actions/About/index.html#module-altar.actions.About) |   
|  [`altar.actions.Forward`](https://altar.readthedocs.io/en/cuda/api/altar/actions/Forward/index.html#module-altar.actions.Forward) |   
|  [`altar.actions.Sample`](https://altar.readthedocs.io/en/cuda/api/altar/actions/Sample/index.html#module-altar.actions.Sample) |   
|  [`altar.bayesian`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/index.html#module-altar.bayesian) |   
|  [`altar.bayesian.Annealer`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Annealer/index.html#module-altar.bayesian.Annealer) |   
|  [`altar.bayesian.AnnealingMethod`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/AnnealingMethod/index.html#module-altar.bayesian.AnnealingMethod) |   
|  [`altar.bayesian.Brent`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Brent/index.html#module-altar.bayesian.Brent) |   
|  [`altar.bayesian.Controller`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Controller/index.html#module-altar.bayesian.Controller) |   
|  [`altar.bayesian.CoolingStep`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/CoolingStep/index.html#module-altar.bayesian.CoolingStep) |   
|  [`altar.bayesian.COV`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/COV/index.html#module-altar.bayesian.COV) |   
|  [`altar.bayesian.CUDAAnnealing`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/CUDAAnnealing/index.html#module-altar.bayesian.CUDAAnnealing) |   
|  [`altar.bayesian.Grid`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Grid/index.html#module-altar.bayesian.Grid) |   
|  [`altar.bayesian.H5Recorder`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/H5Recorder/index.html#module-altar.bayesian.H5Recorder) |   
|  [`altar.bayesian.Metropolis`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Metropolis/index.html#module-altar.bayesian.Metropolis) |   
|  [`altar.bayesian.MPIAnnealing`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/MPIAnnealing/index.html#module-altar.bayesian.MPIAnnealing) |   
|  [`altar.bayesian.Notifier`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Notifier/index.html#module-altar.bayesian.Notifier) |   
|  [`altar.bayesian.Profiler`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Profiler/index.html#module-altar.bayesian.Profiler) |   
|  [`altar.bayesian.Recorder`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Recorder/index.html#module-altar.bayesian.Recorder) |   
|  [`altar.bayesian.Sampler`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Sampler/index.html#module-altar.bayesian.Sampler) |   
|  [`altar.bayesian.Scheduler`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Scheduler/index.html#module-altar.bayesian.Scheduler) |   
|  [`altar.bayesian.SequentialAnnealing`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/SequentialAnnealing/index.html#module-altar.bayesian.SequentialAnnealing) |   
|  [`altar.bayesian.Solver`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Solver/index.html#module-altar.bayesian.Solver) |   
|  [`altar.bayesian.ThreadedAnnealing`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/ThreadedAnnealing/index.html#module-altar.bayesian.ThreadedAnnealing) |   
|  [`altar.cuda`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/index.html#module-altar.cuda) |   
|  [`altar.cuda.bayesian`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/bayesian/index.html#module-altar.cuda.bayesian) |   
|  [`altar.cuda.bayesian.cudaAdaptiveMetropolis`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/bayesian/cudaAdaptiveMetropolis/index.html#module-altar.cuda.bayesian.cudaAdaptiveMetropolis) |   
|  [`altar.cuda.bayesian.cudaCoolingStep`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/bayesian/cudaCoolingStep/index.html#module-altar.cuda.bayesian.cudaCoolingStep) |   
|  [`altar.cuda.bayesian.cudaMetropolis`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/bayesian/cudaMetropolis/index.html#module-altar.cuda.bayesian.cudaMetropolis) |   
|  [`altar.cuda.bayesian.cudaMetropolisVaryingSteps`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/bayesian/cudaMetropolisVaryingSteps/index.html#module-altar.cuda.bayesian.cudaMetropolisVaryingSteps) |   
|  [`altar.cuda.data`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/data/index.html#module-altar.cuda.data) |   
|  [`altar.cuda.data.cudaDataL2`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/data/cudaDataL2/index.html#module-altar.cuda.data.cudaDataL2) |   
|  [`altar.cuda.distributions`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/distributions/index.html#module-altar.cuda.distributions) |   
|  [`altar.cuda.distributions.cudaDistribution`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/distributions/cudaDistribution/index.html#module-altar.cuda.distributions.cudaDistribution) |   
|  [`altar.cuda.distributions.cudaGaussian`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/distributions/cudaGaussian/index.html#module-altar.cuda.distributions.cudaGaussian) |   
|  [`altar.cuda.distributions.cudaPreset`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/distributions/cudaPreset/index.html#module-altar.cuda.distributions.cudaPreset) |   
|  [`altar.cuda.distributions.cudaTGaussian`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/distributions/cudaTGaussian/index.html#module-altar.cuda.distributions.cudaTGaussian) |   
|  [`altar.cuda.distributions.cudaUniform`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/distributions/cudaUniform/index.html#module-altar.cuda.distributions.cudaUniform) |   
|  [`altar.cuda.ext`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/ext/index.html#module-altar.cuda.ext) |   
|  [`altar.cuda.models`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/models/index.html#module-altar.cuda.models) |   
|  [`altar.cuda.models.cudaBayesian`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/models/cudaBayesian/index.html#module-altar.cuda.models.cudaBayesian) |   
|  [`altar.cuda.models.cudaBayesianEnsemble`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/models/cudaBayesianEnsemble/index.html#module-altar.cuda.models.cudaBayesianEnsemble) |   
|  [`altar.cuda.models.cudaParameterEnsemble`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/models/cudaParameterEnsemble/index.html#module-altar.cuda.models.cudaParameterEnsemble) |   
|  [`altar.cuda.models.cudaParameterSet`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/models/cudaParameterSet/index.html#module-altar.cuda.models.cudaParameterSet) |   
|  [`altar.cuda.norms`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/norms/index.html#module-altar.cuda.norms) |   
|  [`altar.cuda.norms.cudaL2`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/norms/cudaL2/index.html#module-altar.cuda.norms.cudaL2) |   
|  [`altar.data`](https://altar.readthedocs.io/en/cuda/api/altar/data/index.html#module-altar.data) |   
|  [`altar.data.DataL2`](https://altar.readthedocs.io/en/cuda/api/altar/data/DataL2/index.html#module-altar.data.DataL2) |   
|  [`altar.data.DataObs`](https://altar.readthedocs.io/en/cuda/api/altar/data/DataObs/index.html#module-altar.data.DataObs) |   
|  [`altar.distributions`](https://altar.readthedocs.io/en/cuda/api/altar/distributions/index.html#module-altar.distributions) |   
|  [`altar.distributions.Base`](https://altar.readthedocs.io/en/cuda/api/altar/distributions/Base/index.html#module-altar.distributions.Base) |   
|  [`altar.distributions.Distribution`](https://altar.readthedocs.io/en/cuda/api/altar/distributions/Distribution/index.html#module-altar.distributions.Distribution) |   
|  [`altar.distributions.Gaussian`](https://altar.readthedocs.io/en/cuda/api/altar/distributions/Gaussian/index.html#module-altar.distributions.Gaussian) |   
|  [`altar.distributions.Uniform`](https://altar.readthedocs.io/en/cuda/api/altar/distributions/Uniform/index.html#module-altar.distributions.Uniform) |   
|  [`altar.distributions.UnitGaussian`](https://altar.readthedocs.io/en/cuda/api/altar/distributions/UnitGaussian/index.html#module-altar.distributions.UnitGaussian) |   
|  [`altar.ext`](https://altar.readthedocs.io/en/cuda/api/altar/ext/index.html#module-altar.ext) |   
|  [`altar.meta`](https://altar.readthedocs.io/en/cuda/api/altar/meta/index.html#module-altar.meta) |   
|  [`altar.models`](https://altar.readthedocs.io/en/cuda/api/altar/models/index.html#module-altar.models) |   
|  [`altar.models.Bayesian`](https://altar.readthedocs.io/en/cuda/api/altar/models/Bayesian/index.html#module-altar.models.Bayesian) |   
|  [`altar.models.BayesianL2`](https://altar.readthedocs.io/en/cuda/api/altar/models/BayesianL2/index.html#module-altar.models.BayesianL2) |   
|  [`altar.models.cdm`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/index.html#module-altar.models.cdm) |   
|  [`altar.models.cdm.CDM`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/CDM/index.html#module-altar.models.cdm.CDM) |   
|  [`altar.models.cdm.CUDA`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/CUDA/index.html#module-altar.models.cdm.CUDA) |   
|  [`altar.models.cdm.Data`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/Data/index.html#module-altar.models.cdm.Data) |   
|  [`altar.models.cdm.ext`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/ext/index.html#module-altar.models.cdm.ext) |   
|  [`altar.models.cdm.Fast`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/Fast/index.html#module-altar.models.cdm.Fast) |   
|  [`altar.models.cdm.libcdm`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/libcdm/index.html#module-altar.models.cdm.libcdm) |   
|  [`altar.models.cdm.meta`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/meta/index.html#module-altar.models.cdm.meta) |   
|  [`altar.models.cdm.Native`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/Native/index.html#module-altar.models.cdm.Native) |   
|  [`altar.models.cdm.Source`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/Source/index.html#module-altar.models.cdm.Source) |   
|  [`altar.models.Contiguous`](https://altar.readthedocs.io/en/cuda/api/altar/models/Contiguous/index.html#module-altar.models.Contiguous) |   
|  [`altar.models.cudalinear`](https://altar.readthedocs.io/en/cuda/api/altar/models/cudalinear/index.html#module-altar.models.cudalinear) |   
|  [`altar.models.cudalinear.cudaLinear`](https://altar.readthedocs.io/en/cuda/api/altar/models/cudalinear/cudaLinear/index.html#module-altar.models.cudalinear.cudaLinear) |   
|  [`altar.models.cudalinear.meta`](https://altar.readthedocs.io/en/cuda/api/altar/models/cudalinear/meta/index.html#module-altar.models.cudalinear.meta) |   
|  [`altar.models.emhp`](https://altar.readthedocs.io/en/cuda/api/altar/models/emhp/index.html#module-altar.models.emhp) |   
|  [`altar.models.emhp.EMHP`](https://altar.readthedocs.io/en/cuda/api/altar/models/emhp/EMHP/index.html#module-altar.models.emhp.EMHP) |   
|  [`altar.models.emhp.meta`](https://altar.readthedocs.io/en/cuda/api/altar/models/emhp/meta/index.html#module-altar.models.emhp.meta) |   
|  [`altar.models.Ensemble`](https://altar.readthedocs.io/en/cuda/api/altar/models/Ensemble/index.html#module-altar.models.Ensemble) |   
|  [`altar.models.gaussian`](https://altar.readthedocs.io/en/cuda/api/altar/models/gaussian/index.html#module-altar.models.gaussian) |   
|  [`altar.models.gaussian.ext`](https://altar.readthedocs.io/en/cuda/api/altar/models/gaussian/ext/index.html#module-altar.models.gaussian.ext) |   
|  [`altar.models.gaussian.Gaussian`](https://altar.readthedocs.io/en/cuda/api/altar/models/gaussian/Gaussian/index.html#module-altar.models.gaussian.Gaussian) |   
|  [`altar.models.gaussian.meta`](https://altar.readthedocs.io/en/cuda/api/altar/models/gaussian/meta/index.html#module-altar.models.gaussian.meta) |   
|  [`altar.models.linear`](https://altar.readthedocs.io/en/cuda/api/altar/models/linear/index.html#module-altar.models.linear) |   
|  [`altar.models.linear.Linear`](https://altar.readthedocs.io/en/cuda/api/altar/models/linear/Linear/index.html#module-altar.models.linear.Linear) |   
|  [`altar.models.linear.meta`](https://altar.readthedocs.io/en/cuda/api/altar/models/linear/meta/index.html#module-altar.models.linear.meta) |   
|  [`altar.models.Model`](https://altar.readthedocs.io/en/cuda/api/altar/models/Model/index.html#module-altar.models.Model) |   
|  [`altar.models.mogi`](https://altar.readthedocs.io/en/cuda/api/altar/models/mogi/index.html#module-altar.models.mogi) |   
|  [`altar.models.mogi.CUDA`](https://altar.readthedocs.io/en/cuda/api/altar/models/mogi/CUDA/index.html#module-altar.models.mogi.CUDA) |   
|  [`altar.models.mogi.Data`](https://altar.readthedocs.io/en/cuda/api/altar/models/mogi/Data/index.html#module-altar.models.mogi.Data) |   
|  [`altar.models.mogi.ext`](https://altar.readthedocs.io/en/cuda/api/altar/models/mogi/ext/index.html#module-altar.models.mogi.ext) |   
|  [`altar.models.mogi.Fast`](https://altar.readthedocs.io/en/cuda/api/altar/models/mogi/Fast/index.html#module-altar.models.mogi.Fast) |   
|  [`altar.models.mogi.meta`](https://altar.readthedocs.io/en/cuda/api/altar/models/mogi/meta/index.html#module-altar.models.mogi.meta) |   
|  [`altar.models.mogi.Mogi`](https://altar.readthedocs.io/en/cuda/api/altar/models/mogi/Mogi/index.html#module-altar.models.mogi.Mogi) |   
|  [`altar.models.mogi.Native`](https://altar.readthedocs.io/en/cuda/api/altar/models/mogi/Native/index.html#module-altar.models.mogi.Native) |   
|  [`altar.models.mogi.Source`](https://altar.readthedocs.io/en/cuda/api/altar/models/mogi/Source/index.html#module-altar.models.mogi.Source) |   
|  [`altar.models.Null`](https://altar.readthedocs.io/en/cuda/api/altar/models/Null/index.html#module-altar.models.Null) |   
|  [`altar.models.ParameterSet`](https://altar.readthedocs.io/en/cuda/api/altar/models/ParameterSet/index.html#module-altar.models.ParameterSet) |   
|  [`altar.models.regression`](https://altar.readthedocs.io/en/cuda/api/altar/models/regression/index.html#module-altar.models.regression) |   
|  [`altar.models.regression.Linear`](https://altar.readthedocs.io/en/cuda/api/altar/models/regression/Linear/index.html#module-altar.models.regression.Linear) |   
|  [`altar.models.regression.meta`](https://altar.readthedocs.io/en/cuda/api/altar/models/regression/meta/index.html#module-altar.models.regression.meta) |   
|  [`altar.models.seismic`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/index.html#module-altar.models.seismic) |   
|  [`altar.models.seismic.actions`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/actions/index.html#module-altar.models.seismic.actions) |   
|  [`altar.models.seismic.actions.About`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/actions/About/index.html#module-altar.models.seismic.actions.About) |   
|  [`altar.models.seismic.actions.Forward`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/actions/Forward/index.html#module-altar.models.seismic.actions.Forward) |   
|  [`altar.models.seismic.actions.Sample`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/actions/Sample/index.html#module-altar.models.seismic.actions.Sample) |   
|  [`altar.models.seismic.cuda`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/cuda/index.html#module-altar.models.seismic.cuda) |   
|  [`altar.models.seismic.cuda.cudaCascaded`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/cuda/cudaCascaded/index.html#module-altar.models.seismic.cuda.cudaCascaded) |   
|  [`altar.models.seismic.cuda.cudaKinematicG`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/cuda/cudaKinematicG/index.html#module-altar.models.seismic.cuda.cudaKinematicG) |   
|  [`altar.models.seismic.cuda.cudaKinematicGCp`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/cuda/cudaKinematicGCp/index.html#module-altar.models.seismic.cuda.cudaKinematicGCp) |   
|  [`altar.models.seismic.cuda.cudaMoment`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/cuda/cudaMoment/index.html#module-altar.models.seismic.cuda.cudaMoment) |   
|  [`altar.models.seismic.cuda.cudaStatic`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/cuda/cudaStatic/index.html#module-altar.models.seismic.cuda.cudaStatic) |   
|  [`altar.models.seismic.cuda.cudaStaticCp`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/cuda/cudaStaticCp/index.html#module-altar.models.seismic.cuda.cudaStaticCp) |   
|  [`altar.models.seismic.ext`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/ext/index.html#module-altar.models.seismic.ext) |   
|  [`altar.models.seismic.meta`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/meta/index.html#module-altar.models.seismic.meta) |   
|  [`altar.models.seismic.Moment`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/Moment/index.html#module-altar.models.seismic.Moment) |   
|  [`altar.models.seismic.shells`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/shells/index.html#module-altar.models.seismic.shells) |   
|  [`altar.models.seismic.shells.Action`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/shells/Action/index.html#module-altar.models.seismic.shells.Action) |   
|  [`altar.models.seismic.shells.cudaApplication`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/shells/cudaApplication/index.html#module-altar.models.seismic.shells.cudaApplication) |   
|  [`altar.models.seismic.shells.Seismic`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/shells/Seismic/index.html#module-altar.models.seismic.shells.Seismic) |   
|  [`altar.models.seismic.Static`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/Static/index.html#module-altar.models.seismic.Static) |   
|  [`altar.models.seismic.StaticCp`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/StaticCp/index.html#module-altar.models.seismic.StaticCp) |   
|  [`altar.models.sir`](https://altar.readthedocs.io/en/cuda/api/altar/models/sir/index.html#module-altar.models.sir) |   
|  [`altar.models.sir.meta`](https://altar.readthedocs.io/en/cuda/api/altar/models/sir/meta/index.html#module-altar.models.sir.meta) |   
|  [`altar.models.sir.SIR`](https://altar.readthedocs.io/en/cuda/api/altar/models/sir/SIR/index.html#module-altar.models.sir.SIR) |   
|  [`altar.norms`](https://altar.readthedocs.io/en/cuda/api/altar/norms/index.html#module-altar.norms) |   
|  [`altar.norms.L2`](https://altar.readthedocs.io/en/cuda/api/altar/norms/L2/index.html#module-altar.norms.L2) |   
|  [`altar.norms.Norm`](https://altar.readthedocs.io/en/cuda/api/altar/norms/Norm/index.html#module-altar.norms.Norm) |   
|  [`altar.shells`](https://altar.readthedocs.io/en/cuda/api/altar/shells/index.html#module-altar.shells) |   
|  [`altar.shells.Action`](https://altar.readthedocs.io/en/cuda/api/altar/shells/Action/index.html#module-altar.shells.Action) |   
|  [`altar.shells.AlTar`](https://altar.readthedocs.io/en/cuda/api/altar/shells/AlTar/index.html#module-altar.shells.AlTar) |   
|  [`altar.shells.Application`](https://altar.readthedocs.io/en/cuda/api/altar/shells/Application/index.html#module-altar.shells.Application) |   
|  [`altar.shells.cudaAlTar`](https://altar.readthedocs.io/en/cuda/api/altar/shells/cudaAlTar/index.html#module-altar.shells.cudaAlTar) |   
|  [`altar.shells.cudaApplication`](https://altar.readthedocs.io/en/cuda/api/altar/shells/cudaApplication/index.html#module-altar.shells.cudaApplication) |   
|  [`altar.simulations`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/index.html#module-altar.simulations) |   
|  [`altar.simulations.Archiver`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/Archiver/index.html#module-altar.simulations.Archiver) |   
|  [`altar.simulations.Dispatcher`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/Dispatcher/index.html#module-altar.simulations.Dispatcher) |   
|  [`altar.simulations.GSLRNG`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/GSLRNG/index.html#module-altar.simulations.GSLRNG) |   
|  [`altar.simulations.Job`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/Job/index.html#module-altar.simulations.Job) |   
|  [`altar.simulations.Monitor`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/Monitor/index.html#module-altar.simulations.Monitor) |   
|  [`altar.simulations.Recorder`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/Recorder/index.html#module-altar.simulations.Recorder) |   
|  [`altar.simulations.Reporter`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/Reporter/index.html#module-altar.simulations.Reporter) |   
|  [`altar.simulations.RNG`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/RNG/index.html#module-altar.simulations.RNG) |   
|  [`altar.simulations.Run`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/Run/index.html#module-altar.simulations.Run) | 


---

## Source: https://altar.readthedocs.io/en/cuda/cuda/Static.html

# Static Slip Inversion[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#static-slip-inversion "Link to this heading")
## Static Source model[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#static-source-model "Link to this heading")
The finite fault earthquake source models infer the spatial distribution and temporal evolution of coseismic slips at fault plane(s) from the observed surface or subsurface displacements.
In the static source model, the spatial distributions of coseismic slips are determined. We model the fault plane(s) as a set of patches; each patch treated as a point source with two orthogonal displacement components, slips along the strike and dip directions. Each slip on the fault plane can be translated into surface deformation, e.g., by the Okada model, which is derived from a Green’s function solution to the elastic half space problem. The observed surface deformation at a given location is the linear combination due to (strike/dip) slips of all patches.
Note
For the static inversion, the patches can be of any shape or area as long as each patch can be treated a single point source. Note that the kinematic inversion currently only supports a rectangle fault divided into ndd×nas square patches. If you plan for a joint static-kinematic inversion, you need to run the static inversion on the square patches as well.
Therefore, the forward model can be expressed as a linear model
dpred=Gθ.
where θ (also denoted as m in geophysics literatures) is a vector with Nparam=2Npatch components, representing the slips along the strike and dip directions for Npatch patches; d is a vector of observed surface deformations (may include vertical and east, north horizontal components) at different locations, with Nobs observations; and G is a 2Npatch×Nobs matrix, pre-calculated Green’s functions connecting a slip source to a deformation component at an observation location.
A generalized forward model could also include other linear parameters, for example, the InSAR ramp parameters (a,b,c), used to fit the spurious ramp-like displacement fields a+bx+cy from InSAR interferograms, where x and y are the locations of the data in local Cartesian coordinates.
## Input[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#input "Link to this heading")
To run the static inversion, in addition to an AlTar configuration file (see next section), you are required to prepare three input files, 

data.txt: 
    
the observed data with Nobs observations, a vector in one row or one column. 

cd.txt: 
    
covariance of data representing measurement errors/uncertainties, prepared in a Nobs×Nobs matrix (could have off-diagonal terms for correlated errors). A constant could be used (with _cd_std_ option) if all data are uncorrelated and have the same variance. Epistemic uncertainties such as elastic heterogeneity and fault geometry uncertainties may also be included, see [Static Slip Inversion with Cp](https://altar.readthedocs.io/en/cuda/cuda/StaticCp.html#static-inversion-cp) for more details. 

green.txt: 
    
the pre-calculated Green’s functions, prepared in a Nobs×Nparam matrix, with Nparam as the leading dimension, i.e., data arranged in row-major as
```
G[Obs1][Param1] G[Obs1][Param2] ... G[Obs1][ParamNparam]
G[Obs2][Param2] G[Obs2][Param2] ... G[Obs2][ParamNparam]
... ...
G[ObsNobs][Param1] G[ObsNobs][Param2] ... G[ObsNobs][ParamNparam]

```

You may use any other names for these files but need to specify them in the configuration file. These files are arranged under the same directory which can be specified by `case` setting in the configuration file.
In addition to plain text inputs with suffix `.txt`, AlTar 2 also accepts binary data, including
>   * Raw binary format with suffix `.bin` or `.dat`. The precision should be consistent with the numerical precision specified by `job.gpuprecision`. Its size should match the size of the corresponding quantity. For example, the Green’s function has in total Nobs×Nparam elements. They will be reshaped by the program.
>   * HDF5 files with suffix `.h5`. HDF5 files are preferred because they include the data shape and precision in metadata. AlTar2 performs the reshaping and precision conversions if necessary.
> 

AlTar2 recognizes the data format automatically by the file suffixes.
There are some software packages which can pre-calculate the Green’s functions and/or prepare the input files for AlTar, e.g., 
## Configurations[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#configurations "Link to this heading")
A configuration file for the static inversion appears as
```
; application instance name
slipmodel:

    ; model to be sampled
    model = altar.models.seismic.cuda.static
    model:

        ; the name of the test case, also as the directory for input files
        case = 9patch

        ; number of patches
        patches = 9

        ; green's function (observations, parameters)
        green = static.gf.h5

        ; data observations
        dataobs = altar.cuda.data.datal2
        dataobs:
            observations = 108
            data_file = static.data.h5
            cd_file = static.Cd.h5
            ; or use a constant cd
            ; cd_std = 1e-4

        ; list of parameter sets
        ; the order should be consistent with the green's function
        psets_list = [strikeslip, dipslip, ramp]

        ; define parameter sets
        psets:
            strikeslip = altar.cuda.models.parameterset
            dipslip = altar.cuda.models.parameterset
            ramp = altar.cuda.models.parameterset

            strikeslip:
                count = {slipmodel.model.patches}
                prior = altar.cuda.distributions.gaussian
                prior.mean = 0
                prior.sigma = 0.5

            dipslip:
                count = {slipmodel.model.patches}
                prep = altar.models.seismic.cuda.moment
                prep:
                    Mw_mean = 7.3
                    Mw_sigma = 0.2
                    Mu = [30] ; in GPa
                    area = [400] ; patch area in km^2
                prior = altar.cuda.distributions.uniform
                prior.support = (-0.5, 20)

            ramp:
                count = 3
                prior = altar.cuda.distribution.uniform
                prior.support = (-1, 1)

    controller:
        sampler = altar.cuda.bayesian.metropolis
        archiver:
            output_dir = results/static ; output directory
            output_freq = 3 ; output frequency in beta steps


    ; run configuration
    job:
        tasks = 1 ; number of tasks per host
        gpus = 1  ; number of gpus per task
        gpuprecision = float32 ; double(float64) or single(float32) precision for gpu computations
        ;gpuids = [0] ; a list gpu device ids for tasks on each host, default range(job.gpus)
        chains = 2**10 ; number of chains per task
        steps = 1000 ; MC burn-in steps for each beta step

```

We explain each section below.
### Application Instance Name[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#application-instance-name "Link to this heading")
We use a shell command `slipmodel` for all seismic slip models, including static and kinematic inversions, which uses `slipmodel` as the application instance name. Therefore, please use `slipmodel` as the root in the configuration file. By the `slipmodel.pfg` in current path. If you name your configuration file as `slipmodel.pfg`, you may simply run
```
$ slipmodel

```

to invoke simulations for any slip models. If you want to name the configuration file as something else, e.g., `static.pfg`, `static_mpi.pfg`, or `Nepal_static.pfg`, you may specify the configuration file from the command line by the `--config` option,
```
$ slipmodel  --config=static.pfg

```

### Model[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#model "Link to this heading")
For static inversion, you need to specify `model = altar.models.seismic.cuda.static` (or the CPU version, `model=altar.models.seismic.static`).
**Model Attributes** 

case: 
    
the directory where all input files are located; 

patches: 
    
the number of patches, or point sources; 

green: 
    
the file name for the Green’s functions, as prepared from the instructions above; 

dataobs: 
    
a component to process the data observations and calculate the data likelihood with L2 norm, with details provided in [Data Observations](https://altar.readthedocs.io/en/cuda/cuda/Static.html#data-observations); 

psets_lists: 
    
a list of parameter sets, the order will be used for many purposes, e.g., enforcing the order of parameters in θ; 

psets: 
    
components to describe the parameter sets, with details provided in [Parameter Sets](https://altar.readthedocs.io/en/cuda/cuda/Static.html#static-parameter-sets).
### Data Observations[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#data-observations "Link to this heading")
The observed data are handled by a component named `dataobs`. We use exclusively the L2 norm for the likelihood computation because it accommodates the uncertainty quantification from the data covariance matrix (Cd). Therefore,
```
dataobs = altar.cuda.data.datal2
dataobs:
    observations = 108
    data_file = static.data.h5
    cd_file = static.Cd.h5
    ; cd_std = 1e-2

```

For the data observations with the data covariance matrix `datal2`, the following attributes are required 

observations: 
    
the number of data observations 

data_file: 
    
the name of the file containing the data observations, a vector with `observations` elements 

cd_file: 
    
the name of the file containing the data covariance, a matrix with `observations x observations` elements 

cd_std: 
    
if the data covariance has only constant diagonal elements, you may use this option instead of `cd_file`.
### Parameter Sets[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#parameter-sets "Link to this heading")
A parameter set is a group of parameters which share the same prior distributions and are arranged continuously in θ. In static model, we use the following parameter sets `strikeslip`, `dipslip`, and optionally, `ramp` (you may use any other names for the parameter sets as long as they are intuitive).
The order of the parameter sets in θ is enforced by the attribute `psets_list`,
```
psets_list = [strikeslip, dipslip, ramp]

```

If the number of patches is 9 and there are 3 InSAR ramp parameters for one set of interferograms. The 21 parameters in θ are (0-8), strike slips of 9 patches; (9-17), dip slipd of 9 patches; and (18-20), ramp parameters. The order of the parameter sets can be varied, but has to be consistent with that in the Green’s function matrix.
For each parameter set, you need to define it as a parameterset, e.g., `strikeslip = altar.cuda.models.parameterset` or (`strikeslip = contiguous` for CPU models). Its attributes include 

count: 
    
the number of parameters in this set. `{slipmodel.model.patches}` is another way to assign values with pre-defined parameters; 

prior: 
    
the prior distribution to initialize random samples in the beginning, and compute prior probabilities during the sampling process. See [(Prior) Distributions](https://altar.readthedocs.io/en/cuda/cuda/Priors.html#prior-distributions) for choices of priors.
  * `prep` (optional), a distribution to initialize samples only. If it is defined, `prep` distribution is used to initialize samples while `prior` distribution is used for computing prior probabilities. If `prep` is not defined, `prior` distribution is used for both.


**Example**
For dip-slip faults, you may use a `uniform` prior to limit the range of dip slips while using a [Moment Distribution](https://altar.readthedocs.io/en/cuda/cuda/Static.html#moment-distribution) to initialize samples so that the moment magnitude is consistent with an estimate scale Mw.
```
dipslip = altar.cuda.models.parameterset
dipslip:
    count = {slipmodel.model.patches}
    prep = altar.models.seismic.cuda.moment
    prep:
        Mw_mean = 7.3 ; mean moment magnitude scale
        Mw_sigma = 0.2 ; sd for moment magnitude scale
        Mu = [30] ; in GPa
        area = [400] ; patch area in km^2
    prior = altar.cuda.distributions.uniform
    prior:
        support = (-0.5, 20)

```

Meanwhile, a Gaussian distribution centered at 0 may be used for strike slips
```
strikeslip = altar.cuda.models.parameterset
strikeslip:
    count = {cudastatic.model.patches}
    prior = altar.cuda.distributions.gaussian
    prior:
        mean = 0
        sigma = 0.5

```

Since the same distribution is also used to initialize samples, no `prep` setting is needed.
For InSAR ramps, either a uniform or a Gaussian prior can be used
```
ramp = altar.cuda.models.parameterset
ramp:
    count = 3
    prior = altar.cuda.distributions.uniform
    prior.support = (-0.5, 0.5)

```

If you prefer to use different priors for different patches, for example, to limit the range of slips far away from the hypocenter, you can further divide the strikeslip/dipslip into several parameter sets, such as
```
psets_list = [strikeslip_p1-3, strikeslip_p4-6, strikeslip_p7-9, ...]

```

and define each parameter set by specifying its count and range.
### Controller[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#controller "Link to this heading")
Please refer to [Controller/Annealer](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#controller) for Bayesian framework configurations. You may use this section to choose and customize the `sampler` - to process MCMC (e.g, `altar.cuda.bayesian.metropolis` or `altar.cuda.bayesian.adaptivemetropolis`), `archiver` - to record the results (default is `H5Recorder`), and `scheduler` - to control the annealing schedule (default is `COV` scheduler).
### Job[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#job "Link to this heading")
Please refer to [Job](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#job-management) on details how to deploy AlTar simulation to different platforms, e.g., single GPU, multiple GPUs on one computer (with mpi), or multiple GPUs distributed in different nodes of a cluster (with mpi and PBS/Slurm scheduler).
## Output[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#output "Link to this heading")
By default, the static inversion simulation results are stored in HDF5 files, see [H5Recorder](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#h5recorder) for more details.
## Moment Distribution[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#moment-distribution "Link to this heading")
For strike (dip) faults, we may want the generated seismic moment from all strike (dip) slips to be consistent with the estimated moment magnitude scale Mw,
Mw=(log⁡M0−9.1)/1.5
M0 is the scalar seismic moment, defined by
M0=μ∑p=1NpatchApDp
where μ is the shear modulus of the rocks involved in the earthquake (in pascals), Ap and Dp are the area (in square meters) and the slip (in meters) of a patch.
A `Moment` distribution is designed to generate random slips for this purpose : it generates a random Mw from a Gaussian distribution Mw∼N(Mwmean,Mwσ), then distributes the corresponding M0/μ to different patches with a Dirichlet distribution (i.e., the sum is a constant), and divides the values by the patch area to obtain slips.
**Example**
The Moment distribution is used as a `prep` distribution to initialize samples in a parameter set,
```
prep = altar.models.seismic.cuda.moment
prep:
    Mw_mean = 7.3 ; mean moment magnitude scale
    Mw_sigma = 0.2 ; sd for moment magnitude scale
    Mu = [30] ; in GPa
    area = [400] ; patch area in km^2

```

**Attributes** 

Mw_mean: 
    
the mean value of the moment magnitude scale. 

Mw_sigma: 
    
the standard deviation of the moment magnitude scale. 

Mu: 
    
the shear modulus of the rocks (in GPa), a list with Npatch elements. If only one element, the same value will be used for all patches. 

area: 
    
the patch area (in square kilometers), also a list with Npatch elements. If the areas for all patches are the same, you may input only one value `area = [400]`. If the areas are different, you may input the list as `area = [400, 300, 200, 300, ...]`, or use a file option below. 

area_patch_file: 
    
a text file as input for patch areas, a vector with Npatch elements, e.g., `area_patch_file = area.txt`. 

slip_sign: 
    
`positive` (default) or `negative`. By default, the moment distribution generates all positive slips, i.e., the displacement is along the positive axis along the dip or strike direction. If the slips are along the opposite direction, use `negative` to generate negative slips.
Note also since `Mu` and `area` appear as products for each patch, you may also use, e.g., `Mu=[1]` and input their products to `area` or `area_patch_file`.
## Forward Model Application[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#forward-model-application "Link to this heading")
When analyzing the results, you may need to run the forward problem once for an obtained mean, median, or MAP model, or a synthetic model, to produce data predictions and compare with data observations. For the static model, it is straightforward: obtain the mean model (vector), read the Green’s function (matrix), and perform a matrix-vector multiplication.
AlTar2 also provides an option to run the forward problem only instead of the full-scale Bayesian simulation, with a slightly modified configuration file. Please follow the steps below.
The first step is to prepare a model file, e.g., `static_mean_model.txt`, including a set of parameters (a vector of Nparam elements), and copy the file to the input directory - the `case` directory. To obtain the mean model from the simulations, see [Utilities](https://altar.readthedocs.io/en/cuda/cuda/Static.html#static-model-utilities) below.
The second step is to modify the configuration file, e.g, `static.pfg`, by adding the following settings under `model` configuration,
```
slipmodel:

    ; model to be sampled
    model = altar.models.seismic.cuda.static
    model:

        ; settings for running forward problem only
        ; forward theta input
        theta_input = static_mean_model.txt
        ; forward output file
        forward_output = static_forward_prediction.h5

        ... ...
        ; the rest is the same

```


theta_input: 
    
the input model file, a text, binary or HDF5 file. 

theta_dataset: 
    
to specify the dataset name in an HDF5 file. 

forward_output: 
    
the output file including the predicted data in HDF5 format.
Note that the forward problem option runs with one GPU (and one thread), please make adjustment to the `job` configuration if necessary,
```
job:
    tasks = 1 ; number of tasks per host
    gpus = 1  ; number of gpus per task
    gpuprecision = float32 ; double(float64) or single(float32) precision for gpu computations
    ;gpuids = [0] ; a list gpu device ids for tasks on each host, default range(job.gpus)
    ... ...

```

The third step is to run a command
```
$ slipmodel.plexus forward --config=static.pfg

```

Check the generated `static_forward_prediction.h5` file for the predicted data from the input model.
Please check the 
`slipmodel.plexus` is a new AlTar application which supports multiple options/workflows how to run the program, a functionality provided by the pyre plexus application class. It currently offers three options,
```
$ slipmodel.plexus about # show application info
$ slipmodel.plexus sample --config=...  # full simulation, equivalent to slipmodel command
$ slipmodel.plexus forward --config=... # forward modeling only
$ slipmodel.plexus  #  call about (as default) to show application info

```

The same configuration file can be used for either options. To run the Bayesian simulations, you may use the same file and run
```
$ slipmodel --config=static.pfg
# or
$ slipmodel.plexus sample --config=static.pfg

```

the settings for the forward problem option have no effect on the simulation workflow and vice versa.
## Utilities[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#utilities "Link to this heading")
We also provide some utilities (in Python) which may be useful to analyze the data or data conversions. Some of scripts may require user modification. Before we can finalize them into standard features, these utilities are currently located at 
### HDF5 Converter tool[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#hdf5-converter-tool "Link to this heading")
We recommend HDF5 as the input format. A conversion tool `H5Converter` is provided if you need to convert any `.txt` or `.bin` files (e.g., from AlTar 1.1) to `.h5`. 

Examples: 

Convert a text file to hdf5
```
H5Converter --inputs=static.gf.txt

```

Convert a binary file to hdf5, additional information such as the precision (default=float32) and the shape (default = 1d vector and will be reshaped to 2d in program if needed) of the output can be added by
```
H5Converter --inputs=kinematicG.gf.bin --precision='float32' --shape=[100,11000]

```

Merge several files into one hdf5, e.g., to prepare the sensitivity kernels for Cp,
```
H5Converter --inputs=[static.kernel.pertL1.txt,static.kernel.pertL2.txt] --output=static.kernel.h5

```

For help on all available options
```
H5Converter --help

```

### Plot histograms of Bayesian probabilities[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#plot-histograms-of-bayesian-probabilities "Link to this heading")
You may want to check the distributions of the (log) prior/likelihood/posterior probabilities, which usually a good indication for the simulation performance. The utility is named `plotBayesian`.
Taking the source code example as an example,
```
# go to the result directory
cd results/static
# run plotBayesian for the final step output,
# which shows the histograms of (log) prior/likelihood/posterior
../../utils/plotBayesian
# to show output from a different beta step
../../utils/plotBayesian --step=step_000.h5
# to change the number of bins for the histogram
../../utils/plotBayesian --bin=20

```

The `plotBayesian` utility generates a `Bayesian_histograms.pdf` file for the plot. You may change the script to show the plot on GUI directly or save it to a different format.
### Compute the mean model[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#compute-the-mean-model "Link to this heading")
`meanModelStatic.py` under the `utils` directory can be used to compute the mean model of the samples in a given beta step. It also serves as a tool to convert AlTar2 H5 output to AlTar-1.1 H5 output.
```
# go the result directory
cd results/static
# run the utility
python3 ../../utils/meanModelStatic.py

```

which prints out the mean values and variances of all parameters, as well as saving them to text files. It also converts `step_final.h5` to AlTar-1.1 H5 format, `step_final_v1.h5`.
For a different beta step output, you need to change `input` and `output` in the script.
The script can also be used for other models, not limited to `static`: you will just need to change `psets_list` to the same as your simulation script.
### Compare the data predications and observations[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#compare-the-data-predications-and-observations "Link to this heading")
`checkDataPrediction.py` reads the data predictions from the forward modeling application and compares them with the input data observations. You may need to change the input files in the script.
### Convert AlTar2 output to AlTar-1.1 output[](https://altar.readthedocs.io/en/cuda/cuda/Static.html#convert-altar2-output-to-altar-1-1-output "Link to this heading")
AlTar2’s H5 output differs from AlTar-1.1 H5 output in rearranging the samples in their parameter sets. You may use the `meanModelStatic.py` utility above to convert them.


---

## Source: https://altar.readthedocs.io/en/cuda/cuda/Models.html

# Models[](https://altar.readthedocs.io/en/cuda/cuda/Models.html#models "Link to this heading")
  * [Static Slip Inversion](https://altar.readthedocs.io/en/cuda/cuda/Static.html)
    * [Static Source model](https://altar.readthedocs.io/en/cuda/cuda/Static.html#static-source-model)
    * [Input](https://altar.readthedocs.io/en/cuda/cuda/Static.html#input)
    * [Configurations](https://altar.readthedocs.io/en/cuda/cuda/Static.html#configurations)
      * [Application Instance Name](https://altar.readthedocs.io/en/cuda/cuda/Static.html#application-instance-name)
      * [Model](https://altar.readthedocs.io/en/cuda/cuda/Static.html#model)
      * [Data Observations](https://altar.readthedocs.io/en/cuda/cuda/Static.html#data-observations)
      * [Parameter Sets](https://altar.readthedocs.io/en/cuda/cuda/Static.html#parameter-sets)
      * [Controller](https://altar.readthedocs.io/en/cuda/cuda/Static.html#controller)
      * [Job](https://altar.readthedocs.io/en/cuda/cuda/Static.html#job)
    * [Output](https://altar.readthedocs.io/en/cuda/cuda/Static.html#output)
    * [Moment Distribution](https://altar.readthedocs.io/en/cuda/cuda/Static.html#moment-distribution)
    * [Forward Model Application](https://altar.readthedocs.io/en/cuda/cuda/Static.html#forward-model-application)
    * [Utilities](https://altar.readthedocs.io/en/cuda/cuda/Static.html#utilities)
      * [HDF5 Converter tool](https://altar.readthedocs.io/en/cuda/cuda/Static.html#hdf5-converter-tool)
      * [Plot histograms of Bayesian probabilities](https://altar.readthedocs.io/en/cuda/cuda/Static.html#plot-histograms-of-bayesian-probabilities)
      * [Compute the mean model](https://altar.readthedocs.io/en/cuda/cuda/Static.html#compute-the-mean-model)
      * [Compare the data predications and observations](https://altar.readthedocs.io/en/cuda/cuda/Static.html#compare-the-data-predications-and-observations)
      * [Convert AlTar2 output to AlTar-1.1 output](https://altar.readthedocs.io/en/cuda/cuda/Static.html#convert-altar2-output-to-altar-1-1-output)
  * [Static Slip Inversion with Cp: Epistemic Uncertainties](https://altar.readthedocs.io/en/cuda/cuda/cp.html)
  * [Kinematic Slip Inversion](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html)
    * [Kinematic Source Model](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#kinematic-source-model)
    * [Joint Kinematic-Static Inversion](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#joint-kinematic-static-inversion)
    * [Configurations (Kinematic Model only)](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#configurations-kinematic-model-only)
      * [An example configuration file](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#an-example-configuration-file)
      * [Parameter Sets](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#parameter-sets)
      * [Input files](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#input-files)
      * [Other attributes](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#other-attributes)
    * [Configurations (Joint inversion)](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#configurations-joint-inversion)
    * [Examples](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#examples)
      * [Cascading Scheme](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#cascading-scheme)
      * [Non-cascading Scheme](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#non-cascading-scheme)
    * [Forward Model Application (new version)](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#forward-model-application-new-version)
    * [Forward Model Application (old version)](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#forward-model-application-old-version)




---

## Source: https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html

# QuickStart[](https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html#quickstart "Link to this heading")
As a quick start, we use the linear model as an example to demonstrate how to run Bayesian MCMC simulations with AlTar. :
  1. Prepare a configuration file, e.g., `linear.pfg`, to specify various parameters and settings;
  2. Prepare input data files required for the model, e.g., for the linear model, the observed data, the data covariance and the Green’s function;
  3. Run a dedicated AlTar application, e.g., `linear` for the linear model;
  4. Collect and analyze the simulation results.


The linear model example demonstrated here comes with the AlTar source package, under the directory 
## Prepare the configuration file[](https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html#prepare-the-configuration-file "Link to this heading")
A configuration file is used to pass various settings to an AlTar application. Here is an example for the linear model,
linear.pfg[](https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html#id3 "Link to this code")
```
 1;
 2; michael a.g. aïvázis
 3; orthologue
 4; (c) 1998-2020 all rights reserved
 5;
 6
 7; the application
 8linear:
 9    ; the model
10    model = linear
11    ; the linear model configurations
12    model:
13        ; the directory for input files
14        case = patch-9
15        ; the number of parameters
16        parameters = 18
17
18        ; the number of observations
19        observations = 108
20        ; the data observations file
21        data = data.txt
22        ; the data covariance file
23        cd = cd.txt
24
25        ; prior distribution for parameters
26        ; prior is used to calculate the prior probability
27        ;    and check ranges during the simulation
28        prior = gaussian
29        ; prior configurations
30        prior:
31            parameters = {linear.model.parameters}
32            center = 0.0
33            sigma = 0.5
34        ; prep is used to initialize the samples in the beginning of the simulation
35        ; it can be different from prior
36        prep = uniform
37        prep:
38            parameters = {linear.model.parameters}
39            support = (-0.5, 0.5)
40
41    ; controller/annealer, use the default CATMIP annealer
42    controller:
43        ; archiver, use the default HDF5 achiver
44        archiver:
45            output_dir = results ; results output directory
46            output_freq = 1 ; output frequency in annealing beta steps
47
48    ; run configuration
49    job.tasks = 1 ; number of tasks per host
50    job.gpus = 0  ; number of gpus per task
51    job.chains = 2**10 ; number of chains per task
52
53; end of file

```

The `.pfg` (pyre config) files follow a human-readable data-serialization format similar to YAML, where the data-structure hierarchy is maintained by whitespace indentation (or by full/partial paths, such as job.tasks, see [Pyre Config Format (.pfg)](https://altar.readthedocs.io/en/cuda/cuda/Pyre.html#pyre-config-format) for more detailed instructions).
The name of the AlTar application (instance), `linear`, is set as the root. Configurable components of an AlTar application include
  * `model`, for model specific configurations, such as the prior distributions of the model parameters, parameters in the forward model, and the data observations;
  * `job`, which configures the size of the simulation, and how the job will be deployed, e.g., single or multiple threads, single machine or multi-node cluster, CPU or GPU;
  * `controller`, for configurations to control the Bayesian MCMC procedure.


Note
If a component is not specified in the configuration, its default value/implementation will be used instead.
Model configurations vary depending on its own forward problem: model-specific instructions are provided in the respective sections of this Guide. Instructions for the main framework, such as `job` and `controller`, can be found in the [AlTar Framework](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#altar-framework) section.
## Prepare input files[](https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html#prepare-input-files "Link to this heading")
While simple configurations can be specified in the configuration file, large sets of data are passed to the AlTar application by data files. Different model may require different categories of data, in different input format.
For the linear model, the data likelihood is computed as
P(d|θ)=1(2π)mdet(Cd)×exp⁡[−12(d−dpred)TCd−1(d−dpred)]
where θ is a vector with n unknown model parameters, d a vector with m observations, the covariance Cd a m×m matrix representing the data uncertainties. The data prediction dpred is given by the forward model
dpred=Gθ.
where the Green’s function G, is a n×m matrix.
The computation requires three users’ input, d, Cd, and G, as plain text files `data.txt`, `cd.txt` and `green.txt`. The location of the files can be specified by the `linear.model.case` parameter in the configuration file, while the file names can be specified by `linear.model.data`, `linear.model.cd` and `linear.model.green`.
## Run an AlTar application[](https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html#run-an-altar-application "Link to this heading")
For each model, we have provided a dedicated command for running AlTar simulations (in fact, you can run `altar` for all models, but it may require some changes to the configuration file). The dedicated command for the linear model is `linear`, which is a Python script as shown below
```
 1#!/usr/bin/env python3
 2# -*- coding: utf-8 -*-
 3#
 4# michael a.g. aïvázis (michael.aivazis@para-sim.com)
 5#
 6# (c) 2010-2020 california institute of technology
 7# (c) 2013-2020 parasim inc
 8# all rights reserved
 9#
10
11# get the package
12import altar
13
14# make a specialized app that uses this model by default
15class Linear(altar.shells.application, family='altar.applications.linear'):
16    """
17    A specialized AlTar application that exercises the Linear model
18    """
19
20    # user configurable state
21    model = altar.models.model(default='linear')
22    model.doc = "the AlTar model to sample"
23
24
25# bootstrap
26if __name__ == "__main__":
27    # build an instance of the default app
28    app = Linear(name="linear")
29    # invoke the main entry point
30    status = app.run()
31    # share
32    raise SystemExit(status)
33
34
35# end of file

```

It defines a `Linear` application class and provides a `main` entry point for execution. The `linear` can be run as any other shell commands, but you do need to run it at the directory where the `linear.pfg` and the `case (patch-9)` directory are located,
```
linear

```

and the simulation begins.
If you would like to use a different script file other than `linear.pfg`
```
linear --config=linear2.pfg

```

or if you would like to pass/change a parameter from command lines, e.g., to increase the number of Markov chains
```
linear --job.chains=2**10

```

More run options will be explained in the [AlTar Framework](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#altar-framework) section.
## Collect and analyze results[](https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html#collect-and-analyze-results "Link to this heading")
AlTar offers several options how to output the simulation results. The default is an HDF5 archiver, which outputs the simulation results from each β-step to HDF5 files located at `results` directory. Data in these HDF5 files, named as `step_nnn.h5`, can be viewed by a HDF Viewer, such as HDFView, HDFCompass.
For each `step_nnn.h5`, the following structures are used
```
+---------- step_nnn.h5 ------
├── Annealer ; annealing data
|   ├── beta ; the beta value
|   └── covariance ; the covariance matrix for Gaussian proposal
├── Bayesian ; the Bayesian probabilities
|   ├── prior ; prior probability for all samples, vector (samples)
|   ├── likelihood ; data likelihood for all samples
|   └── posterior ; posterior probability for all samples
└── ParameterSets ;  samples
    └── theta ; samples of model parameters, 2d array (samples, parameters)

```

```
import h5py
import numpy

```

You may draw a histogram of the posterior to check its distribution. Since the log values of the probabilities are used and saved, the distribution will normally show a lognormal form. You may also do some statistics on the samples, for example, mean and standard deviations. If the posterior assumes a Gaussian distribution, the mean model provides an estimated solution to the linear inverse problem. Some of the data analysis programs are also included with AlTar.
AlTar is a software package developed to perform Bayesian inference to inverse problems with the Markov Chain Monte-Carlo methods. It consists of
  * a main framework which performs the Bayesian MCMC, and controls the job deployment;
  * a model which performs the forward modeling and feeds the data likelihood results to the Bayesian framework. Model implementations for various inverse problems are included. Users may add new models by


An AlTar application integrates a model with the main framework and serves as the main program to run simulations.


---

## Source: https://altar.readthedocs.io/en/cuda/cuda/Overview.html

# Overview[](https://altar.readthedocs.io/en/cuda/cuda/Overview.html#overview "Link to this heading")
AlTar is a software package implementing the [CATPMIP algorithm](https://altar.readthedocs.io/en/cuda/cuda/Background.html#catmip) and other Markov-Chain Monte-Carlo algorithms for Bayesian inference. It adopts the component-based software architecture and the job management system from the 
An AlTar application includes the following root components,
>   * the Bayesian framework (controller/annealer), which performs the MCMC sampling of the Bayesian posterior distribution;
>   * a model, which performs the forward modelling and feeds the resulting data likelihood to the Bayesian framework;
>   * job, which manages the size of a simulation and its deployment to different platforms;
> 

while each component can be configured from the configuration file, e.g., switching between different algorithms/implementations, turning on/off features, and of course, changing the value of a parameter.
To run AlTar simulations, it is usually done by a simple command
```
anAlTarApp --config=anAlTarApp.pfg

```

where `anAlTarApp` is an AlTar application tailed for a given inverse problem, while `anAlTarApp.pfg` is a configuration file you where specify settings for your simulation.
We have provided examples with each implemented inverse problem. You may use them as templates to prepare your own simulation. This Guides aims to provide detailed instructions on how to configure each component, which is organized as follows,
  * [Quickstart](https://altar.readthedocs.io/en/cuda/cuda/QuickStart.html#quickstart): to demonstrate how to run AlTar simulations with a linear inverse problem;
  * [An introduction to pyre](https://altar.readthedocs.io/en/cuda/cuda/Pyre.html#pyre-framework): to offer a brief introduction to components and the `.pfg` configuration file format;
  * [The AlTar framework](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#altar-framework): how to configure the components of the MCMC simulation;
  * [Job Management](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#job-management): how to configure the job deployment to different platforms;
  * Models: to provide instructions on how to prepare data and run simulations for inverse problems in geophysics,
>     * [Static inversion of earthquake source models](https://altar.readthedocs.io/en/cuda/cuda/Static.html#static-inversion);
>     * [Static inversion with Cp (forward model uncertainty)](https://altar.readthedocs.io/en/cuda/cuda/StaticCp.html#static-inversion-cp);
>     * [Kinematic inversion of earthquake source models](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#kinematic-inversion);
>     * Mogi source model for volcanoes (TBD);
>     * Compound dislocation model (CDM) for volcanoes (TBD).


Users may also develop new models for other inverse problems. A [Programming Guide](https://altar.readthedocs.io/en/cuda/cuda/Programming.html#programming-guide) is also provided.


---

## Source: https://altar.readthedocs.io/en/cuda/api/index.html

# API Reference[](https://altar.readthedocs.io/en/cuda/api/index.html#api-reference "Link to this heading")
This page contains auto-generated API reference documentation [[1]](https://altar.readthedocs.io/en/cuda/api/index.html#f1).
  * [`altar`](https://altar.readthedocs.io/en/cuda/api/altar/index.html)
    * [`altar.actions`](https://altar.readthedocs.io/en/cuda/api/altar/actions/index.html)
      * [`altar.actions.About`](https://altar.readthedocs.io/en/cuda/api/altar/actions/About/index.html)
      * [`altar.actions.Forward`](https://altar.readthedocs.io/en/cuda/api/altar/actions/Forward/index.html)
      * [`altar.actions.Sample`](https://altar.readthedocs.io/en/cuda/api/altar/actions/Sample/index.html)
    * [`altar.bayesian`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/index.html)
      * [`altar.bayesian.Annealer`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Annealer/index.html)
      * [`altar.bayesian.AnnealingMethod`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/AnnealingMethod/index.html)
      * [`altar.bayesian.Brent`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Brent/index.html)
      * [`altar.bayesian.COV`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/COV/index.html)
      * [`altar.bayesian.CUDAAnnealing`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/CUDAAnnealing/index.html)
      * [`altar.bayesian.Controller`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Controller/index.html)
      * [`altar.bayesian.CoolingStep`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/CoolingStep/index.html)
      * [`altar.bayesian.Grid`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Grid/index.html)
      * [`altar.bayesian.H5Recorder`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/H5Recorder/index.html)
      * [`altar.bayesian.MPIAnnealing`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/MPIAnnealing/index.html)
      * [`altar.bayesian.Metropolis`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Metropolis/index.html)
      * [`altar.bayesian.Notifier`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Notifier/index.html)
      * [`altar.bayesian.Profiler`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Profiler/index.html)
      * [`altar.bayesian.Recorder`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Recorder/index.html)
      * [`altar.bayesian.Sampler`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Sampler/index.html)
      * [`altar.bayesian.Scheduler`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Scheduler/index.html)
      * [`altar.bayesian.SequentialAnnealing`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/SequentialAnnealing/index.html)
      * [`altar.bayesian.Solver`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/Solver/index.html)
      * [`altar.bayesian.ThreadedAnnealing`](https://altar.readthedocs.io/en/cuda/api/altar/bayesian/ThreadedAnnealing/index.html)
    * [`altar.cuda`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/index.html)
      * [`altar.cuda.bayesian`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/bayesian/index.html)
        * [`altar.cuda.bayesian.cudaAdaptiveMetropolis`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/bayesian/cudaAdaptiveMetropolis/index.html)
        * [`altar.cuda.bayesian.cudaCoolingStep`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/bayesian/cudaCoolingStep/index.html)
        * [`altar.cuda.bayesian.cudaMetropolis`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/bayesian/cudaMetropolis/index.html)
        * [`altar.cuda.bayesian.cudaMetropolisVaryingSteps`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/bayesian/cudaMetropolisVaryingSteps/index.html)
      * [`altar.cuda.data`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/data/index.html)
        * [`altar.cuda.data.cudaDataL2`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/data/cudaDataL2/index.html)
      * [`altar.cuda.distributions`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/distributions/index.html)
        * [`altar.cuda.distributions.cudaDistribution`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/distributions/cudaDistribution/index.html)
        * [`altar.cuda.distributions.cudaGaussian`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/distributions/cudaGaussian/index.html)
        * [`altar.cuda.distributions.cudaPreset`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/distributions/cudaPreset/index.html)
        * [`altar.cuda.distributions.cudaTGaussian`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/distributions/cudaTGaussian/index.html)
        * [`altar.cuda.distributions.cudaUniform`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/distributions/cudaUniform/index.html)
      * [`altar.cuda.ext`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/ext/index.html)
      * [`altar.cuda.models`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/models/index.html)
        * [`altar.cuda.models.cudaBayesian`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/models/cudaBayesian/index.html)
        * [`altar.cuda.models.cudaBayesianEnsemble`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/models/cudaBayesianEnsemble/index.html)
        * [`altar.cuda.models.cudaParameterEnsemble`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/models/cudaParameterEnsemble/index.html)
        * [`altar.cuda.models.cudaParameterSet`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/models/cudaParameterSet/index.html)
      * [`altar.cuda.norms`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/norms/index.html)
        * [`altar.cuda.norms.cudaL2`](https://altar.readthedocs.io/en/cuda/api/altar/cuda/norms/cudaL2/index.html)
    * [`altar.data`](https://altar.readthedocs.io/en/cuda/api/altar/data/index.html)
      * [`altar.data.DataL2`](https://altar.readthedocs.io/en/cuda/api/altar/data/DataL2/index.html)
      * [`altar.data.DataObs`](https://altar.readthedocs.io/en/cuda/api/altar/data/DataObs/index.html)
    * [`altar.distributions`](https://altar.readthedocs.io/en/cuda/api/altar/distributions/index.html)
      * [`altar.distributions.Base`](https://altar.readthedocs.io/en/cuda/api/altar/distributions/Base/index.html)
      * [`altar.distributions.Distribution`](https://altar.readthedocs.io/en/cuda/api/altar/distributions/Distribution/index.html)
      * [`altar.distributions.Gaussian`](https://altar.readthedocs.io/en/cuda/api/altar/distributions/Gaussian/index.html)
      * [`altar.distributions.Uniform`](https://altar.readthedocs.io/en/cuda/api/altar/distributions/Uniform/index.html)
      * [`altar.distributions.UnitGaussian`](https://altar.readthedocs.io/en/cuda/api/altar/distributions/UnitGaussian/index.html)
    * [`altar.ext`](https://altar.readthedocs.io/en/cuda/api/altar/ext/index.html)
    * [`altar.models`](https://altar.readthedocs.io/en/cuda/api/altar/models/index.html)
      * [`altar.models.cdm`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/index.html)
        * [`altar.models.cdm.ext`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/ext/index.html)
        * [`altar.models.cdm.CDM`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/CDM/index.html)
        * [`altar.models.cdm.CUDA`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/CUDA/index.html)
        * [`altar.models.cdm.Data`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/Data/index.html)
        * [`altar.models.cdm.Fast`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/Fast/index.html)
        * [`altar.models.cdm.Native`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/Native/index.html)
        * [`altar.models.cdm.Source`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/Source/index.html)
        * [`altar.models.cdm.libcdm`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/libcdm/index.html)
        * [`altar.models.cdm.meta`](https://altar.readthedocs.io/en/cuda/api/altar/models/cdm/meta/index.html)
      * [`altar.models.cudalinear`](https://altar.readthedocs.io/en/cuda/api/altar/models/cudalinear/index.html)
        * [`altar.models.cudalinear.cudaLinear`](https://altar.readthedocs.io/en/cuda/api/altar/models/cudalinear/cudaLinear/index.html)
        * [`altar.models.cudalinear.meta`](https://altar.readthedocs.io/en/cuda/api/altar/models/cudalinear/meta/index.html)
      * [`altar.models.emhp`](https://altar.readthedocs.io/en/cuda/api/altar/models/emhp/index.html)
        * [`altar.models.emhp.EMHP`](https://altar.readthedocs.io/en/cuda/api/altar/models/emhp/EMHP/index.html)
        * [`altar.models.emhp.meta`](https://altar.readthedocs.io/en/cuda/api/altar/models/emhp/meta/index.html)
      * [`altar.models.gaussian`](https://altar.readthedocs.io/en/cuda/api/altar/models/gaussian/index.html)
        * [`altar.models.gaussian.ext`](https://altar.readthedocs.io/en/cuda/api/altar/models/gaussian/ext/index.html)
        * [`altar.models.gaussian.Gaussian`](https://altar.readthedocs.io/en/cuda/api/altar/models/gaussian/Gaussian/index.html)
        * [`altar.models.gaussian.meta`](https://altar.readthedocs.io/en/cuda/api/altar/models/gaussian/meta/index.html)
      * [`altar.models.linear`](https://altar.readthedocs.io/en/cuda/api/altar/models/linear/index.html)
        * [`altar.models.linear.Linear`](https://altar.readthedocs.io/en/cuda/api/altar/models/linear/Linear/index.html)
        * [`altar.models.linear.meta`](https://altar.readthedocs.io/en/cuda/api/altar/models/linear/meta/index.html)
      * [`altar.models.mogi`](https://altar.readthedocs.io/en/cuda/api/altar/models/mogi/index.html)
        * [`altar.models.mogi.ext`](https://altar.readthedocs.io/en/cuda/api/altar/models/mogi/ext/index.html)
        * [`altar.models.mogi.CUDA`](https://altar.readthedocs.io/en/cuda/api/altar/models/mogi/CUDA/index.html)
        * [`altar.models.mogi.Data`](https://altar.readthedocs.io/en/cuda/api/altar/models/mogi/Data/index.html)
        * [`altar.models.mogi.Fast`](https://altar.readthedocs.io/en/cuda/api/altar/models/mogi/Fast/index.html)
        * [`altar.models.mogi.Mogi`](https://altar.readthedocs.io/en/cuda/api/altar/models/mogi/Mogi/index.html)
        * [`altar.models.mogi.Native`](https://altar.readthedocs.io/en/cuda/api/altar/models/mogi/Native/index.html)
        * [`altar.models.mogi.Source`](https://altar.readthedocs.io/en/cuda/api/altar/models/mogi/Source/index.html)
        * [`altar.models.mogi.meta`](https://altar.readthedocs.io/en/cuda/api/altar/models/mogi/meta/index.html)
      * [`altar.models.regression`](https://altar.readthedocs.io/en/cuda/api/altar/models/regression/index.html)
        * [`altar.models.regression.Linear`](https://altar.readthedocs.io/en/cuda/api/altar/models/regression/Linear/index.html)
        * [`altar.models.regression.meta`](https://altar.readthedocs.io/en/cuda/api/altar/models/regression/meta/index.html)
      * [`altar.models.seismic`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/index.html)
        * [`altar.models.seismic.actions`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/actions/index.html)
          * [`altar.models.seismic.actions.About`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/actions/About/index.html)
          * [`altar.models.seismic.actions.Forward`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/actions/Forward/index.html)
          * [`altar.models.seismic.actions.Sample`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/actions/Sample/index.html)
        * [`altar.models.seismic.cuda`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/cuda/index.html)
          * [`altar.models.seismic.cuda.cudaCascaded`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/cuda/cudaCascaded/index.html)
          * [`altar.models.seismic.cuda.cudaKinematicG`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/cuda/cudaKinematicG/index.html)
          * [`altar.models.seismic.cuda.cudaKinematicGCp`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/cuda/cudaKinematicGCp/index.html)
          * [`altar.models.seismic.cuda.cudaMoment`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/cuda/cudaMoment/index.html)
          * [`altar.models.seismic.cuda.cudaStatic`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/cuda/cudaStatic/index.html)
          * [`altar.models.seismic.cuda.cudaStaticCp`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/cuda/cudaStaticCp/index.html)
        * [`altar.models.seismic.ext`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/ext/index.html)
        * [`altar.models.seismic.shells`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/shells/index.html)
          * [`altar.models.seismic.shells.Action`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/shells/Action/index.html)
          * [`altar.models.seismic.shells.Seismic`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/shells/Seismic/index.html)
          * [`altar.models.seismic.shells.cudaApplication`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/shells/cudaApplication/index.html)
        * [`altar.models.seismic.Moment`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/Moment/index.html)
        * [`altar.models.seismic.Static`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/Static/index.html)
        * [`altar.models.seismic.StaticCp`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/StaticCp/index.html)
        * [`altar.models.seismic.meta`](https://altar.readthedocs.io/en/cuda/api/altar/models/seismic/meta/index.html)
      * [`altar.models.sir`](https://altar.readthedocs.io/en/cuda/api/altar/models/sir/index.html)
        * [`altar.models.sir.SIR`](https://altar.readthedocs.io/en/cuda/api/altar/models/sir/SIR/index.html)
        * [`altar.models.sir.meta`](https://altar.readthedocs.io/en/cuda/api/altar/models/sir/meta/index.html)
      * [`altar.models.Bayesian`](https://altar.readthedocs.io/en/cuda/api/altar/models/Bayesian/index.html)
      * [`altar.models.BayesianL2`](https://altar.readthedocs.io/en/cuda/api/altar/models/BayesianL2/index.html)
      * [`altar.models.Contiguous`](https://altar.readthedocs.io/en/cuda/api/altar/models/Contiguous/index.html)
      * [`altar.models.Ensemble`](https://altar.readthedocs.io/en/cuda/api/altar/models/Ensemble/index.html)
      * [`altar.models.Model`](https://altar.readthedocs.io/en/cuda/api/altar/models/Model/index.html)
      * [`altar.models.Null`](https://altar.readthedocs.io/en/cuda/api/altar/models/Null/index.html)
      * [`altar.models.ParameterSet`](https://altar.readthedocs.io/en/cuda/api/altar/models/ParameterSet/index.html)
    * [`altar.norms`](https://altar.readthedocs.io/en/cuda/api/altar/norms/index.html)
      * [`altar.norms.L2`](https://altar.readthedocs.io/en/cuda/api/altar/norms/L2/index.html)
      * [`altar.norms.Norm`](https://altar.readthedocs.io/en/cuda/api/altar/norms/Norm/index.html)
    * [`altar.shells`](https://altar.readthedocs.io/en/cuda/api/altar/shells/index.html)
      * [`altar.shells.Action`](https://altar.readthedocs.io/en/cuda/api/altar/shells/Action/index.html)
      * [`altar.shells.AlTar`](https://altar.readthedocs.io/en/cuda/api/altar/shells/AlTar/index.html)
      * [`altar.shells.Application`](https://altar.readthedocs.io/en/cuda/api/altar/shells/Application/index.html)
      * [`altar.shells.cudaAlTar`](https://altar.readthedocs.io/en/cuda/api/altar/shells/cudaAlTar/index.html)
      * [`altar.shells.cudaApplication`](https://altar.readthedocs.io/en/cuda/api/altar/shells/cudaApplication/index.html)
    * [`altar.simulations`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/index.html)
      * [`altar.simulations.Archiver`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/Archiver/index.html)
      * [`altar.simulations.Dispatcher`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/Dispatcher/index.html)
      * [`altar.simulations.GSLRNG`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/GSLRNG/index.html)
      * [`altar.simulations.Job`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/Job/index.html)
      * [`altar.simulations.Monitor`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/Monitor/index.html)
      * [`altar.simulations.RNG`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/RNG/index.html)
      * [`altar.simulations.Recorder`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/Recorder/index.html)
      * [`altar.simulations.Reporter`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/Reporter/index.html)
      * [`altar.simulations.Run`](https://altar.readthedocs.io/en/cuda/api/altar/simulations/Run/index.html)
    * [`altar.meta`](https://altar.readthedocs.io/en/cuda/api/altar/meta/index.html)




---

## Source: https://altar.readthedocs.io/en/cuda/cuda/Issues.html

# Common Issues[](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#common-issues "Link to this heading")
## Installation Issues[](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#installation-issues "Link to this heading")
### Cannot find `gmake`[](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#cannot-find-gmake "Link to this heading")
When the command of GNU make is `make` instead of `gmake`, please set the environmental variable
```
$ export GNU_MAKE=make # for bash
$ setenv GNU_MAKE make # for csh/tcsh

```

or set the variable when calling mm,
```
$ GNU_MAKE=make mm

```

### Cannot find `cublas_v2.h`[](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#cannot-find-cublas-v2-h "Link to this heading")
For certain Linux systems, NVIDIA installer installs `cublas` to the system directory `/usr/include` and `/usr/lib/x86_64-linux-gnu` instead of `/usr/local/cuda`. In this case, please add the include and library paths to `cuda.incpath` and `cuda.libpath` in `config.mm` file.
## Run-time Issues[](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#run-time-issues "Link to this heading")
### Locales[](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#locales "Link to this heading")
Error
UnicodeDecodeError: ‘ascii’ codec can’t decode byte 0xc3 in position 18: ordinal not in range(128)
You might need to set the `LANG` variable,
```
$ export LANG=en_US.UTF-8

```

if `en_US.UTF-8` locale is not installed, update your locale by
```
$ sudo apt install locales
$ sudo locale-gen --no-purge --lang en_US.UTF-8
$ sudo update-locale LANG=en_US.UTF-8 LANGUAGE

```

### Base case name[](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#base-case-name "Link to this heading")
Error
altar: bad case name: ‘patch-9’
The AlTar App cannot find the configuration file (usually) or the input file directory. Please go to the job directory with the configuration and input files, and run the App again. If the configuration is not named as `theAlTarApp.pfg`, you need `--config=YourConfigFile.pfg` option, e.g.,
```
linear --config=my_linear_model.pfg

```

### Configuration Parser Error[](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#configuration-parser-error "Link to this heading")
Error
File “/opt/anaconda3/envs/altar/lib/python3.9/site-packages/pyre/parsing/Scanner.py”, line 71, in pyre_tokenize
match = stream.match(scanner=self, tokenizer=tokenizer)
This is usually due to a bad format in configuration file. For example, `.pfg` files do not recognize TABs; please check your file for possible TABs and replace them with SPACEs. See [Pyre Config Format (.pfg)](https://altar.readthedocs.io/en/cuda/cuda/Pyre.html#pyre-config-format) for more details.
### MPI launcher error[](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#mpi-launcher-error "Link to this heading")
Error
launcher = self.mpi.launcher
AttributeError: ‘NoneType’ object has no attribute ‘launcher’
This happens when AlTar cannot locate the `mpirun` command. It can be solved by manually setting up an `mpi.pfg` file. See [MPI setup](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#mpi-setup) for more details.
### Intel MKL Library[](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#intel-mkl-library "Link to this heading")
Error
Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so.1 or libmkl_def.so.1.
This is due to a Conda issue with MKL libraries. The solution is to preload certain MKL libraries before running AlTar applications,
```
LD_PRELOAD=$CONDA_PREFIX/lib/libmkl_core.so:$CONDA_PREFIX/lib/libmkl_sequential.so altarApp --config=configFile

```



---

## Source: https://altar.readthedocs.io/en/cuda/cuda/Installation.html

# Installation Guide[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#installation-guide "Link to this heading")
## Overview[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#overview "Link to this heading")
In brief, the installation steps consist of:
  1. check [Supported Platforms](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#platforms) and install [Prerequisites](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#prerequisites);
  2. [download](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#downloads) the source packages from github;
  3. follow the [Installation Guide](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#general-steps) to compile/install `pyre` and `altar`.


Step-by-step instructions are also provided for some representative systems:
>   * [Conda method](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#anaconda3) _recommended method for Linux, Linux Clusters and MacOSX_
>   * [Ubuntu 18.04/20.04](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#ubuntu) _a standard platform_
>   * [RHEL/CentOS 7](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#redhat) _a standard platform_
>   * [Linux with environmental modules](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#lmod) _for Linux clusters_
>   * [Docker container](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#docker) _out-of-the-box delivery_
> 

## Supported Platforms[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#supported-platforms "Link to this heading")
### Hardware[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#hardware "Link to this heading")
  * CPU: `x86_64` (Intel/AMD), `ppc64` (IBM), `arm64` (Apple Silicon/ARM Neoverse/Fujitsu A64FX/…)
  * GPU: 
>     * Server/Workstation GPUs - Tesla K40, K80, P100, V100, A100, …
>     * Workstation graphic Cards - Quadro K6000, M6000, P6000, GV100, RTX x000, …
>     * Gaming graphic cards GTX10x0, RTX20x0, RTX30x0, …


Note
AlTar supports both single and double precision GPU computations. Most Quadro and gaming cards have limited double precision computing cores. However, single-precision simulation is sufficient for most models.
### Operation systems[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#operation-systems "Link to this heading")
  * Linux: any distribution should work though not all have been tested;
  * Linux clusters: with MPI support and a job queue scheduler (PBS/Slurm);
  * MacOSX: with 
  * Windows: not tested; 


Note
AlTar is designed for large scale simulations. We recommend clusters or a workstation with multiple GPUs for simulating compute-intensive models.
### Prerequisites[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#prerequisites "Link to this heading")
AlTar and pyre have several dependencies:
Required:
  * `Python3 >= 3.6` with additional Python packages `numpy` and `h5py`.
  * C/C++ compiler: `GCC >= gcc7`, or `clang >= 7`, with C++17 support. Note also that CUDA Toolkit may require certain versions of C/C++ compiler, see 
  * `GSL >= 1.15`, various numerical libraries including linear algebra and statistics.
  * `HDF5 >= 1.10`, a data management system for large and complex data sets.
  * `Postgresql`, a SQL database management system (only the library, the server itself is not required).
  * `make >= 4`, build tool.
  * `cmake >= 3.14`, build tool. Numpy component in FindPython is only supported after 3.14.


Optional:
  * `MPI` for multi-thread computations on single machine or cluster system. The recommended option is `openmpi > 1.10` with CXX support (note that on many cluster systems, openmpi is compiled without the `--enable-mpi-cxx` option and therefore doesn’t have the libmpi_cxx.so library). Other MPI implementations such as MPICH, Intel MPI are also supported.
  * `CUDA toolkit >= 10.0` for GPU-accelerated computations. Additional libraries including `cublas`, `curand`, and `cusolver` are also required.
  * An accelerated `BLAS` library, such as `atlas`, `openblas`, or `mkl`. Otherwise, the `gslcblas` library, as included in `GSL`, will be used by default.


## Downloads[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#downloads "Link to this heading")
Please choose a directory where you plan to put all the source files, e.g., `${HOME}/tools/src`,
```
mkdir -p ${HOME}/tools/src
cd ${HOME}/tools/src

```

and download the source packages of 
```
git clone https://github.com/pyre/pyre.git
git clone https://github.com/AlTarFramework/altar.git

```

Currently, some CUDA extensions to pyre and AlTar are not fully merged to the main branch. To install and run the CUDA version of AlTar 2.0, you need to download pyre and altar packages from 
```
git clone https://github.com/lijun99/pyre.git
git clone https://github.com/lijun99/altar.git

```

Note
Pyre is under active development and sometimes the newest version doesn’t work properly for AlTar. AlTar users are recommended to obtain pyre from the 
Upon successful downloads, you shall observe two directories `pyre`, `altar` under `${HOME}/tools/src` directory.
## Install with CMake[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#install-with-cmake "Link to this heading")
### General steps[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#general-steps "Link to this heading")
This section provides a general instruction on installation procedures. Please refer to the following sections for more system-specific instructions.
Compile and install PYRE at first, with the following commands,
```
# enter the source directory
cd ${HOME}/tools/src/pyre
# create a build directory
mkdir build && cd build
# call cmake to generate make files
cmake .. -DCMAKE_INSTALL_PREFIX=TARGET_DIR -DCMAKE_CUDA_ARCHITECTURES="xx"
# compile
make  # or make -j to use multi-threads
# install
make install # or sudo make install

```

By default, without using `-DCMAKE_INSTALL_PREFIX`, CMake installs the package to `/usr/local`, . If you plan to install the packages to another directory `TARGET_DIR`, you may use the `-DCMAKE_INSTALL_PREFIX` option. It is always a good practice to specify the targeted GPU architecture by, e.g, `-DCMAKE_CUDA_ARCHITECTURES="60"` (targeting NVIDIA Tesla P100 GPU) or `-DCMAKE_CUDA_ARCHITECTURES="35;60"` (targeting both K40/K80 and P100 GPUs). Please refer to [CMake Options](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#cmake-options) for more details and more options.
The installed files will appear as
```
<install_prefix>
   |--- bin  # executable shell scripts
   |   |- pyre, pyre-config ...
   |- defaults # default configuration files
   |   |- pyre.pfg, merlin.pfg
   |- include # c/c++ header files
   |   |- portinfo, <pyre>
   |- lib # shared libraries
   |   |- libjournal.so libpyre.so ... (or .dylib for Mac)
   |- packages # python packages/scripts
       |- <pyre>, <merlin>, <journal> ...

```

You may also run a few tests to check whether pyre is properly installed.
First, set up the environmental variables (you may also consider to add them to your `.bashrc` or `.cshrc`),
```
# for bash/zsh
export PATH=/usr/local/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=/usr/local/packages:${PYTHONPATH}
# for csh/tcsh
setenv PATH "/usr/local/bin:$PATH"
setenv LD_LIBRARY_PATH "/usr/local/lib:$LD_LIBRARY_PATH"
setenv PYTHONPATH "/usr/local/packages:$PYTHONPATH"

```

then run commands such as
```
# check pyre module import
python3 -c 'import pyre'
# check cuda module if enabled
# an error will be reported if the module couldn't find a GPU device
python3 -c 'import cuda'
# show the pyre installation directory
pyre-config --prefix

```

There are more test scripts under the source package `${HOME}/tools/src/pyre/tests`.
After installing PYRE and setting up properly the PATHs, you may proceed to compile/install AlTar, with the same procedure,
```
# enter the source directory
cd ${HOME}/tools/src/altar
# create a build directory
mkdir build && cd build
# call cmake to generate make files
cmake .. -DCMAKE_INSTALL_PREFIX=TARGET_DIR -DCMAKE_CUDA_ARCHITECTURES="xx"
# compile
make  # or make -j  to use multi-threads
# install
make install # or sudo make install

```

By default, AlTar is also installed to `/usr/local`. If you choose to install to another directory, you may use the same `-DCMAKE_INSTALL_PREFIX` as for PYRE. By doing so, all the PATHs only need to be set once.
To test whether AlTar is properly installed, you may run the following commands
```
# check altar module import
python3 -c 'import altar'
# show the altar installation directory
altar about prefix

```

There are also tests available in `examples` directories under each model in the source package, for example, `$(HOME)/tools/src/altar/models/linear/examples`.
### CMake Options[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#cmake-options "Link to this heading")
Here are some commonly used options to control the compilation/installation.
#### Installation path[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#installation-path "Link to this heading")
```
cmake -DCMAKE_INSTALL_PREFIX=${HOME}/tools ..

```

By default, `cmake` installs the compiled package to `/usr/local`. If you plan to install it to another system directory, or your home directory (for users who don’t have admin access), such as ${HOME}/tools as shown above. Remember to set properly the environmental variables `PATH`, `LD_LIBRARY_PATH` and `PYTHONPATH`. If you use `Conda`, you may use `-DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX`.
#### Enable/disable CUDA[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#enable-disable-cuda "Link to this heading")
```
cmake -DWITH_CUDA=ON (or OFF) ..

```

By default, WITH_CUDA=ON for the cuda branch version and WITH_CUDA=OFF for the main branch version. To enable CUDA extensions, you will also need the CUDA Toolkit. If not found, `cmake` will automatically turn WITH_CUDA=OFF.
#### Target GPU architecture(s)[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#target-gpu-architecture-s "Link to this heading")
Note
We recommend specifying a proper GPU architecture with `-DCMAKE_CUDA_ARCHITECTURES="xx"` or `-DCMAKE_CUDA_FLAGS="-arch=sm_xx"`. It not only ensures the efficient GPU executable code for your device, but also avoids the issue of code incompatibilities. CUDA Toolkit 9 and 10 use `sm_30` as the default target architecture, while CUDA 11 uses `sm_52`. The compiled code will continue to run on GPU devices with higher compute capabilities, but not on GPU devices with lower compute capabilities, reporting an error `no kernel image is available for execution on the device.` This happens, e.g., when you have a K40 (`sm_35`) and use a `sm_52` flag (default by CUDA 11). AlTar does not support CUDA on Mac, and `-DCMAKE_CUDA_FLAGS=...` will be neglected if provided.
To specify the targeted GPU architecture(s),
```
# target one architecture
cmake -DCMAKE_CUDA_ARCHITECTURES="60" ..
# target multiple architectures
cmake -DCMAKE_CUDA_ARCHITECTURES="35;60" ..

```

Note that `CUDA_ARCHITECTURES` is only available on CMake 3.18 and later versions. For earlier versions, you may use `CUDA_FLAGS` instead,
```
# target one architecture
cmake -DCMAKE_CUDA_FLAGS="-arch=sm_60" ..
# target multiple architectures
cmake -DCMAKE_CUDA_FLAGS="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_60,code=sm_60" ..

```

`CUDA_FLAGS` may also be used to pass other compiling options to CUDA compiler `nvcc`.
You may find out which type(s) of GPU are installed by running
```
nvidia-smi

```

Compute capabilities for some common NVIDIA GPUs are K40/80 (`sm_35`), V100 (`sm_70`), A100 (`sm_80`), GTX1050/1070/1080 ((`sm_61`), RTX 2060/2070/2080 (`sm_75`), RTX 3060/3070/3080(`sm_86`. More details can be found at 
```
python3 -c "import cuda; [print(f'Device {device.id} {device.name} has compute capability {device.capability}') for device in cuda.devices]"

```

#### C++ Compiler[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#c-compiler "Link to this heading")
To specify the C++ compiler, e.g., /usr/bin/g++, you may use
```
cmake -DCMAKE_CXX_COMPILER=/usr/bin/g++ ..

```

Note that pyre requires a GCC>=7 for c++17 support.
C++ compiler may also be specified from the environmental variable `CXX`, for example,
```
# bash/zsh
export CXX = "/usr/bin/g++"
# csh/tcsh
setenv CXX  "/usr/bin/g++"

```

#### CUDA Compiler[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#cuda-compiler "Link to this heading")
To specify the CUDA compiler `nvcc`, e.g., /usr/local/cuda-11.3/bin/nvcc, you may use
```
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.3/bin/nvcc ..

```

C++ compiler may also be specified from the environmental variable `CUDACXX`, for example,
```
# bash/zsh
export CUDACXX = "/usr/local/cuda-11.3/bin/nvcc -arch=sm_60"
# csh/tcsh
setenv CUDACXX  "/usr/local/cuda-11.3/bin/nvcc -arch=sm_60"

```

#### BLAS Library[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#blas-library "Link to this heading")
Pyre requires a BLAS library for its `gsl` module. CMake searches automatically an available BLAS library by default. If none is found, the `gslcblas` library included with GSL package will be used. You may also specify which BLAS library to use by
```
cmake .. -DBLA_VENDOR=vendor

```

where `vendor` can be `Generic``(``libblas.so`), `ATLAS`, `Intel10_64lp`, `OpenBLAS` …. You may also add `-DCMAKE_PREFIX_PATH=/path/to/blas` to enforce a search path.
#### Library search path[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#library-search-path "Link to this heading")
To specify the locations of a prerequisite library instead of the default one, for example, on some Linux systems, `cmake` may find and use libraries from `/usr/` instead of the libraries provided by conda, you may use
```
cmake -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} ..

```

to enforce libraries installed under Conda to be used.
For more than one paths, use a semicolon separated list, -DCMAKE_PREFIX_PATH=”PATH1;PATH2;PATH3”.
#### Build type[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#build-type "Link to this heading")
```
cmake -DCMAKE_BUILD_TYPE=Release (or Debug) ..

```

For the Debug build type, the `-g` compiler flag will be added to generate debugging information. For the Release type, the `-O3` optimization flag will be added. If none is specified, the default flags of `g++` are used.
#### Show compiling details[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#show-compiling-details "Link to this heading")
By default, the compiling step `make` only shows one line summary of each file being compiled. To the detailed compiling command and options, you may use
```
make VERBOSE=1

```

#### More options[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#more-options "Link to this heading")
For more options of `cmake`, please check 
## Conda method (Linux/MacOSX)[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#conda-method-linux-macosx "Link to this heading")
### Install Anaconda/Miniconda[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#install-anaconda-miniconda "Link to this heading")
Conda(Anaconda/Miniconda) offers an easy way to install Python, packages and libraries on different platforms, especially for users without the admin privilege to their computers. We recommend a full version of 
For MacOSX with Apple Silicon, you may install the native `arm64` version from 
If Anaconda3 is not installed, please `${HOME}/anaconda3` (default) or a system directory, e.g., `/opt/anaconda3`. The path to the Anaconda3 is set as an environmental variable `CONDA_PREFIX`. To check whether Anaconda3 is properly installed and loaded, you may try the following commands
```
$ which conda
/opt/anaconda3/bin/conda
$ which python3
/opt/anaconda3/bin/python3
$ echo ${CONDA_PREFIX}
/opt/anaconda3

```

You may also create a virtual environment
```
$ conda create -n altar
$ conda activate altar
$ which python3
/opt/anaconda3/envs/altar/bin/python3
$ echo ${CONDA_PREFIX}
/opt/anaconda3/envs/altar

```

### Install Conda packages[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#install-conda-packages "Link to this heading")
Install the required libraries and packages by Conda:
```
$ conda install git make cmake hdf5 h5py openmpi gsl openblas postgresql numpy scipy

```

### C++ Compiler[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#id7 "Link to this heading")
You will also need a c++ compiler.
  * **Ubuntu 18.04/20.04** GCC 7.4.0/9.3.0 is installed by default and is sufficient. If GCC/G++ are not installed, run


```
sudo apt install gcc g++

```

  * **Redhat/CentOS 7** The system default compiler GCC 4.x doesn’t support C++17. Higher versions of GCC are offered through `devtoolset`. Please follow instructions for `devtoolset-7`. An alternative is to use GNU compilers provided by Conda, see below.
  * **MacOSX** You will need to install either the full version of Xcode or the (compact) Command Line Tools. Xcode can be installed from the App Store. To install the Command Line Tools, run


```
sudo xcode-select --install
# To select or switch compilers,
sudo xcode-select --switch /Library/Developer/CommandLineTools/

```

  * **Conda GNU Compilers** Conda also offers compiler packages, which work well for most Linux/MacOSX(Intel) systems,


```
# for Linux x86_64
conda install gcc_linux-64 gxx_linux-64
# for Mac (Intel Only)
conda install clang_osx-64 clangxx_osx-64
# for Mac Big Sur with Xcode 12 (Intel only), you need to use clang-11,
conda install clang_osx-64=11.0.0 clangxx_osx-64=11.0.0 -c conda-forge
# for Mac with Apple Silicon, please use only Command Line Tools or Xcode

```

If you would like to use a c++ compiler other than the default version, or the version (auto) discovered by `cmake`, you may use `-DCMAKE_CXX_COMPILER=...` to specify the compiler.
### CUDA compiler (nvcc)[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#cuda-compiler-nvcc "Link to this heading")
CUDA Toolkit integrates tools to develop GPU applications, including the compiler (`nvcc`), libraries (`libcudart.so`, `libcublas.so` …). If CUDA is installed, you may obtain and install CUDA Toolkit following the 
CUDA Toolkit is usually installed to `/usr/local/cuda`. On Linux clusters, many version of CUDA toolkit may be provided as modules. You may select a version by
```
module load cuda/11.3

```

Conda also provides a CUDA Toolkit package,
```
conda install cudatoolkit

```

which is installed to `$CONDA_PREFIX` directory.
You may check the CUDA Toolkit installation by
```
# check the nvcc availability and path
$ which nvcc
/usr/local/cuda/bin/nvcc
# check the CUDA Toolkit version
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 11.3, V11.3.109

```

`CMake` discovers the default `nvcc` command to compile CUDA programs. You may also specify anther CUDA compiler by `-DCMAKE_CUDA_COMPILER=/path/to/nvcc`.
Note that NVIDIA driver, including the CUDA driver (`libcuda.so`), is required on a GPU workstation or GPU nodes in a cluster. NVIDIA drivers can only be installed/updated by _root_ users. You may check their availability and versions by the command `nvidia-smi`. CUDA shared libraries should also be available on GPU workstations.
### Download pyre and AlTar[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#download-pyre-and-altar "Link to this heading")
Please download the source packages of pyre/AlTar from github following the [Download instructions](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#downloads). Taking CUDA branch versions as an example,
```
mkdir -p ${HOME}/tools/src
cd ${HOME}/tools/src
git clone https://github.com/lijun99/pyre.git
git clone https://github.com/lijun99/altar.git

```

### Install pyre[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#install-pyre "Link to this heading")
With Conda, we recommend installing pyre and AlTar to `$CONDA_PREFIX`, so that both packages are loaded automatically when conda or conda venv is activated. We need an extra step to make a symbolic link to `lib/python3.x/site-packages`,
```
# the python command returns the path of site-packages, and we link it as $CONDA_PREFIX/packages
ln -sf `python3 -c 'import site; print(site.getsitepackages()[0])'` $CONDA_PREFIX/packages

```

Compile and install pyre
```
cd ${HOME}/tools/src/pyre
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_CUDA_ARCHITECTURES=native -DBLA_VENDOR=OpenBLAS -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python3

make -j && make install

```

where `INSTALL_PREFIX` is the installation path and `PREFIX_PATH` is the path to search the prerequisite packages. Replace `native` with appropriate compute capability number(s) for your GPU(s). See [GPU architecture(s)](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#gpu-architecture) for more details.
**Note** : `FindPython3` in new versions of `cmake` sometimes finds the system python3 interpreter instead of the conda installed one, please add `-DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python3` as above to assist the search. The new standard is to use `FindPython` instead; we will update the pyre/altar cmake files.
### Install AlTar[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#install-altar "Link to this heading")
Since pyre is installed to `$CONDA_PREFIX`, there is no need to set the PATHs. We proceed to compile and install AlTar, with the same procedure,
```
cd ${HOME}/tools/src/altar
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_CUDA_ARCHITECTURES=native -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python3

make -j && make install

```

Please read [CMake Options](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#cmake-options) if you have some problems or need more customizations. Please also read [Installation instructions](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#general-steps) on how to make tests.
For future runs, you may simply activate conda or the conda venv to load AlTar,
```
# activate altar if it is installed in a venv
conda activate altar
# test
altar about

```

### MPI setup[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#mpi-setup "Link to this heading")
AlTar runs MPI jobs by automatically forking multiple threads and invoking `mpirun` command with any AlTar Application, a capability offered by pyre `mpi` shell. However, pyre sometimes does not recognize the Conda-installed openMPI. You will need to create manually a configuration file, `mpi.pfg`, either under `$(HOME)/.pyre` directory or under the current job directory, as
```
; mpi.pfg file

mpi.shells.mpirun:
  ; mpi implementation
  mpi = openmpi#mpi_conda

; mpi configuration
pyre.externals.mpi.openmpi # mpi_conda:
  version = 4.0.5
  launcher = mpirun
  prefix = /opt/anaconda3/envs/altar
  bindir = {mpi_conda.prefix}/bin
  incdir = {mpi_conda.prefix}/include
  libdir = {mpi_conda.prefix}/lib

```

You need to replace `/opt/anaconda3/envs/altar` with the actual path of your `$CONDA_PREFIX`, which can be revealed by the command `echo $CONDA_PREFIX`.
Another option is to insert these lines to your job configuration file, without creating a separate `mpi.pfg` file.
This setup procedure also applies to other MPIs not automatically recognized by pyre, e.g., loaded by environmental modules.
## Linux Systems[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#linux-systems "Link to this heading")
We recommend Conda methods for all Linux systems. If you prefer to use the standard Linux packages, please follow the instructions in this section.
### Ubuntu 18.04/20.04[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#ubuntu-18-04-20-04 "Link to this heading")
#### Install prerequisites[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#install-prerequisites "Link to this heading")
```
$ sudo apt update && sudo apt install -y gcc g++ python3 python3-dev python3-numpy python3-scipy python3-h5py libgsl-dev libopenblas-dev libpq-dev postgresql-server-dev-all libopenmpi-dev libhdf5-serial-dev make git

```

For Ubuntu 18.04 only: the system CMake version is 3.10; you need to manually upgrade cmake from 
```
$ sudo wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
$ sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
$ sudo apt-get update
$ sudo apt-get install cmake

```

To install/run CUDA modules, you will also need to install CUDA Toolkit if it is not pre-installed. See [CUDA compiler (nvcc)](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#cuda-toolkit) for more details.
#### Install pyre/AlTar[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#install-pyre-altar "Link to this heading")
Please follow the instructions in [General steps](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#general-steps).
### RHEL/CentOS 7[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#rhel-centos-7 "Link to this heading")
#### Install prerequisites[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#id9 "Link to this heading")
Enable EPEL repo
```
yum install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm

```

and install prerequisites
```
yum install -y python3 python3-devel hdf5-devel gsl-devel postgresql-devel openmpi openmpi-devel git environment-modules
# install numpy/scipy/h5py via pip
pip3 install numpy scipy h5py
# load openmpi
module load mpi
# install cmake from Kitware
wget https://github.com/Kitware/CMake/releases/download/v3.19.3/cmake-3.19.3-Linux-x86_64.sh
sh cmake-3.19.3-Linux-x86_64.sh  --skip-license --prefix=/usr/local

```

Install C/C++ compiler
```
# 1. Install a package with repository for your system:
# On CentOS, install package centos-release-scl available in CentOS repository:
sudo yum install centos-release-scl

# On RHEL, enable RHSCL repository for you system:
sudo yum-config-manager --enable rhel-server-rhscl-7-rpms

# 2. Install the collection:
sudo yum install devtoolset-7

# 3. Start using software collections:
scl enable devtoolset-7 bash

```

#### Install pyre/AlTar[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#id10 "Link to this heading")
Please follow the instructions in [General steps](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#general-steps).
### Linux with software modules[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#linux-with-software-modules "Link to this heading")
Many clusters use software modules to load libraries and software packages, e.g.,
```
# list available modules
module av
# load a certain module
module load cuda/10.2
# list loaded modules
module list
# show the loaded module information
module show cuda

```

Please load all necessary modules as listed in [Prerequisites](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#prerequisites). You may then follow the [General steps](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#general-steps) above to install pyre and AlTar.
Since modules are set up differently in different computers, we only provide a general prescription if CMake fails to locate the prerequisite package from auto search.
You may provide the package path to CMake by,
>   * `-DCMAKE_PREFIX_PATH`, which specifies the package installation prefix to be searched. Files are expected to be arranged in a standard fashion under prefix, `bin`, `includes`, `lib`. If the files are not arranged in the standard way, you may use options below,
>   * `-DCMAKE_INCLUDE_PATH`, which specifies the search path(s) for header files;
>   * `-DCMAKE_LIBRARY_PATH`. which specifies the search paths(s) for library files.
> 

Each of these three parameters can be a semicolon-separated list to include more than one paths, e.g., `-DCMAKE_PREFIX_PATH="/PATH/To/GSL;/PATH/To/HDF5;/PATH/To/MPI`.
CMake uses various builtin _Find modules_ to search various packages, while each _Find module_ may use some _hints_ to locate the package. For example, `FindGSL` uses `GSL_ROOT_DIR`, and `FindMPI` uses `MPIEXEC_EXECUTABLE` or `MPI_HOME`. These mint may be passed as environmental variables `export GSL_ROOT_DIR=...` or as cmake options, e.g., `-DGSL_ROOT_DIR=...`.
Note
Many clusters have their own recommended MPI packages which are optimized for the specific type of interconnects. Before using these pre-installed MPI packages, please check whether they have `cxx` devel support, or `libmpi_cxx.so` is available, which is required by AlTar.
## Docker container[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#docker-container "Link to this heading")
You may follow the steps below to build a docker image, based on a NVIDIA cuda 10.2 image with Ubuntu.
```
wget https://gitlab.com/nvidia/container-images/cuda/raw/master/dist/ubuntu18.04/10.2/runtime/Dockerfile
docker build --build-arg IMAGE_NAME=nvidia/cuda . -t cuda/nvidia:10.2
docker exec -it cuda/nvidia:10.2
apt update && apt install -y gcc g++ python3 python3-dev python3-numpy python3-numpy-dev python3-h5py libgsl-dev libopenblas-dev libpq-dev postgresql-server-dev-all libopenmpi-dev libhdf5-serial-dev make git wget software-properties-common locales
locale-gen --no-purge --lang en_US.UTF-8 && update-locale LANG=en_US.UTF-8 LANGUAGE
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && apt-get update && apt install -y cmake
apt install -y cuda-compiler-10-2 cuda-cudart-dev-10-2 cuda-curand-dev-10-2 libcublas-dev cuda-cusolver-dev-10-2
ln -sf /usr/lib/python3/dist-packages /usr/local/packages
cd /usr/local/src
git clone https://github.com/lijun99/pyre.git
git clone https://github.com/lijun99/altar.git
cd /usr/local/src/pyre && mkdir build && cd build && cmake .. && make all && make install
cd /usr/local/src/altar && mkdir build && cd build && cmake .. && make all && make install
echo ': "${LANG:=en_US.UTF-8}"; export LANG' >> /etc/profile

```

Open another terminal, find out the _CONTAINER ID_ for this image, named _cuda/nvidia:10.2_ , and commit the changes to a new image
```
$ docker commit CONTAINER_ID altar:2.0.2-cuda

```

To run AlTar from the container
```
$ docker run --gpus all -ti -v ${PWD}:/mnt altar:2.0.2-cuda

```

which also mounts the current directory as /mnt in the virtual system. You may go to your job directory and run AlTar from there.
If you meet a `UnicodeDecodeError`, you will need to `export LANG=en_US.UTF-8` at first. See [Locales](https://altar.readthedocs.io/en/cuda/cuda/Issues.html#locales) for more details.
OpenMPI may issue a warning to run MPI jobs as a _root_ user, you may add the `--allow-run-as-root` option to your job configuration file as follows,
```
; for parallel runs
mpi.shells.mpirun:
    extra = -mca btl self,tcp --allow-run-as-root

```

## Install with the [](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#install-with-the-mm-build-tool "Link to this heading")
The 
### Download `mm` build tool[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#download-mm-build-tool "Link to this heading")
```
cd ${HOME}/tools/src
git clone https://github.com/aivazis/mm.git

```

### Prepare a `config.mm` file[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#prepare-a-config-mm-file "Link to this heading")
The `mm` build tool requires a `config.mm` file to locate dependent libraries or packages. Taking Ubuntu 18.04 as an example, the `config.mm` file appear as
```
# file config.mm

# gsl
gsl.dir = /usr
gsl.incpath = /usr/include
gsl.libpath = /usr/lib/x86_64-linux-gnu

# mpi
mpi.dir = /usr/lib/x86_64-linux-gnu/openmpi/
mpi.binpath = /usr/bin
mpi.incpath = /usr/lib/x86_64-linux-gnu/openmpi/include
mpi.libpath = /usr/lib/x86_64-linux-gnu/openmpi/lib
mpi.flavor = openmpi
mpi.executive = mpirun

# hdf5
hdf5.dir = /usr
hdf5.incpath = /usr/include
hdf5.libpath = /usr/lib/x86_64-linux-gnu

# postgresql
libpq.dir = /usr
libpq.incpath = /usr/include/postgresql
libpq.libpath = /usr/lib/x86_64-linux-gnu

# openblas
openblas.dir = /usr
openblas.libpath = /usr/lib/x86_64-linux-gnu

# python3
python.version = 3.6
python.dir = /usr
python.binpath = /usr/bin
python.incpath = /usr/include/python3.6m
python.libpath = /usr/lib/python3.6

# numpy
numpy.dir = /usr/lib/python3/dist-packages/numpy/core

# cuda
cuda.dir = /usr/local/cuda
cuda.binpath = /usr/local/cuda/bin
cuda.incpath = /usr/local/cuda/include
cuda.libpath = /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu/
cuda.libraries := cudart cublas curand cusolver

# end of file

```

You may leave the `config.mm` file in the `pyre/.mm`, `altar/.mm` directories, or in the `${HOME}/.mm` directory to be shared by all projects.
Examples of config.mm files are available at 
### Install pyre[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#id11 "Link to this heading")
After preparing all required libraries/packages and the `config.mm` file (in `pyre/.mm` or `${HOME}/.mm`), you need to compile and install pyre at first.
Make an alias of the `bash`
```
$ alias mm='python3 ${HOME}/tools/src/mm/mm.py'

```

or in `csh/tcsh`,
```
$ alias mm 'python3 ${HOME}/tools/src/mm/mm.py'

```

Now, you can compile `pyre` by
```
$ cd ${HOME}/tools/src/pyre
$ mm

```

By default, the compiled files are located at `${HOME}/tools/src/pyre/products/debug-shared-linux-x86_64`. If you need to customize the installation, you can check the options offered by `mm` by
```
$ mm --help

```

For example, if you prefer to install pyre to a system folder, you may use `--prefix` option, such as
```
$ mm --prefix=/usr/local

```

After compiling/installation, you need to set up some environmental variables for other applications to access `pyre`, for example, create a `${HOME}/.pyre.rc` for `bash`,
```
# file .pyre.rc
export PYRE_DIR=${HOME}/tools/src/pyre/products/debug-shared-linux-x86_64
export PATH=${PYRE_DIR}/bin:$PATH
export LD_LIBRARY_PATH=${PYRE_DIR}/lib:$LD_LIBRARY_PATH
export PYTHONPATH=${PYRE_DIR}/packages:$PYTHONPATH
export MM_INCLUDES=${PYRE_DIR}/include
export MM_LIBPATH=${PYRE_DIR}/lib
# end of file

```

or `${HOME}/.pyre.cshrc` for `csh/tcsh`,
```
# file .pyre.cshrc
setenv PYRE_DIR "${HOME}/tools/src/pyre/products/debug-shared-linux-x86_64"
setenv PATH "${PYRE_DIR}/bin:$PATH"
setenv LD_LIBRARY_PATH "${PYRE_DIR}/lib:$LD_LIBRARY_PATH"
setenv PYTHONPATH "${PYRE_DIR}/packages:$PYTHONPATH"
setenv MM_INCLUDES "${PYRE_DIR}/include"
setenv MM_LIBPATH "${PYRE_DIR}/lib"
# end of file

```

You will also need to append `pyre` configurations to `${HOME}/.mm/config.mm` or `altar/.mm/config.mm` or any other application who requires `pyre`,
```
# append to the following lines to an existing config.mm
# pyre
pyre.dir =  ${HOME}/tools/src/pyre/products/debug-shared-linux-x86_64
pyre.libraries := pyre journal ${if ${value cuda.dir}, pyrecuda}

```

### Install AlTar[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#id12 "Link to this heading")
First, make sure that you have a prepared `config.mm` file, which also includes the `pyre` configuration, in either `altar/.mm/` or `${HOME}/.mm` directory.
Follow the same step to compile AlTar,
```
$ cd ${HOME}/tools/src/altar
$ mm

```

Similar to `pyre` installation, the AlTar products are located at `${HOME}/tools/src/altar/products/debug-shared-linux-x86_64`, or the directory specified by `mm --prefix=`.
Also, you need to set up some environmental variables for `altar` as well, for example, create a `${HOME}/.altar2.rc` for `bash`,
```
# file .altar2.rc
export ALTAR2_DIR=${HOME}/tools/src/altar/products/debug-shared-linux-x86_64
export PATH=${ALTAR2_DIR}/bin:$PATH
export LD_LIBRARY_PATH=${ALTAR2_DIR}/lib:$LD_LIBRARY_PATH
export PYTHONPATH=${ALTAR2_DIR}/packages:$PYTHONPATH
# end of file

```

or `${HOME}/.altar2.cshrc` for `csh/tcsh`,
```
# file .altar2.cshrc
setenv ALTAR2_DIR "${HOME}/tools/src/altar/products/debug-shared-linux-x86_64"
setenv PATH "${ALTAR2_DIR}/bin:$PATH"
setenv LD_LIBRARY_PATH "${ALTAR2_DIR}/lib:$LD_LIBRARY_PATH"
setenv PYTHONPATH "${ALTAR2_DIR}/packages:$PYTHONPATH"
# end of file

```

Before running an altar/pyre application, you need to load the altar/pyre environmental settings
```
$ source ${HOME}/.pyre.rc
$ source ${HOME}/.altar2.rc

```

## Tests and Examples[](https://altar.readthedocs.io/en/cuda/cuda/Installation.html#tests-and-examples "Link to this heading")
Pyre tests are available at `${HOME}/tools/src/pyre/tests`.
AlTar examples are are available for each model.
For details how to run AlTar applications, please refer to [User Guide](https://altar.readthedocs.io/en/cuda/cuda/Manual.html#user-guide).


---

## Source: https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html

# Kinematic Slip Inversion[](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#kinematic-slip-inversion "Link to this heading")
## Kinematic Source Model[](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#kinematic-source-model "Link to this heading")
The kinematic source model studies also the temporal evolution of coseismic slips. The kinematic source model currently implemented in AlTar is formulated as follows.
The fault plane, as a rectangular area, is divided into Ndd×Nas square patches (dd=down dip, as=along strike), and time is divided into Nt intervals. The kinematic slip function Mb(ξ→,t) (big M) can be in general written as
Mb(ξ→,t)=D(ξ→)S(t−TR(ξ→);Tr(ξ→)),
where ξ→,t label the patch and time, respectively, and
  * D(ξ→) is the coseismic slip vector, also subject to the static source model inversion;
  * TR(ξ→) is the initial rupture time of each patch, determined by solving the Eikonal equation with a Fast Sweeping algorithm, given the location of the hypocenter H0 and the rupture velocity Vr(ξ→) (assumed to be isotropic and the same within each patch);
  * Tr(ξ→) is the slip duration (rise time) for each patch;
  * S(t,Tr) is the source time function which is finite only for t∈[0,Tr] and integrated to 1: we choose a triangular source time function.


To produce smooth synthetics, we refine each patch into Nmesh×Nmesh grids when solving the eikonal equation, while dividing each time interval into Npt points. Mb(ξ→,t) are then interpolated and integrated from these finer spatial and temporal meshes.
The predicted observations can be calculated by,
dprediction(x→,t)=∑ξ→,t′Gb(x→,ξ→;t−t′)Mb(ξ→,t′)
which is a linear model (note that the Eikonal equation is non-linear). Here, Gb(x→,ξ→;t−t′) (big G) are the kinematic Green’s functions relating the source Mb(ξ→,t′) to an observation location at a given time (x→,t). The kinematic Green’s functions are pre-calculated as inputs to AlTar. There are some existing software packages for the calculation, e.g., the frequency-wavenumber integration code of 
In summary, the kinematic inversion uses the following source parameters
>   * the two components of coseismic slip D(ξ→) (2×Ndd×Nas elements),
>   * the rupture velocity Vr(ξ→) (Ndd×Nas elements),
>   * the rise time Tr(ξ→) (Ndd×Nas elements),
>   * the location of the hypocenter H0 (2 elements),
> 

and the forward model
dpred=G(D(ξ→),Vr(ξ→),Tr(ξ→),H0)
is preformed in two steps,
>   * to obtain Mb(ξ→,t′) from the Eikonal equation sovler,
>   * to calculate dpred=GbMb.
> 

## Joint Kinematic-Static Inversion[](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#joint-kinematic-static-inversion "Link to this heading")
Because the slips D(ξ→) are subject to both static and kinematic observations, we in general run static and kinematic models together, by a model ensemble with or without cascading.
The joint Bayesian probability for static and kinematic models can be written as
P(θc,θs,θk|ds,dk)=P(θc)P(θs)P(θk)P(ds|θc,θs)P(dk|θc,θk).
where θc=[strikeslip,dipslip] as shared (common) parameters, θs=[ramp], and θk=[risetime,rupturevelocity,hypocenter]. The data likelihoods are computed from
  * the static model with observations ds and the forward model dspred=Gs(θc,θs),
  * the kinematic model with dk and dkpred=Gk(θc,θk).


In the annealing schemes such as CATMIP, we can introduce transitioning distributions
Pβs,βk(θc,θs,θk|ds,dk)=P(θc)P(θs)P(θk)[P(ds|θc,θs)]βs[P(dk|θc,θk)]βk.
where both βs and βk vary from 0 to 1, either jointly or independently. Depending on how βs,βk vary, AlTar supports two schemes:
  * In the non-cascading scheme, we set β=βs=βk, i.e., both increasing at the same pace, while their increment in the COV scheduler is determined by the coefficient of variation of the weights w=[P(ds|θc,θs)P(dk|θc,θk)]βm+1−βm.
  * In the cascading scheme, we run AlTar twice:
    1. In the first run, we perform inversion on the static model only, or producing samples of θc and θs pursuant to the posterior distribution of the static model P(θc,θs|ds)=P(θc)P(θs)[P(ds|θc,θs)].
    2. In the second run, we run static and kinematic models together, by setting βs=1 (`cascaded = True`) and varying βk from 0 to 1 (`cascaded = False`). Here, the samples of θc and θs from the static inversion are used as the initial samples for the joint inversion. The increment of βk in the COV scheduler is determined by the coefficient of variation of the weights w=[P(dk|θc,θk)]βk,m+1−βk,m.


With the optimized and (usually greatly) reduced search range of θc and θs in parameter space from the static inversion, the cascaded joint-static-kinematic inversion runs more efficiently than the non-cascading scheme. In general, the cascading scheme is recommended for all model ensembles if there is a computation-intensive model (such as the kinematic model) present.
## Configurations (Kinematic Model only)[](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#configurations-kinematic-model-only "Link to this heading")
We illustrate the settings of the kinematic model by assuming to run it alone (in practice this is rarely adopted).
### An example configuration file[](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#an-example-configuration-file "Link to this heading")
The configuration file (`kinematicg_only.pfg`) for the kinematic model appears as
```
; application instance name
slipmodel:

    ; model to be sampled
    model = altar.models.seismic.cuda.kinematicg
    model:

        dataobs:
            observations = 14148 ; number of observed data points
            data_file = kinematicG.data.h5
            cd_std = 5.0e-3
            ; or cd_file = kinematicG.cd.h5 if using a file input

        ; fixed model parameters
        ; green's function (2*Ndd*Nas*Nt, observations)
        ; [Nt][2(strike/dip)][Nas][Ndd] with leading dimensions on the right
        green = kinematicG.gf.h5

        Ndd = 3 ; patches along dip
        Nas = 3 ; patches along strike
        Nmesh = 30 ; mesh points for each patch
        dsp = 20.0 ; length for each patch, km
        Nt = 90 ; number of time intervals
        Npt = 2 ; mesh points for each time interval
        dt = 1.0 ; time unit for each interval, second
        ; initial starting time for each patch, in addition to the fast sweeping calculated arrival time
        t0s = [0.0] * {slipmodel.model.patches}

        ; parameters to be simulated
        ; provide a list at first, serving as their orders in theta
        psets_list = [strikeslip, dipslip, risetime, rupturevelocity, hypocenter]

        ; define each parameterset
        psets:
            strikeslip = altar.cuda.models.parameterset
            dipslip = altar.cuda.models.parameterset
            risetime = altar.cuda.models.parameterset
            rupturevelocity = altar.cuda.models.parameterset
            hypocenter = altar.cuda.models.parameterset

            ; variables for patches are arranged along dip direction at first [Nas][Ndd]
            strikeslip:
                count = {slipmodel.model.patches}
                prep = altar.cuda.distributions.preset ; load preset samples
                prep.input_file = theta_cascaded.h5 ; file name
                prep.dataset = ParameterSets/strikeslip ; dataset name in h5
                prior = altar.cuda.distributions.gaussian
                prior.mean = 0
                prior.sigma = 0.5

            dipslip:
                count = {slipmodel.model.patches}
                prep = altar.cuda.distributions.preset
                prep.input_file = theta_cascaded.h5 ; file name
                prep.dataset = ParameterSets/dipslip ; dataset name in h5
                prior = altar.cuda.distributions.uniform
                prior.support = (-0.5, 20.0)

            risetime:
                count = {slipmodel.model.patches}
                prior = altar.cuda.distributions.uniform
                prior.support = (10.0, 30.0)

            rupturevelocity:
                count = {slipmodel.model.patches}
                prior = altar.cuda.distributions.uniform
                prior.support= (1.0, 6.0)

            ; along strike(first), dip directions
            ; could be separated into 2 for dip and strike direction
            hypocenter:
                count = 2
                prior = altar.cuda.distributions.gaussian
                prior.mean = 20.0
                prior.sigma = 5.0

```

### Parameter Sets[](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#parameter-sets "Link to this heading")
The parameter sets or `psets` for the kinematic models are `psets_list = [strikeslip, dipslip, risetime, rupturevelocity, hypocenter]`.
  * The names the parameter sets can be changed per your preference, e.g., `strike_slip`, `StrikeSlip`. But the order of the parameter sets must be preserved because the forward model uses the order to map appropriate parameters. `strikeslip` and `dipslip` may be switched as long as their order is consistent with the Green’s functions.
  * `strikeslip` and `dipslip` are two components of the cumulative slip displacement. If you prefer to load their initial samples from the static inversion results, use the `altar.cuda.distributions.preset` distribution for `prep`, see [Preset](https://altar.readthedocs.io/en/cuda/cuda/Priors.html#preset) distribution for more details. Only `HDF5` format is accepted for Preset prior and therefore, its dataset name `prep.dataset=ParameterSets/strikeslip` is also required. If you choose to generate samples from a given distribution, e.g., gaussian/moment scale distributions, please follow the [Parameter Sets](https://altar.readthedocs.io/en/cuda/cuda/Static.html#static-parameter-sets) example in static inversion to set their `prep` and `prior` distributions. The slips are usually in unit of meters.
  * `risetime` (in unit of seconds) and `rupturevelocity` (in unit of km/s) are rupture duration and velocities for each patch. As they are positive, usually uniform or truncated gaussian distributions are used as their priors.
  * `strikeslip`, `dipslip`, `risetime` and `rupturevelocity` are defined for each patch and their counts are the same as the number of patches. The sequence of patches is arranged as, for Ndd×Nas patches, (as0,dd0),(as0,dd1),...(as0,ddNdd−1),(as1,dd0),...,(asNas−1,ddNdd−1). Or `dd` is the leading dimension.
  * `hypocenter` (in unit of km) is the location of the hypocenter measured from the **CENTER** of the (as0,dd0) patch (note that it’s not the origin or the corner), in unit of kilometers. If the distances down dip (dd) and along strike (as) directions are different, you may separate them as two parameter sets `hypo_as` and `hypo_dd`, with `as` component being first.


### Input files[](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#input-files "Link to this heading")
The kinematic model requires the following input files 

green: 
    
the kinematic Green’s functions, with the `shape=(2*Ndd*Nas*Nt, observations)`. The `observations` is the number of observed data points, and is the leading dimension. `[Nt][2(strike/dip)][Nas][Ndd]` labels the spatial-temporal source displacements with leading dimensions on the right (or which comes first):
```
(t=0, strike, as_0, dd_0, obs_0), (t=0, strike, as_0, dd_0, obs_1), ..., (t=0, strike, as_0, dd_0, obs_{Nobs-1})
(t=0, strike, as_0, dd_1, obs_0), (t=0, strike, as_0, dd_1, obs_1), ..., (t=0, strike, as_0, dd_1, obs_{Nobs-1})
... ...
(t=0, strike, as_0, dd_{Ndd-1}, obs_0), (t=0, strike, as_0, dd_{Ndd-1}, obs_1), ...,  (t=0, strike, as_0, dd_{Ndd-1}, obs_{Nobs-1})
(t=0, strike, as_1, dd_0, obs_0), (t=0, strike, as_1, dd_0, obs_1), ..., (t=0, strike, as_1, dd_0, obs_{Nobs-1})
... ...
(t=0, strike, as_{Nas-1}, dd_{Ndd-1}, obs_0), (t=0, strike, as_{Nas-1}, dd_{Ndd-1}, obs_1), ..., (t=0, strike, as_{Nas-1}, dd_{Ndd-1}, obs_{Nobs-1})
(t=0, dip, as_0, dd_0, obs_0), (t=0, dip, as_0, dd_0, obs_1), ..., (t=0, dip, as_0, dd_0, obs_{Nobs-1})
... ...
(t=0, dip, as_{Nas-1}, dd_{Ndd-1}, obs_0), (t=0, dip, as_{Nas-1}, dd_{Ndd-1}, obs_1), ..., (t=0, dip, as_{Nas-1}, dd_{Ndd-1}, obs_{Nobs-1})
(t=1, strike, as_0, dd_0, obs_0), (t=1, strike, as_0, dd_0, obs_1), ..., (t=1, strike, as_0, dd_0, obs_{Nobs-1})
... ...
(t={Nt-1}, dip, as_{Nas-1}, dd_{Ndd-1}, obs_0), (t={Nt-1}, dip, as_{Nas-1}, dd_{Ndd-1}, obs_1), ..., (t={Nt-1}, dip, as_{Nas-1}, dd_{Ndd-1}, obs_{Nobs-1})

You need to follow the above order when preparing the Green's functions as it's the order how big-M is arranged in the forward model.

```


dataobs.data_file: 
    
1d vector of observed data. 

dataobs.cd_file: 
    
the data covariance matrix with `shape=(observations, observations)`. If not available, a constant `dataobs.cd_std` may be used instead.
The input files can be a text file (.txt), a raw binary (.bin or .dat) or an HDF5 (.h5) file, with its format recognized by the file suffix.
### Other attributes[](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#other-attributes "Link to this heading") 

Ndd: 
    
integer, number of patches down the dip direction 

Nas: 
    
integer, number of patches along the strike direction 

Nmesh: 
    
integer, number of mesh points for each patch, i.e., each patch is divided into Nmesh×Nmesh grids for solving the Eikonal equation 

dsp: 
    
float, the length for each patch, in km 

Nt: 
    
integer, number of time intervals, should be long enough to cover the rupture process 

Npt: 
    
integer, number of mesh points for each time interval 

dt: 
    
float, time unit for each time interval, in second 

t0s: 
    
a list of floats with Npatch elements, initial starting time for each patch, in addition to the fast sweeping calculated arrival time. If configured properly, they can reduce the total number of time intervals needed for the computation.
## Configurations (Joint inversion)[](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#configurations-joint-inversion "Link to this heading")
The configuration for the joint kinematic-static inversion (`kinematicg.pfg`) appears as
```
model = altar.models.seismic.cuda.cascaded
model:
    ; parameters to be simulated (priors)
    ; provide a list at first, serving as their orders in theta
    psets_list = [strikeslip, dipslip, ramp, risetime, rupturevelocity, hypocenter]
    ; define parametersets
    psets:
        ; define the prior for each parameter set
        ; use preset prior to load samples from static inversion for cascading scheme
        ; or use regular priors for non-cascading scheme
        strikeslip = ... ...
        dipslip = ... ...
        ... ...

    ; the model ensemble
    models:
        static = altar.models.seismic.cuda.static
        kinematic = altar.models.seismic.cuda.kinematicg

        static:
            cascaded = True ; or False for non-cascading scheme
            psets_list = [strikeslip, dipslip, ramp]
            ; other static model configurations
            ... ...

        kinematic:
            cascaded = False ; default setting for model
            psets_list = [strikeslip, dipslip, risetime, rupturevelocity, hypocenter]
            ; other kinematic model configurations
            ... ...

```

Here, the main model is a model ensemble `altar.models.seismic.cuda.cascaded`, while its embedded-models `[static, kinematic]` listed as elements of the attribute `models` (a dict).
The parametersets are properties of the main model and are processed by the main model for sample initializations and prior probability computations. Each embedded-model only requires a `psets_list` attribute to extract a sub set of parameters from `model.psets` for its own forward modelling, with the data likelihood computed with respect to its own data observations. The main model collects the data likelihood from all embedded models and assembles them into the Bayesian posterior.
The configuration for each embedded model will be the same as when running it independently, except for an extra flag `cascaded` (default=``False``) to control the cascading scheme.
For the non-cascading scheme with βs=βk=β varying from 0 to 1 simultaneously, set
```
static:
    cascaded=False
kinematic:
    cascaded=False

```

while for the cascading scheme with βs=1, and βk=β varying from 0 to 1, after running the static inversion,
```
static:
    cascaded=True
kinematic:
    cascaded=False

```

## Examples[](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#examples "Link to this heading")
The examples for the joint static and kinematic inversion are available at `9patch` directory.
### Cascading Scheme[](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#cascading-scheme "Link to this heading")
The first step is to run the static inversion only:
```
$ slipmodel --config=static.pfg

```

Please refer to [Static Slip Inversion](https://altar.readthedocs.io/en/cuda/cuda/Static.html#static-inversion) for more details.
The results are saved in the directory `results/static` specified by the config `controller.archiver.output_dir`, which include HDF5 files for all or selected annealing steps. The final step (β=1) results are saved in `step_final.h5`. Copy that file to `9patch` directory so that the final samples of strike/dip slips serve as initial samples for the joint inversion:
```
$ cp results/static/step_final.h5 9patch/theta_cascaded.h5

```

Please also note that the number of chains `job.chains` in the static inversion should be the same or larger than that of the joint inversion so that there are enough samples available.
We now can run the joint static-kinematic inversion,
```
$ slipmodel --config=kinematicg.pfg

```

The results for the jointly inversion will be saved to `results/cascaded`, or any other directory by changing `controller.archiver.output_dir` in `kinematicg.pfg`.
### Non-cascading Scheme[](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#non-cascading-scheme "Link to this heading")
For the non-cascading scheme, you don’t need the step to run static inversion.
You may edit the `kinematicg.pfg` file (or make a copy at first),
>   * change the `static.cascaded` to `False`;
>   * change the `prep` distributions for `strikeslip`, `dipslip`, and `ramp` from `preset` to appropriate distributions, e.g., copying them from `static.pfg` file.
>   * change the output directory `controller.archiver.output_dir` to, e.g., `results/non-cascaded`.
> 

Then run the joint inversion:
```
$ slipmodel --config=kinematicg.pfg

```

In general, the non-cascading takes long iterations to converge and therefore is slower than the cascading scheme.
Please refer to the [AlTar Framework](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html#altar-framework) for the Bayesian MCMC framework options and job/output controls. For example, the Adaptive Metropolis Sampler in general has better performance than the fixed-length Metropolis Sampler, which can be selected by setting `sampler=altar.cuda.bayesian.adapativemetropolis` in the configuration file.
## Forward Model Application (new version)[](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#forward-model-application-new-version "Link to this heading")
It is essentially the same as the static [Forward Model Application](https://altar.readthedocs.io/en/cuda/cuda/Static.html#static-forward-model). The steps are,
  1. prepare a file with a set of parameters in `case` input directory;
  2. add the forward problem settings to the configuration file `kinematicg.pfg` and change the `job` configuration to run with one GPU,


```
; the model
model = altar.models.seismic.cuda.cascaded
model:

    ; settings for running forward problem only
    ; forward theta input
    theta_input = kinematicG_mean_model.txt
    ; forward output file
    forward_output = forward_prediction.h5

... ...

job:
tasks = 1 ; number of tasks per host
gpus = 1  ; number of gpus per task
gpuprecision = float32 ; double(float64) or single(float32) precision for gpu computations
;gpuids = [0] ; a list gpu device ids for tasks on each host, default range(job.gpus)

```

  1. run the plexus command,


```
$ slipmodel.plexus forward --config=kinematicg.pfg

```

  1. the predicted data from both static and kinematic models, as well as the bigM, will be saved to one `forward_prediction.h5` file.


An example is available at 
Note also that the same script can be used for Bayesian simulation, with the commands,
```
$ slipmodel --config=kinematicg.pfg
# or
$ slipmodel.plexus sample --config=kinematicg.pfg

```

See [Forward Model Application](https://altar.readthedocs.io/en/cuda/cuda/Static.html#static-forward-model) for more details.
## Forward Model Application (old version)[](https://altar.readthedocs.io/en/cuda/cuda/Kinematic.html#forward-model-application-old-version "Link to this heading")
Note
This section describes an old implementation, which will be depreciated in the next release.
When analyzing the results, you may need to run the forward model once for the obtained mean-model or any set of parameters, to produce data predictions in comparison with data observations. Since the kinematic forward model is not straightforward, we provide an additional application for running the forward model only, named `kinematicForwardModel`.
An example configuration file is available as `model` configuration copied from `kinematicg.pfg`, with extra settings
```
; theta input
theta_input = kinematicG_synthetic_theta.txt

; output h5 file name
; data prediction is 1d vector with dimension observations
data_output = kinematicG_synthetic_data.h5
; Mb is 1d vector arranged as [Nt][2(strike/dip)][Nas][Ndd] with leading dimensions on the right
mb_output = kinematicG_synthetic_mb.h5

```

where `theta_input` is the input of a mean model or any synthetic model, and `data_out` and `mb_output` are output file names for the data predictions and the big M (you can create an animation from it to observe the rupture process).
The forward model application may be run as
```
$ kinematicForwardModel --config=kinematicg_forward.pfg

```



---

