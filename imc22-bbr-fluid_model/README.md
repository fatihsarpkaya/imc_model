# Fluid-Model Simulator + Mininet Experiment Suite

The following code was used in the paper 'Model-Based Insights on the Performance, Fairness, and Stability of BBR' by Simon Scherrer, Markus Legner, Adrian Perrig, and Stefan Schmid, published at the ACM Internet Measurement Conference (IMC) 2022.

The authors of the code are Simon Scherrer and James Dermelj.  

## Prerequisites

- Install [Mininet](http://mininet.org/download/)
- Download [POX](https://noxrepo.github.io/pox-doc/html/#installing-pox) and change the configuration option `pox_directory` in the configuration files under `configs/` to the POX directory on the local machine
- Ensure that you have sudo rights on the target machine, as Mininet must be run as root.
- Install the other depencies via `./install_dependencies.sh` (both as root and as normal user).
- To install BBRv2 on an Ubuntu system, the easiest way is to install the the [Liquorix kernel](https://liquorix.net/#install). You may need Ubuntu 21.04 for it.

## Run the code

Create a configuration file (from the pattern of `configs/test_config.yml`) specifying: 
- the parameter ranges that should be tested (The scripts will create the Cartesian product of them)
- the plots that should be created from the resulting data

Then:
```
# Run the fluid model
./run_model.py -c configs/test_config.yml

# Run the Mininet experiments
sudo ./run_experiments.py configs/test_config.yml

# Generate the plots for the traces
./plot_trace.py configs/test_config.py

# Generate the plots from the aggregated results
./plot.py configs/test_config.py
```

Finally, the results can be found under `results/[name]`, where `[name]` is specified in the used configuration file.

## Licensing

Copyright 2023 ETH Zurich

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
