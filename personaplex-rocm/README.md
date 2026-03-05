# PersonaPlex ROCm

## Setup Steps

This document outlines the steps to set up PersonaPlex on Ubuntu 24.04 with ROCm and PyTorch ROCm.

### System Requirements
- Ubuntu 24.04
- ROCm installed according to the official [ROCm installation guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html).

### Installing PyTorch with ROCm
Follow the instructions in the [PyTorch ROCm installation guide](https://pytorch.org/get-started/locally/#start-locally).

### Additional Setup
Ensure that the following dependencies are installed:

- `transformers`
- `safetensors`
- `sounddevice`
- `numpy`