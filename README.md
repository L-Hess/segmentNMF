# segmentNMF
---
A non-negative matrix factorization algorithm for demixing calcium imaging data.
The user can specify initial values for the structural and temporal components;
in particular structural components can be initialized with an anatomical segmentation.
Distributed functions are available for large datasets, small crops defined by the
initial segmentation are run in parallel on distributed hardware.

## Example results - crops from a whole brain dataset
---
Upper left: raw data\
Upper right: mean of raw data over time\
Lower left: NMF reconstruction (space\_components @ time\_components)\
Lower right: residual (raw - reconstruction)

![crop0](resources/crop0.gif)
![crop1](resources/crop1.gif)

![crop2](resources/crop2.gif)
![crop3](resources/crop3.gif)

## Installation
---
Currently only from source but will be on PyPI in the near future.


