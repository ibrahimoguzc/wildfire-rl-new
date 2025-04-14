==========================
Fire Propagation Modelling
==========================

CUDA Implementation
-------------------
The selected fire-model has great potential for parallelization as the state of
the fire at the next iteration depends only on local quantities at the
current moment of time. Therefore, use of Graphic Processing Units (GPUs)
can provide increased performance due to their knack for performing massively
parallel computations on arrays. As the Numba Python package provides
an easy-to-use interface to Nvidia CUDA programming, using Numba once again
reduces the dependencies of the software.

In CUDA functions are referred to as kernels and the hardware handles
execution of this kernel (function) across the array. An array is subdivided
into blocks, which each have a set of threads or workers which simultaneously
execute the kernel function in parallel. Therefore, unlike the nested for-loop
implementation in the CPU fire-model, on the GPU the fire-propagation is
solved for the entire array at once.

Picture of subdivision w/ nvidia

Table with device = gpu, host = cpu. (Nvidia talk)

.. _fire_process:
.. uml:: ../../plantuml/src/firemodel.wsd
   :caption: Activity Diagram of the Cellular Automata Fire Model
   :scale: 50 %
   :align: center

Fire States
===========
.. automodule:: examples.wildfire.fire_model.states
   :members:
   :member-order: "bysource"

GPU Step
========
.. automodule:: examples.wildfire.fire_model.gpu.step
   :members:
   :private-members:
   :special-members:

.. autofunction:: examples.wildfire.fire_model.gpu.step.step

