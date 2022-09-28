# LoREPorTS
A pore-throat segmentation method based on local resistance equivalence for pore-network modeling
The objective of the present program is to assign a boundary between a pore and a throat, that is to say, to specify which voxels constitute a pore or throat.
Based on this pore-throat segmentation, conduit lengths and volumes of network elements are defined and calculated.

Input: A binary image in which True for pore structures and False for solid
Output: A dictionary containing all the extracted network properties in OpenPNM format.
