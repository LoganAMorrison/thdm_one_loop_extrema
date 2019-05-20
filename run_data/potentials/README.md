# Potentials Data
The data in this directory is for plotting potential along a line connecting
the deepest normal and charge-breaking minima in the case where the 
charge-breaking minimum is the deepest. The ordering of the data matches the
order of the data in the `type_a1.csv` data file.

## Connecting the Vacuua
Let the vacuua be labeled by `phi_n` and `phi_cb` for the normal and 
charge-breaking  VEVs. Both `phi_n` and `phi_cb` are three-vectors. We 
interpolate between the VEVs using:
```$xslt
phi(t) = phi_n * t + phi_cb * (1-t)
```
where 0 <= t <= 1.