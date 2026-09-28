# Data

The full table (126,153,720 rows x 21 columns, 15.9 GB) is **not** in this
repository. It is regenerated from the S175 WeatherRouting simulator.

## Regenerating

1. Obtain the simulator (`WeatherRouting.linux.bin`) and `S175_model_data/`.
2. Write the factorial grid into `S175_model_data/MetamodInputParams.dat`.
   Values used for the published dataset:

   | Parameter | Values | n |
   |---|---|---|
   | Draft (m) | 8.0, 8.5, 9.0 | 3 |
   | Trim (m) | -0.75, 0.00, 0.25 | 3 |
   | EngRPM (1/min) | 122.4 : 7.2 : 144.0 | 4 |
   | P/D | 0.7 : 0.1 : 1.1 | 5 |
   | Hs (m) | 0 : 0.25 : 10 | 41 |
   | Tp (s) | 4.49 … 18.50 | 11 |
   | Chi (deg) | 0 : 5 : 180 | 37 |
   | Vwind (m/s) | 0 : 5 : 25 | 6 |
   | Theta_wind (deg) | 0 : 30 : 180 | 7 |

   GMt is held at 0.55 m and shaft generator power at 0 kW.
   Full factorial: 3 x 3 x 4 x 5 x 41 x 11 x 37 x 6 x 7 = 126,153,720.

3. Run `S175createmetaindices.sh` and keep `output.txt`.
4. Verify: `sha256sum` must match `dataset.sha256`.

## Infeasibility encoding

The simulator writes -1 for an infeasible operating point, in two modes:
all ten outputs (18.1% of rows), or fuel consumption alone (14.7%). The
fuel-only share rises with sea state, from 5.9% at Hs = 0 to 25.0% at Hs >= 8 m.

## Sample

`sample/` holds a small committed extract so `make smoke` runs the whole
pipeline without the full dataset. It is for testing the code path only and is
not statistically representative.
