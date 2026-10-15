| problem | d | PINN energy error | PINN CPU | FD2 to match | FD4 to match | sinc to match | DMC CPU to match |
|---|---|---|---|---|---|---|---|
| oscillator | 1 | 1.2e-04 | 3.0 min | N=192, 3.26e-05 GB, 0.0 s | N=32, 9.66e-06 GB, 0.0 s | N=8, 8.8e-06 GB, 0.0 s | 0.84 h |
| oscillator | 2 | 5.3e-05 | 5.3 min | N=384, 0.0202 GB, 0.7 s | N=40, 0.000249 GB, 0.0 s | N=8, 1.95e-05 GB, 0.0 s | 4.5 h |
| oscillator | 3 | 2.3e-05 | 6.3 min | N=407, 9.17 GB (exact, 1D) | N=48, 0.0152 GB, 0.7 s | N=10, 0.000157 GB, 0.1 s | 13 h |
| oscillator | 4 | 3.4e-06 | 8.1 min | N=957, 1.14e+05 GB (exact, 1D) | N=66, 2.58 GB (exact, 1D) | N=10, 0.00145 GB, 0.1 s | 1.1e+03 h |
| oscillator | 5 | 9.8e-06 | 9.8 min | N=609, 1.14e+07 GB (exact, 1D) | N=52, 51.7 GB (exact, 1D) | N=10, 0.0144 GB, 1.0 s | 84 h |
| oscillator | 6 | 1.4e-05 | 11.4 min | N=524, 2.82e+09 GB (exact, 1D) | N=49, 1.88e+03 GB (exact, 1D) | N=10, 0.144 GB, 9.6 s | 57 h |
| oscillator | 7 | 3.9e-05 | 13.3 min | N=315, 4.19e+10 GB (exact, 1D) | N=38, 1.56e+04 GB (exact, 1D) | N=8, 0.302 GB, 16.3 s | 6.6 h |
| oscillator | 8 | 1.1e-04 | 14.1 min | N=188, 2.12e+11 GB (exact, 1D) | N=29, 6.8e+04 GB (exact, 1D) | N=8, 2.42 GB, 193.9 s | 1 h |
| coupled | 2 | 4.2e-06 | 5.6 min | N=2048, 0.571 GB, 437.6 s | N=160, 0.00362 GB, 0.9 s | N=32, 0.000163 GB, 0.1 s | 1.3e+03 h |
| coupled | 3 | 3.2e-06 | 6.5 min | not reached | N=128, 0.285 GB, 102.2 s | N=24, 0.002 GB, 0.2 s | 1.1e+03 h |
| coupled | 4 | 3.6e-05 | 8.2 min | N=768, 4.73e+04 GB (est.) | N=64, 2.28 GB, 535.6 s | N=16, 0.00945 GB, 1.1 s | 13 h |
| coupled | 5 | 7.0e-05 | 9.8 min | N=512, 4.79e+06 GB (est.) | N=64, 146 GB (est.) | N=14, 0.0775 GB, 7.2 s | 3 h |
| coupled | 6 | 3.9e-04 | 11.7 min | N=192, 6.81e+06 GB (est.) | N=40, 557 GB (est.) | N=12, 0.43 GB, 46.0 s | 0.082 h |
| coupled | 7 | 9.5e-04 | 13.1 min | N=128, 7.66e+07 GB (est.) | N=32, 4.67e+03 GB (est.) | N=10, 1.44 GB, 177.6 s | 0.016 h |
| coupled | 8 | 5.6e+00 (failed) | 14.3 min | - | - | - | - |

Matched on the wavefunction (relative L2 error; nodal for the grids):

| problem | d | PINN L2 error | FD2 to match | FD4 to match | sinc to match |
|---|---|---|---|---|---|
| oscillator | 1 | 2.1e-03 | N=48, 1.18e-05 GB, 0.0 s | N=16, 7.46e-06 GB, 0.0 s | N=8, 8.8e-06 GB, 0.0 s |
| oscillator | 2 | 2.0e-03 | N=48, 0.000356 GB, 0.0 s | N=16, 4.5e-05 GB, 0.0 s | N=8, 1.95e-05 GB, 0.0 s |
| oscillator | 3 | 1.8e-03 | N=64, 0.0358 GB, 1.3 s | N=20, 0.00121 GB, 0.0 s | N=8, 8.63e-05 GB, 0.1 s |
| oscillator | 4 | 1.4e-03 | N=66, 2.58 GB (exact, 1D) | N=20, 0.0218 GB, 1.4 s | N=8, 0.000597 GB, 0.1 s |
| oscillator | 5 | 2.1e-03 | N=55, 68.4 GB (exact, 1D) | N=20, 0.435 GB, 26.3 s | N=8, 0.00473 GB, 0.2 s |
| oscillator | 6 | 4.0e-03 | N=42, 747 GB (exact, 1D) | N=16, 2.28 GB, 221.5 s | N=8, 0.0378 GB, 2.3 s |
| oscillator | 7 | 5.2e-03 | N=38, 1.56e+04 GB (exact, 1D) | N=14, 14.3 GB (exact, 1D) | N=8, 0.302 GB, 16.3 s |
| oscillator | 8 | 9.8e-03 | N=29, 6.8e+04 GB (exact, 1D) | N=12, 58.5 GB (exact, 1D) | N=6, 0.242 GB, 19.3 s |
| coupled | 2 | 3.0e-04 | N=192, 0.00515 GB, 4.3 s | N=64, 0.000629 GB, 0.1 s | N=28, 0.000126 GB, 0.1 s |
| coupled | 3 | 5.1e-04 | N=128, 0.285 GB, 89.1 s | N=40, 0.00883 GB, 2.0 s | N=18, 0.000864 GB, 0.1 s |
| coupled | 4 | 2.9e-03 | N=64, 2.28 GB, 428.3 s | N=24, 0.0451 GB, 5.2 s | N=12, 0.00299 GB, 0.4 s |
| coupled | 5 | 4.3e-03 | N=48, 34.7 GB (est.) | N=20, 0.435 GB, 57.0 s | N=12, 0.0358 GB, 4.3 s |
| coupled | 6 | 1.3e-02 | N=24, 26 GB (est.) | N=16, 2.28 GB, 341.1 s | N=10, 0.144 GB, 16.0 s |
| coupled | 7 | 2.5e-02 | N=24, 624 GB (est.) | N=10, 1.36 GB (est.) | N=6, 0.0403 GB, 3.9 s |
