# Results

## Detection summary

| model                   |   mean_fdr |   false_alarm_rate |   n_components |   faults_fdr_over_90 |   median_delay_min |
|:------------------------|-----------:|-------------------:|---------------:|---------------------:|-------------------:|
| DPCA (2 lags)           |      0.671 |              0.006 |             65 |                   11 |               40.5 |
| PCA                     |      0.644 |              0.003 |             31 |                   11 |               33   |
| PCA, theoretical limits |      0.772 |              0.141 |             31 |                   11 |               24   |

## Fault detection rate (%) per fault

| Fault                                                               |   PCA |   DPCA (2 lags) |
|:--------------------------------------------------------------------|------:|----------------:|
| IDV1: A/C feed ratio, B composition constant (step)                 |  99.8 |            99.6 |
| IDV2: B composition, A/C ratio constant (step)                      |  98.6 |            98.5 |
| IDV3: D feed temperature (step)                                     |   0.8 |             1   |
| IDV4: Reactor cooling water inlet temperature (step)                |  97.8 |           100   |
| IDV5: Condenser cooling water inlet temperature (step)              |  24.6 |            27.4 |
| IDV6: A feed loss (step)                                            | 100   |           100   |
| IDV7: C header pressure loss (step)                                 | 100   |           100   |
| IDV8: A, B, C feed composition (random variation)                   |  97.5 |            97.6 |
| IDV9: D feed temperature (random variation)                         |   1.9 |             0.6 |
| IDV10: C feed temperature (random variation)                        |  47.5 |            51.5 |
| IDV11: Reactor cooling water inlet temperature (random variation)   |  64.9 |            85.5 |
| IDV12: Condenser cooling water inlet temperature (random variation) |  98.9 |            99.2 |
| IDV13: Reaction kinetics (slow drift)                               |  95.2 |            95.2 |
| IDV14: Reactor cooling water valve (sticking)                       | 100   |           100   |
| IDV15: Condenser cooling water valve (sticking)                     |   2.1 |             2.2 |
| IDV16: Unknown                                                      |  36.4 |            42.8 |
| IDV17: Unknown                                                      |  92.6 |            96.9 |
| IDV18: Unknown                                                      |  90.5 |            90.1 |
| IDV19: Unknown                                                      |   7.5 |            22.6 |
| IDV20: Unknown                                                      |  51.6 |            55.8 |
| IDV21: Stream 4 valve fixed at steady-state position                |  44   |            43.1 |

## Diagnosis

- **Logistic regression (raw)**: accuracy 0.383, macro-F1 0.412, macro-F1 excl. IDV3/9/15 0.462
- **LightGBM (window features)**: accuracy 0.694, macro-F1 0.751, macro-F1 excl. IDV3/9/15 0.844
