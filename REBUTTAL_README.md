# Experimental Setup

The experiments were conducted on a machine with the following characteristics for all methods except MMPAD:
* **CPU:** 13th Gen Intel(R) Core(TM) i7-13620H (10 Cores, 16 Threads)
* **RAM:** 8 GB
* **Operating System:** Ubuntu 24.04.2 LTS (WSL2, Kernel 6.6.87.2-microsoft-standard-WSL2)

The characteristics of the machine for MMPAD tecnique were:
 * **CPU:** Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz (12 Cores, 24 Threads)
* **RAM:** 62 GB
* **Operating System:** Debian GNU/Linux 12 (bookworm) 

---

# Results for Anomaly Detection in (Spectral) Data Streams

## AUC-PR Results

**Table 1.** AUC-PR performance for anomaly detection in spectral data streams.

| Method | DA1 | DA2 | DA3 | SA1 | SA2 | SA3 | TA1 | TA2 | TA3 | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EStorm**<br>(2007) | 0.507<br>- | 0.536<br>- | 0.571<br>- | 0.507<br>- | 0.536<br>- | 0.571<br>- | 0.536<br>- | 0.536<br>- | 0.507<br>- | 0.534<br>- |
| HStree*<br>(2011) | 0.032<br>- | 0.812<br>- | 0.692<br>- | 0.054<br>- | 0.290<br>- | 0.621<br>- | 0.054<br>- | 0.068<br>- | 0.018<br>- | 0.293<br>- |
| IFASD<br>(2013) | 0.976<br>± 5.7e-03 | 0.107<br>± 2.0e-02 | 0.317<br>± 2.8e-02 | 0.251<br>± 2.4e-02 | 0.621<br>± 1.3e-02 | 0.400<br>± 2.4e-03 | 0.087<br>± 4.5e-03 | 0.202<br>± 1.2e-02 | 0.041<br>± 1.9e-03 | 0.333<br>± 1.2e-02 |
| KitNet*<br>(2018) | 0.946<br>- | 0.975<br>- | 0.918<br>- | 0.318<br>- | 0.406<br>- | 0.937<br>- | 0.068<br>- | 0.061<br>- | 0.063<br>- | 0.521<br>- |
| OIF*<br>(2024) | 0.078<br>- | 0.043<br>- | 0.448<br>- | 0.030<br>- | 0.250<br>- | 0.398<br>- | 0.064<br>- | 0.143<br>- | 0.025<br>- | 0.164<br>- |
| RRCF<br>(2016) | 0.127<br>± 3.2e-02 | 0.092<br>± 1.6e-02 | 0.166<br>± 1.7e-02 | 0.157<br>± 5.2e-02 | 0.130<br>± 2.0e-02 | 0.157<br>± 1.6e-02 | 0.082<br>± 4.7e-03 | 0.078<br>± 6.4e-03 | 0.039<br>± 6.5e-03 | 0.114<br>± 1.9e-02 |
| RSHash<br>(2011) | 0.017<br>± 3.7e-05 | 0.118<br>± 6.9e-04 | 0.115<br>± 1.0e-04 | 0.015<br>± 2.9e-05 | 0.420<br>± 3.8e-03 | 0.325<br>± 1.1e-03 | 0.068<br>± 2.3e-04 | 0.056<br>± 5.7e-05 | 0.010<br>± 8.0e-06 | 0.127<br>± 6.7e-04 |

*Mean AUC-PR performance (± standard deviation) over 5 runs for each scenario and overall.*  
*\*Methods with no variability due to a fixed seed.*  
*\*\*Methods whose score is invariant when anomalies occur.*

<br>

## VUS-PR Results

**Table 2.** VUS-PR performance for anomaly detection in spectral data streams.

| Method | DA1 | DA2 | DA3 | SA1 | SA2 | SA3 | TA1 | TA2 | TA3 | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EStorm**<br>(2007) | 0.021<br>- | 0.105<br>- | 0.210<br>- | 0.021<br>- | 0.105<br>- | 0.210<br>- | 0.105<br>- | 0.105<br>- | 0.019<br>- | 0.100<br>- |
| HStree*<br>(2011) | 0.068<br>- | 0.925<br>- | 0.959<br>- | 0.125<br>- | 0.518<br>- | 0.871<br>- | 0.077<br>- | 0.099<br>- | 0.024<br>- | 0.407<br>- |
| IFASD<br>(2013) | 0.986<br>± 3.7e-03 | 0.162<br>± 1.9e-02 | 0.497<br>± 2.5e-02 | 0.214<br>± 1.7e-02 | 0.658<br>± 1.6e-02 | 0.559<br>± 1.2e-02 | 0.140<br>± 6.3e-03 | 0.261<br>± 1.4e-02 | 0.043<br>± 1.6e-03 | 0.391<br>± 1.3e-02 |
| KitNet<br>(2018) | 0.945<br>- | 0.981<br>- | 0.930<br>- | 0.357<br>- | 0.432<br>- | 0.942<br>- | 0.099<br>- | 0.083<br>- | 0.074<br>- | 0.538<br>- |
| OIF*<br>(2024) | 0.156<br>- | 0.081<br>- | 0.592<br>- | 0.047<br>- | 0.369<br>- | 0.523<br>- | 0.110<br>- | 0.243<br>- | 0.039<br>- | 0.240<br>- |
| RRCF<br>(2016) | 0.100<br>± 2.4e-02 | 0.150<br>± 2.3e-02 | 0.236<br>± 1.3e-02 | 0.129<br>± 4.9e-02 | 0.182<br>± 1.7e-02 | 0.230<br>± 1.7e-02 | 0.125<br>± 4.8e-03 | 0.107<br>± 8.0e-03 | 0.045<br>± 4.4e-03 | 0.145<br>± 1.8e-02 |
| RSHash<br>(2011) | 0.022<br>± 3.3e-05 | 0.177<br>± 6.0e-04 | 0.156<br>± 4.9e-05 | 0.019<br>± 2.1e-05 | 0.573<br>± 1.8e-03 | 0.452<br>± 1.2e-03 | 0.094<br>± 3.6e-04 | 0.068<br>± 8.5e-05 | 0.014<br>± 3.4e-05 | 0.175<br>± 4.7e-04 |

*Mean VUS-PR performance (± standard deviation) over 5 runs for each scenario and overall.*  
*\*Methods with no variability due to a fixed seed.*  
*\*\*Methods whose score is invariant when anomalies occur.*

---

# Results for Anomaly Detection in (Multivariate) Time Series

Results obtained by applying the code available at [TSB-AD](https://thedatumorg.github.io/TSB-AD/#leaderboard), with the same default configurations for the used methods, to the new spectral datasets (DA1, DA2, DA3, SA1, SA2, SA3, TA1, TA2, TA3), split into 40% for training and 60% for testing for semi-supervised methods like LSTMAD and CNN.

> **Note:** Naive_ZScore is a lightweight anomaly detection baseline that does not require training. It uses basic rolling statistics to compute the maximum Z-score across different features (window size = 50).

## AUC-PR Results

**Table 3.** AUC-PR performance for anomaly detection in time series.

| Method | DA1 | DA2 | DA3 | SA1 | SA2 | SA3 | TA1 | TA2 | TA3 | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CNN | 0.978<br>± 4.7e-04 | 0.994<br>± 6.8e-05 | 0.985<br>± 3.4e-04 | 0.348<br>± 1.0e-02 | 0.896<br>± 2.4e-03 | 0.973<br>± 1.2e-03 | 0.222<br>± 6.3e-02 | 0.210<br>± 1.4e-02 | 0.102<br>± 4.1e-03 | 0.634<br>± 1.1e-02 |
| EIF | 0.963<br>± 1.2e-02 | 0.970<br>± 7.1e-03 | 0.469<br>± 1.7e-01 | 0.125<br>± 6.8e-03 | 0.988<br>± 4.1e-03 | 0.807<br>± 1.1e-01 | 0.070<br>± 2.5e-03 | 0.092<br>± 6.5e-03 | 0.093<br>± 4.3e-03 | 0.509<br>± 3.6e-02 |
| KNN* | 0.041<br>- | 0.057<br>- | 0.076<br>- | 0.008<br>- | 0.050<br>- | 0.080<br>- | 0.106<br>- | 0.083<br>- | 0.007<br>- | 0.057<br>- |
| LOF* | 0.090<br>- | 0.100<br>- | 0.085<br>- | 0.043<br>- | 0.048<br>- | 0.102<br>- | 0.143<br>- | 0.057<br>- | 0.009<br>- | 0.075<br>- |
| LSTMAD | 0.978<br>± 2.7e-04 | 0.994<br>± 6.5e-05 | 0.985<br>± 8.7e-05 | 0.388<br>± 2.8e-02 | 0.895<br>± 9.5e-04 | 0.978<br>± 2.1e-03 | 0.102<br>± 9.4e-04 | 0.159<br>± 2.8e-03 | 0.113<br>± 1.1e-02 | 0.621<br>± 5.1e-03 |
| MMPAD* | 0.591<br>- | 0.373<br>- | 0.786<br>- | 0.013<br>- | 0.163<br>- | 0.860<br>- | 0.044<br>- | 0.135<br>- | 0.023<br>- | 0.332<br>- |
| Naive_ZScore* | 0.008<br>- | 0.044<br>- | 0.100<br>- | 0.012<br>- | 0.065<br>- | 0.116<br>- | 0.060<br>- | 0.054<br>- | 0.011<br>- | 0.052<br>- |
| StreamVAE | 0.925<br>± 3.0e-03 | 0.349<br>±4.1e-05 | 0.851<br>±1.8e-05 | 0.939<br>±0.0e-00 | 0.349<br>±0.0e-00 | 0.420<br>±4.1e-07 | 0.192<br>±2.6e-02 | 0.356<br>±1.3e-01 | 0.071<br>±4.9e-03 | 0.495<br>±4.9e-03 |

*Mean AUC-PR performance (± standard deviation) over 5 runs for each scenario and overall.*  
*\*Methods with no variability since they are deterministic.*

<br>

## VUS-PR Results

**Table 4.** VUS-PR performance for anomaly detection in time series.

| Method | DA1 | DA2 | DA3 | SA1 | SA2 | SA3 | TA1 | TA2 | TA3 | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CNN | 0.989<br>± 4.2e-04 | 0.999<br>± 8.0e-06 | 0.987<br>± 3.4e-04 | 0.285<br>± 2.0e-03 | 0.907<br>± 2.0e-03 | 0.975<br>± 1.1e-03 | 0.255<br>± 6.7e-02 | 0.243<br>± 1.4e-02 | 0.112<br>± 3.6e-03 | 0.639<br>± 1.0e-02 |
| EIF | 0.982<br>± 8.3e-04 | 0.975<br>± 6.5e-03 | 0.474<br>± 1.8e-01 | 0.121<br>± 4.9e-03 | 0.995<br>± 3.3e-03 | 0.808<br>± 1.1e-01 | 0.087<br>± 2.3e-03 | 0.109<br>± 6.8e-03 | 0.120<br>± 3.3e-03 | 0.519<br>± 3.5e-02 |
| KNN* | 0.033<br>- | 0.054<br>- | 0.080<br>- | 0.016<br>- | 0.053<br>- | 0.084<br>- | 0.123<br>- | 0.091<br>- | 0.013<br>- | 0.061<br>- |
| LOF* | 0.058<br>- | 0.103<br>- | 0.088<br>- | 0.035<br>- | 0.053<br>- | 0.102<br>- | 0.171<br>- | 0.063<br>- | 0.016<br>- | 0.077<br>- |
| LSTMAD | 0.990<br>± 4.0e-06 | 0.999<br>± 8.0e-06 | 0.987<br>± 7.0e-05 | 0.344<br>± 4.2e-02 | 0.907<br>± 4.8e-04 | 0.979<br>± 2.0e-03 | 0.125<br>± 1.1e-03 | 0.191<br>± 5.1e-03 | 0.124<br>± 1.3e-02 | 0.627<br>± 7.1e-03 |
| MMPAD* | 0.899<br>- | 0.469<br>- | 0.811<br>- | 0.017<br>- | 0.165<br>- | 0.864<br>- | 0.049<br>- | 0.164<br>- | 0.031<br>- | 0.385<br>- |
| Naive_ZScore* | 0.012<br>- | 0.047<br>- | 0.105<br>- | 0.021<br>- | 0.071<br>- | 0.121<br>- | 0.068<br>- | 0.061<br>- | 0.019<br>- | 0.058<br>- |
| StreamVAE | 0.984<br>± 3.5e-04 | 0.382<br>±2.2e-05 | 0.919<br>±3.6e-05 | 0.987<br>±0.0e-00 | 0.382<br>±0.0e-00 | 0.487<br>±0.0e-00 | 0.210<br>±3.4e-02 | 0.386<br>±1.2e-01 | 0.093<br>±1.1e-02 | 0.537<br>±2.0e-02 |


*Mean VUS-PR performance (± standard deviation) over 5 runs for each scenario and overall.*  
*\*Methods with no variability since they are deterministic.*


---

# Correlation Analysis Among Wavelengths

To assess feature correlation in spectral data, we computed pairwise Pearson correlation matrices between wavelength intervals but also individual spectral wavelengths for **DA3** (drift scenario with two connected chambers). Therefore, rhe data was analyzed using the complete spectrum (2048 wavelengths) as well as reduced resolutions of 1024, 512, 256, and 12 intervals. To analyse intervals, we used the maximum intensity.

## Scenario DA3

**Table 5.** Correlation at Different Resolutions, Scenario DA3.
| n_intervals | Total | Not correlated (< 0.5) | Correlated (>= 0.5) | % of Correlated (>= 0.5) |
|---|---|---|---|---|
| None | 2048 | 2048 | 0 | 0.0 |
| 1024 | 1024 | 925 | 99 | 0.0966796875 |
| 512 | 512 | 352 | 160 | 0.3125 |
| 256 | 256 | 119 | 137 | 0.53515625 |
| 12 | 12 | 1 | 11 | 0.9166666666666666 |


**Table 6.** Correlation matrix (Resolution 12 intervals), Scenario DA3.
| Interval | 189.81-254.07 | 254.45-317.81 | 318.18-380.59 | 380.95-442.32 | 442.67-502.93 | 503.28-562.36 | 562.7-620.52 | 620.86-677.35 | 677.68-732.45 | 732.77-786.09 | 786.4-838.19 | 838.49-888.68 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **189.81-254.07** | 1.00 | 0.02 | 0.02 | 0.01 | 0.00 | 0.02 | 0.00 | -0.01 | 0.01 | 0.00 | 0.00 | 0.01 |
| **254.45-317.81** | 0.02 | 1.00 | 0.93 | 0.70 | 0.82 | 0.88 | 0.78 | 0.55 | 0.57 | 0.45 | 0.49 | 0.54 |
| **318.18-380.59** | 0.02 | 0.93 | 1.00 | 0.71 | 0.63 | 0.86 | 0.73 | 0.27 | 0.61 | 0.55 | 0.50 | 0.55 |
| **380.95-442.32** | 0.01 | 0.70 | 0.71 | 1.00 | 0.66 | 0.79 | 0.78 | 0.43 | 0.41 | 0.35 | 0.50 | 0.47 |
| **442.67-502.93** | 0.00 | 0.82 | 0.63 | 0.66 | 1.00 | 0.82 | 0.82 | 0.90 | 0.36 | 0.24 | 0.46 | 0.45 |
| **503.28-562.36** | 0.02 | 0.88 | 0.86 | 0.79 | 0.82 | 1.00 | 0.89 | 0.56 | 0.51 | 0.47 | 0.57 | 0.56 |
| **562.7-620.52** | 0.00 | 0.78 | 0.73 | 0.78 | 0.82 | 0.89 | 1.00 | 0.62 | 0.42 | 0.39 | 0.55 | 0.51 |
| **620.86-677.35** | -0.01 | 0.55 | 0.27 | 0.43 | 0.90 | 0.56 | 0.62 | 1.00 | 0.14 | -0.02 | 0.28 | 0.25 |
| **677.68-732.45** | 0.01 | 0.57 | 0.61 | 0.41 | 0.36 | 0.51 | 0.42 | 0.14 | 1.00 | 0.33 | 0.29 | 0.32 |
| **732.77-786.09** | 0.00 | 0.45 | 0.55 | 0.35 | 0.24 | 0.47 | 0.39 | -0.02 | 0.33 | 1.00 | 0.31 | 0.35 |
| **786.4-838.19** | 0.00 | 0.49 | 0.50 | 0.50 | 0.46 | 0.57 | 0.55 | 0.28 | 0.29 | 0.31 | 1.00 | 0.37 |
| **838.49-888.68** | 0.01 | 0.54 | 0.55 | 0.47 | 0.45 | 0.56 | 0.51 | 0.25 | 0.32 | 0.35 | 0.37 | 1.00 |