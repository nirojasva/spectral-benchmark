# Experimental Setup

The experiments were conducted on a machine with the following characteristics:
* **CPU:** 13th Gen Intel(R) Core(TM) i7-13620H (16 Cores)
* **RAM:** 8 GB
* **Operating System:** Ubuntu 24.04.2 LTS (WSL2, Kernel 6.6.87.2-microsoft-standard-WSL2)

<br>

---

<br>

# Results for Anomaly Detection in (Spectral) Data Streams


## AUC-PR Results

| Method | DA1 | DA2 | DA3 | SA1 | SA2 | SA3 | TA1 | TA2 | TA3 | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EStorm**<br>(2007) | 0.507<br>- | 0.536<br>- | 0.571<br>- | 0.507<br>- | 0.536<br>- | 0.571<br>- | 0.536<br>- | 0.536<br>- | 0.507<br>- | 0.534<br>- |
| HStree*<br>(2011) | 0.032<br>- | 0.812<br>- | 0.692<br>- | 0.054<br>- | 0.290<br>- | 0.621<br>- | 0.054<br>- | 0.068<br>- | 0.018<br>- | 0.293<br>- |
| IFASD<br>(2013) | 0.976<br>± 5.7e-03 | 0.107<br>± 2.0e-02 | 0.317<br>± 2.8e-02 | 0.251<br>± 2.4e-02 | 0.621<br>± 1.3e-02 | 0.400<br>± 2.4e-03 | 0.087<br>± 4.5e-03 | 0.202<br>± 1.2e-02 | 0.041<br>± 1.9e-03 | 0.333<br>± 1.2e-02 |
| KitNet*<br>(2018) | 0.946<br>- | 0.975<br>- | 0.918<br>- | 0.318<br>- | 0.406<br>- | 0.937<br>- | 0.068<br>- | 0.061<br>- | 0.063<br>- | 0.521<br>- |
| OIF*<br>(2024) | 0.078<br>- | 0.043<br>- | 0.448<br>- | 0.030<br>- | 0.250<br>- | 0.398<br>- | 0.064<br>- | 0.143<br>- | 0.025<br>- | 0.164<br>- |
| RRCF<br>(2016) | 0.127<br>± 3.2e-02 | 0.092<br>± 1.6e-02 | 0.166<br>± 1.7e-02 | 0.157<br>± 5.2e-02 | 0.130<br>± 2.0e-02 | 0.157<br>± 1.6e-02 | 0.082<br>± 4.7e-03 | 0.078<br>± 6.4e-03 | 0.039<br>± 6.5e-03 | 0.114<br>± 1.9e-02 |
| RSHash<br>(2011) | 0.017<br>± 3.7e-05 | 0.118<br>± 6.9e-04 | 0.115<br>± 1.0e-04 | 0.015<br>± 2.9e-05 | 0.420<br>± 3.8e-03 | 0.325<br>± 1.1e-03 | 0.068<br>± 2.3e-04 | 0.056<br>± 5.7e-05 | 0.010<br>± 8.0e-06 | 0.127<br>± 6.7e-04 |

Mean AUC-PR performance (± standard deviation) over 5 runs for each scenario and overall. <br>
*Methods with no variability due to a fixed seed. <br>
**Methods whose score is invariant when anomalies occur.<br>



## VUS-PR Results

| Method | DA1 | DA2 | DA3 | SA1 | SA2 | SA3 | TA1 | TA2 | TA3 | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EStorm**<br>(2007) | 0.021<br>- | 0.105<br>- | 0.210<br>- | 0.021<br>- | 0.105<br>- | 0.210<br>- | 0.105<br>- | 0.105<br>- | 0.019<br>- | 0.100<br>- |
| HStree*<br>(2011) | 0.068<br>- | 0.925<br>- | 0.959<br>- | 0.125<br>- | 0.518<br>- | 0.871<br>- | 0.077<br>- | 0.099<br>- | 0.024<br>- | 0.407<br>- |
| IFASD<br>(2013) | 0.986<br>± 3.7e-03 | 0.162<br>± 1.9e-02 | 0.497<br>± 2.5e-02 | 0.214<br>± 1.7e-02 | 0.658<br>± 1.6e-02 | 0.559<br>± 1.2e-02 | 0.140<br>± 6.3e-03 | 0.261<br>± 1.4e-02 | 0.043<br>± 1.6e-03 | 0.391<br>± 1.3e-02 |
| KitNet<br>(2018) | 0.945<br>- | 0.981<br>- | 0.930<br>- | 0.357<br>- | 0.432<br>- | 0.942<br>- | 0.099<br>- | 0.083<br>- | 0.074<br>- | 0.538<br>- |
| OIF*<br>(2024) | 0.156<br>- | 0.081<br>- | 0.592<br>- | 0.047<br>- | 0.369<br>- | 0.523<br>- | 0.110<br>- | 0.243<br>- | 0.039<br>- | 0.240<br>- |
| RRCF<br>(2016) | 0.100<br>± 2.4e-02 | 0.150<br>± 2.3e-02 | 0.236<br>± 1.3e-02 | 0.129<br>± 4.9e-02 | 0.182<br>± 1.7e-02 | 0.230<br>± 1.7e-02 | 0.125<br>± 4.8e-03 | 0.107<br>± 8.0e-03 | 0.045<br>± 4.4e-03 | 0.145<br>± 1.8e-02 |
| RSHash<br>(2011) | 0.022<br>± 3.3e-05 | 0.177<br>± 6.0e-04 | 0.156<br>± 4.9e-05 | 0.019<br>± 2.1e-05 | 0.573<br>± 1.8e-03 | 0.452<br>± 1.2e-03 | 0.094<br>± 3.6e-04 | 0.068<br>± 8.5e-05 | 0.014<br>± 3.4e-05 | 0.175<br>± 4.7e-04 |

Mean VUS-PR performance (± standard deviation) over 5 runs for each scenario and overall. <br>
*Methods with no variability due to a fixed seed. <br>
**Methods whose score is invariant when anomalies occur.<br>


# Results for Anomaly Detection in (Multivariate) Time Series

Results obtained by applying the code available at https://thedatumorg.github.io/TSB-AD/#leaderboard, with the same default configurations for the methods, to the new spectral datasets (DA1, DA2, DA3, SA1, SA2, SA3, TA1, TA2, TA3). Naive_ZScore is a lightweight anomaly detection baseline that does not require training. It uses basic rolling statistics to compute the maximum Z-score across different features (window size = 50).

## AUC-PR Results

| Method | DA1 | DA2 | DA3 | SA1 | SA2 | SA3 | TA1 | TA2 | TA3 | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CNN | 0.978<br>± 4.7e-04 | 0.994<br>± 6.8e-05 | 0.985<br>± 3.4e-04 | 0.348<br>± 1.0e-02 | 0.896<br>± 2.4e-03 | 0.973<br>± 1.2e-03 | 0.222<br>± 6.3e-02 | 0.210<br>± 1.4e-02 | 0.102<br>± 4.1e-03 | 0.634<br>± 1.1e-02 |
| EIF | 0.963<br>± 1.2e-02 | 0.970<br>± 7.1e-03 | 0.469<br>± 1.7e-01 | 0.125<br>± 6.8e-03 | 0.988<br>± 4.1e-03 | 0.807<br>± 1.1e-01 | 0.070<br>± 2.5e-03 | 0.092<br>± 6.5e-03 | 0.093<br>± 4.3e-03 | 0.509<br>± 3.6e-02 |
| KNN* | 0.041<br>- | 0.057<br>- | 0.076<br>- | 0.008<br>- | 0.050<br>- | 0.080<br>- | 0.106<br>- | 0.083<br>- | 0.007<br>- | 0.057<br>- |
| LOF* | 0.090<br>- | 0.100<br>- | 0.085<br>- | 0.043<br>- | 0.048<br>- | 0.102<br>- | 0.143<br>- | 0.057<br>- | 0.009<br>- | 0.075<br>- |
| LSTMAD | 0.978<br>± 2.7e-04 | 0.994<br>± 6.5e-05 | 0.985<br>± 8.7e-05 | 0.388<br>± 2.8e-02 | 0.895<br>± 9.5e-04 | 0.978<br>± 2.1e-03 | 0.102<br>± 9.4e-04 | 0.159<br>± 2.8e-03 | 0.113<br>± 1.1e-02 | 0.621<br>± 5.1e-03 |
| Naive_ZScore* | 0.008<br>- | 0.044<br>- | 0.100<br>- | 0.012<br>- | 0.065<br>- | 0.116<br>- | 0.060<br>- | 0.054<br>- | 0.011<br>- | 0.052<br>- |

Mean AUC-PR performance (± standard deviation) over 5 runs for each scenario and overall. <br>
*Methods with no variability since they are deterministic. <br>


## VUS-PR Results

| Method | DA1 | DA2 | DA3 | SA1 | SA2 | SA3 | TA1 | TA2 | TA3 | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CNN | 0.989<br>± 4.2e-04 | 0.999<br>± 8.0e-06 | 0.987<br>± 3.4e-04 | 0.285<br>± 2.0e-03 | 0.907<br>± 2.0e-03 | 0.975<br>± 1.1e-03 | 0.255<br>± 6.7e-02 | 0.243<br>± 1.4e-02 | 0.112<br>± 3.6e-03 | 0.639<br>± 1.0e-02 |
| EIF | 0.982<br>± 8.3e-04 | 0.975<br>± 6.5e-03 | 0.474<br>± 1.8e-01 | 0.121<br>± 4.9e-03 | 0.995<br>± 3.3e-03 | 0.808<br>± 1.1e-01 | 0.087<br>± 2.3e-03 | 0.109<br>± 6.8e-03 | 0.120<br>± 3.3e-03 | 0.519<br>± 3.5e-02 |
| KNN* | 0.033<br>- | 0.054<br>- | 0.080<br>- | 0.016<br>- | 0.053<br>- | 0.084<br>- | 0.123<br>- | 0.091<br>- | 0.013<br>- | 0.061<br>- |
| LOF* | 0.058<br>- | 0.103<br>- | 0.088<br>- | 0.035<br>- | 0.053<br>- | 0.102<br>- | 0.171<br>- | 0.063<br>- | 0.016<br>- | 0.077<br>- |
| LSTMAD | 0.990<br>± 4.0e-06 | 0.999<br>± 8.0e-06 | 0.987<br>± 7.0e-05 | 0.344<br>± 4.2e-02 | 0.907<br>± 4.8e-04 | 0.979<br>± 2.0e-03 | 0.125<br>± 1.1e-03 | 0.191<br>± 5.1e-03 | 0.124<br>± 1.3e-02 | 0.627<br>± 7.1e-03 |
| Naive_ZScore* | 0.012<br>- | 0.047<br>- | 0.105<br>- | 0.021<br>- | 0.071<br>- | 0.121<br>- | 0.068<br>- | 0.061<br>- | 0.019<br>- | 0.058<br>- |

Mean VUS-PR performance (± standard deviation) over 5 runs for each scenario and overall. <br>
*Methods with no variability since they are deterministic.  <br>