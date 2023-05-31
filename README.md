# FaultDiagnosisAI
This repository contains code for implementing the method described in the paper “Robust and Efficient Fault Diagnosis of mm-Wave Active Phased Arrays Using Baseband Signal” 1.

# Overview
One key communication block in 5G and 6G radios is the active phased array (APA). To ensure reliable operation, efficient and timely fault diagnosis of APAs on-site is crucial. To date, fault diagnosis has relied on measurement of frequency domain radiation patterns using costly equipment and multiple strictly controlled measurement probes, which are time consuming, complex, and therefore infeasible for on-site deployment 1.

This paper proposes a novel method exploiting a deep neural network (DNN) tailored to extract the features hidden in the baseband in-phase and quadrature signals for classifying the different faults. It requires only a single probe in one measurement point for fast and accurate diagnosis of the faulty elements and components in APAs 1.

Validation of the proposed method is done using a commercial 28 GHz APA. Accuracies of 99% and 80% have been demonstrated for single- and multi-element failure detection, respectively. Three different test scenarios are investigated: ON-OFF antenna elements, phase variations, and magnitude attenuation variations. In a low signal-to-noise ratio (SNR) of 4 dB, stable fault detection accuracy above 90% is maintained. This is all achieved with a detection time of milliseconds (e.g., 6 ms), showing a high potential for on-site deployment 1.

# Authors
- Martin Hedegaard Nielsen
- Yufeng Zhang
- Changbin Xue
- Jian Ren
- Yingzeng Yin
- Ming Shen
- Gert Frølund Pedersen

# License
This project is licensed under the MIT License - see the LICENSE file for details

# How to run the project
First use the process_data.py to generate the needed data
Then run the train model file dependent on which model is wanted
Finally run the test_model.py to generate the test results shown in the paper 

# To do: 
Generate a script that does everything and is controlled 

# Citation

If you use this code or find it helpful in your research, please consider citing the following paper:

@ARTICLE{9794293,
  author={Nielsen, Martin H. and Zhang, Yufeng and Xue, Changbin and Ren, Jian and Yin, Yingzeng and Shen, Ming and Pedersen, Gert Frølund},
  journal={IEEE Transactions on Antennas and Propagation}, 
  title={Robust and Efficient Fault Diagnosis of mm-Wave Active Phased Arrays Using Baseband Signal}, 
  year={2022},
  volume={70},
  number={7},
  pages={5044-5053},
  doi={10.1109/TAP.2022.3179898}}

