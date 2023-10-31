In-progress

Files:
* resourceExtraction.ipynb: Tool for creating supervised learning labels based on resource captures
* entropyTest.py: Kernel density estimation tool
* gaussian.ipynb: Taking single resource values and finding variance as an approximation of fuzzer ability
* modelTraining.ipynb: Train a prediction model used for anomaly detection comparisons

Artifacts for the fuzzing metric paper

Network resource capturing assumes the ability to use tcpdump.
Currently /etc/sudoers:

USER ALL=(ALL) NOPASSWD: /usr/sbin/tcpdump
