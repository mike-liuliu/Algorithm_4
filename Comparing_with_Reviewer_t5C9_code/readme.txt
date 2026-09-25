The parallel accelerated version of Algorithm 4 (MMJ distance by Calculation and Copy), is called Algorithm 13 (APPD accelerated by parallel computing).

During the review process of the paper submitted to ICML 2025, a reviewer (referred to as Reviewer t5C9) introduced an alternative algorithm/code for computing the Min-Max-Jump (MMJ) distance matrix. See:
https://openreview.net/forum?id=qNfEkSuGKk

The files in this folder compare several variants of Algorithm 13 with Reviewer t5C9's algorithm/code.

Conclusion summary:

Variant5 of Algorithm 13 is faster than Reviewer t5C9's code in both Python and C++ version. See the files:
-  9_Comparing_Variant5_of_Algorithm_13_kaggle_cpu.ipynb
- 10_Comparing_Variant5_of_Algorithm_13_colab_TPU.ipynb
- 11_Comparing_Variant5_of_Algorithm_13_colab_GPU.ipynb

 