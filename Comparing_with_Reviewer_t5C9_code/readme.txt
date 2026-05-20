The parallel accelerated version of Algorithm 4 (MMJ distance by Calculation and Copy), is called Algorithm 13 (APPD accelerated by parallel computing).

During the review process of the paper submitted to ICML 2025, a reviewer (referred to as Reviewer t5C9) introduced an alternative algorithm/code for computing the Min-Max-Jump (MMJ) distance matrix. See:
https://openreview.net/forum?id=qNfEkSuGKk

The files in this folder compare several variants of Algorithm 13 with Reviewer t5C9's algorithm/code.

Conclusion summary:

Algorithm 13 appears to be slower than the Python version of Reviewer t5C9's code (which is accelerated by Numba), but faster than its C++ version. However, Reviewer t5C9’s code/algorithm requires significantly more memory. For example, under the current 30 GB memory limit on Kaggle (as of July 2025):
- The Python version of Reviewer t5C9’s code can handle graphs with up to ~36,000 nodes.
- The C++ version can process graphs with up to ~38,000 nodes.
- In contrast, Algorithm 13 can handle graphs with up to ~44,000 nodes.

 