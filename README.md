# Group-Sparse Subspace Clustering with Elastic Stars

This repository contains the implementation of the Group-Sparse Subspace Clustering (SSC) method with a novel regularization approach named Elastic Stars (ES), as described in the paper *"Group-Sparse Subspace Clustering with Elastic Stars"* by Huanran Li and Daniel Pimentel-Alarcón.

## Overview

Sparse Subspace Clustering (SSC) is a powerful method in data analysis, particularly in high-dimensional datasets. However, SSC often struggles with over-sparsification due to the L1-norm penalty, leading to sub-optimal clustering results. This repository addresses these challenges by introducing the Elastic Stars (ES) regularization, which balances sparsity and connectivity within clusters, leading to improved clustering accuracy, particularly in challenging datasets such as Hyperspectral Imaging (HSI).

### Key Contributions:

- **Elastic Stars (ES) Regularization**: A novel sparsity regularization approach that minimizes the distance between variables and dynamically evolving centroids, ensuring balanced representation matrices and enhanced cluster connectivity.
- **ADMM Optimization**: An effective solution to the non-convex and non-smooth nature of the ES regularization using the Alternating Direction Method of Multipliers (ADMM).
- **Empirical Validation**: Demonstrated significant improvements in clustering accuracy on both synthetic data and real-world Hyperspectral Imaging datasets compared to existing methods like SSC, SSC-OMP, and EnSC.

## Implementation Details

The script provided in this repository performs the following steps:

1. **Data Initialization**: Generates low-rank matrices to simulate different subspaces.
2. **Subspace Clustering**:
   - **SSC**: Traditional Sparse Subspace Clustering.
   - **SSC_EN**: Elastic Net regularized SSC.
   - **MDSP-EN (Elastic Stars)**: Our proposed method with Elastic Stars regularization.
3. **Graph Analysis**: Evaluates the connectivity and quality of the resulting affinity matrices.
4. **Performance Evaluation**: Computes the mean and standard deviation of clustering results and compares the performance across different methods.
5. **Results Storage**: Saves all results and computation times into a `.npz` file for further analysis.

## Dependencies

To run the code, you need the following Python packages:

- `numpy`
- `matplotlib`
- `cvxpy`
- `seaborn`
- `scipy`
- `sklearn`
- `networkx`

Install them using pip:

```bash
pip install numpy matplotlib cvxpy seaborn scipy scikit-learn networkx
```

## Usage

To execute the script and perform the subspace clustering analysis:

```bash
python script_name.py
```

Replace `script_name.py` with the actual name of the script.

## Files in This Repository

- **`Init.py`**: Functions for initializing low-rank matrices.
- **`plot.py`**: Plotting utilities for visualizing results.
- **`ES_SSC.py`**: Implementation of Elastic Stars regularized SSC.
- **`SSC.py`**: Implementation of traditional SSC.
- **`EnSC.py`**: Implementation of Elastic Net SSC.

## Experimental Results

The results, including the analyzed graphs and computation times, are saved in `results_and_times.npz`. You can load and explore these results using:

```python
import numpy as np

data = np.load('results_and_times.npz')
print(data.files)
```

## Citation

If you use this code, please cite the following paper:

```
@inproceedings{Li2024ElasticStars,
  title={Group-Sparse Subspace Clustering with Elastic Stars},
  author={Huanran Li and Daniel Pimentel-Alarcón},
  year={2024},
  booktitle={IEEE International Symposium on Information Theory (ISIT)},
  organization={IEEE},
  url={https://ieeexplore.ieee.org/document/10619557}
}
```

You can access the paper [here](https://ieeexplore.ieee.org/document/10619557).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This work is a product of collaboration between the Department of Electrical Engineering and the Wisconsin Institute of Discovery at the University of Wisconsin-Madison. The project was supported by research grants and resources provided by the university.
