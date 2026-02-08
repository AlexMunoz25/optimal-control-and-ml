# Optimal Control and Machine Learning — Jupyter Learning Hub

Control and ML notebooks, covering mathematical foundations, dynamic systems, optimization, optimal control, and machine learning.

## Why This Repository

I learn best by building things. If these notes help someone else along the way, even better ...

---

## Repository Structure So Far

```
optimal-control-and-ml/
│
├── README.md
├── requirements.txt
│
├── 01_Foundations/
│   ├── 01_Linear_Algebra/                                    ← 155 notebooks
│   │   ├── 01_Basic_Definitions/                                       (10 notebooks)
│   │   │   ├── 01_sets.ipynb
│   │   │   ├── 02_cartesian_plane.ipynb
│   │   │   ├── 03_vector.ipynb
│   │   │   ├── 04_euclidean_space.ipynb
│   │   │   ├── 05_basis.ipynb
│   │   │   ├── 06_matrix.ipynb
│   │   │   ├── 07_linearity.ipynb
│   │   │   ├── 08_change_of_basis.ipynb
│   │   │   ├── 09_projections.ipynb
│   │   │   └── 10_coordinate_projections.ipynb
│   │   │
│   │   ├── 02_Vector/                                            (6 notebooks)
│   │   │   ├── 01_vector_addition.ipynb
│   │   │   ├── 02_vector_scalar_multiplication.ipynb
│   │   │   ├── 03_vector_dot_product.ipynb
│   │   │   ├── 04_vector_cross_product_3d.ipynb
│   │   │   ├── 05_vector_norm.ipynb
│   │   │   └── 06_vector_orthogonality_condition.ipynb
│   │   │
│   │   ├── 03_Matrix/                                            (19 notebooks)
│   │   │   ├── 01_matrix_addition.ipynb
│   │   │   ├── 02_matrix_scalar_multiplication.ipynb
│   │   │   ├── 03_matrix_vector_multiplication.ipynb
│   │   │   ├── 04_matrix_multiplication.ipynb
│   │   │   ├── 05_matrix_transpose.ipynb
│   │   │   ├── 06_matrix_determinant_2d.ipynb
│   │   │   ├── 07_matrix_inverse_2d.ipynb
│   │   │   ├── 08_matrix_cramers_rule.ipynb
│   │   │   ├── 09_matrix_determinant.ipynb
│   │   │   ├── 10_matrix_inverse.ipynb
│   │   │   ├── 11_matrix_rank_nullity.ipynb
│   │   │   ├── 12_matrix_elementwise_product.ipynb
│   │   │   ├── 13_matrix_outer_product.ipynb
│   │   │   ├── 14_matrix_frobenius_norm.ipynb
│   │   │   ├── 15_matrix_norm_inequality.ipynb
│   │   │   ├── 16_matrix_trace.ipynb
│   │   │   ├── 17_matrix_trace_of_product.ipynb
│   │   │   ├── 18_matrix_block_multiplication.ipynb
│   │   │   └── 19_matrix_kronecker_product.ipynb
│   │   │
│   │   ├── 04_Computational_Linear_Algebra/                      (6 notebooks)
│   │   │   ├── 01_augmented_matrix.ipynb
│   │   │   ├── 02_row_operations.ipynb
│   │   │   ├── 03_reduced_row_echelon_form.ipynb
│   │   │   ├── 04_gauss_jordan_elimination.ipynb
│   │   │   ├── 05_number_of_solutions.ipynb
│   │   │   └── 06_matrix_equations.ipynb
│   │   │
│   │   ├── 05_Geometrical_Aspects_of_Linear_Algebra/            (26 notebooks)
│   │   │   ├── 01_lines_and_planes/
│   │   │   │   ├── 01_lines_and_planes.ipynb
│   │   │   │   ├── 02_lines_parametric_and_symmetric.ipynb
│   │   │   │   ├── 03_planes_general_and_geometric.ipynb
│   │   │   │   └── 04_distance_formulas.ipynb
│   │   │   ├── 02_projections/
│   │   │   │   ├── 01_projections.ipynb
│   │   │   │   ├── 02_projection_onto_line.ipynb
│   │   │   │   ├── 03_projection_onto_plane.ipynb
│   │   │   │   └── 04_projection_matrices.ipynb
│   │   │   ├── 03_coordinate_projections/
│   │   │   │   ├── 01_coordinate_projections.ipynb
│   │   │   │   ├── 02_components_orthonormal_basis.ipynb
│   │   │   │   ├── 03_components_generic_basis.ipynb
│   │   │   │   └── 04_change_of_basis.ipynb
│   │   │   ├── 04_vector_spaces/
│   │   │   │   ├── 01_vector_spaces.ipynb
│   │   │   │   ├── 02_span.ipynb
│   │   │   │   ├── 03_fundamental_subspaces.ipynb
│   │   │   │   ├── 04_rank_nullity_theorem.ipynb
│   │   │   │   └── 05_linear_independence.ipynb
│   │   │   ├── 05_vector_space_techniques/
│   │   │   │   ├── 01_vector_space_techniques.ipynb
│   │   │   │   ├── 02_basis_row_space.ipynb
│   │   │   │   ├── 03_basis_column_space.ipynb
│   │   │   │   └── 04_basis_null_space.ipynb
│   │   │   └── 06_geometrical_problems/
│   │   │       ├── 01_geometrical_problems.ipynb
│   │   │       ├── 02_intersection_of_lines.ipynb
│   │   │       ├── 03_plane_through_three_points.ipynb
│   │   │       ├── 04_distance_point_to_plane.ipynb
│   │   │       └── 05_projection_onto_plane_problem.ipynb
│   │   │
│   │   ├── 06_Linear_Transformations/                            (23 notebooks)
│   │   │   ├── 01_linear_transformations/
│   │   │   │   ├── 01_linear_transformations.ipynb
│   │   │   │   ├── 02_image_space_and_kernel.ipynb
│   │   │   │   ├── 03_input_output_space_decomposition.ipynb
│   │   │   │   ├── 04_composition.ipynb
│   │   │   │   ├── 05_invertible_transformations.ipynb
│   │   │   │   └── 06_affine_transformations.ipynb
│   │   │   ├── 02_finding_matrix_representations/
│   │   │   │   ├── 01_finding_matrix_representations.ipynb
│   │   │   │   ├── 02_projections.ipynb
│   │   │   │   ├── 03_reflections.ipynb
│   │   │   │   ├── 04_rotations.ipynb
│   │   │   │   └── 05_eigenspaces_preview.ipynb
│   │   │   ├── 03_change_of_basis_for_matrices/
│   │   │   │   ├── 01_change_of_basis_for_matrices.ipynb
│   │   │   │   ├── 02_matrix_components.ipynb
│   │   │   │   ├── 03_change_of_basis_formula.ipynb
│   │   │   │   └── 04_similarity_transformation.ipynb
│   │   │   ├── 04_invertible_matrix_theorem/
│   │   │   │   ├── 01_invertible_matrix_theorem.ipynb
│   │   │   │   ├── 02_the_10_equivalent_statements.ipynb
│   │   │   │   ├── 03_proof_structure_and_singular_example.ipynb
│   │   │   │   └── 04_injective_surjective_bijective.ipynb
│   │   │   └── 05_linear_transformations_problems/
│   │   │       ├── 01_linear_transformations_problems.ipynb
│   │   │       ├── 02_p6_1_image_space_r2_to_r3.ipynb
│   │   │       ├── 03_p6_2_transformation_on_function_spaces.ipynb
│   │   │       └── 04_p6_3_derivative_on_polynomials.ipynb
│   │   │
│   │   └── 07_Theoretical_Linear_Algebra/                        (65 notebooks)
│   │       ├── 01_eigenvalues_and_eigenvectors/
│   │       │   ├── 01_eigenvalues_and_eigenvectors.ipynb
│   │       │   ├── 02_definitions.ipynb
│   │       │   ├── 03_eigenvalues.ipynb
│   │       │   ├── 04_eigenvectors.ipynb
│   │       │   ├── 05_eigendecomposition.ipynb
│   │       │   ├── 06_eigenspaces.ipynb
│   │       │   ├── 07_change_of_basis_matrix.ipynb
│   │       │   ├── 08_interpretation.ipynb
│   │       │   ├── 09_invariant_properties.ipynb
│   │       │   ├── 10_relation_to_invertibility.ipynb
│   │       │   ├── 11_normal_matrices_eigendecomposition.ipynb
│   │       │   ├── 12_non_diagonalizable_matrices.ipynb
│   │       │   ├── 13_matrix_power_series.ipynb
│   │       │   └── 14_applications.ipynb
│   │       ├── 02_special_types_of_matrices/
│   │       │   ├── 01_special_types_of_matrices.ipynb
│   │       │   ├── 02_diagonal_matrices.ipynb
│   │       │   ├── 03_symmetric_matrices.ipynb
│   │       │   ├── 04_upper_triangular.ipynb
│   │       │   ├── 05_identity_matrix.ipynb
│   │       │   ├── 06_orthogonal_matrices.ipynb
│   │       │   ├── 07_rotation_matrices.ipynb
│   │       │   ├── 08_reflections.ipynb
│   │       │   ├── 09_permutation_matrices.ipynb
│   │       │   ├── 10_positive_matrices.ipynb
│   │       │   ├── 11_projection_matrices.ipynb
│   │       │   └── 12_normal_matrices.ipynb
│   │       ├── 03_abstract_vector_spaces/
│   │       │   ├── 01_abstract_vector_spaces.ipynb
│   │       │   ├── 02_definitions.ipynb
│   │       │   ├── 03_examples_matrices.ipynb
│   │       │   ├── 04_examples_symmetric_2x2.ipynb
│   │       │   ├── 05_examples_polynomials.ipynb
│   │       │   └── 06_examples_functions.ipynb
│   │       ├── 04_abstract_inner_product_spaces/
│   │       │   ├── 01_abstract_inner_product_spaces.ipynb
│   │       │   ├── 02_definitions.ipynb
│   │       │   ├── 03_orthogonality.ipynb
│   │       │   ├── 04_norm.ipynb
│   │       │   ├── 05_distance.ipynb
│   │       │   ├── 06_matrix_inner_product.ipynb
│   │       │   ├── 07_function_inner_product.ipynb
│   │       │   ├── 08_generalized_dot_product.ipynb
│   │       │   └── 09_valid_invalid_inner_products.ipynb
│   │       ├── 05_gram_schmidt/
│   │       │   ├── 01_gram_schmidt.ipynb
│   │       │   ├── 02_definitions.ipynb
│   │       │   ├── 03_orthonormal_bases.ipynb
│   │       │   └── 04_gram_schmidt_procedure.ipynb
│   │       ├── 06_matrix_decompositions/
│   │       │   ├── 01_matrix_decompositions.ipynb
│   │       │   ├── 02_eigendecomposition.ipynb
│   │       │   ├── 03_svd.ipynb
│   │       │   ├── 04_lu.ipynb
│   │       │   ├── 05_cholesky.ipynb
│   │       │   └── 06_qr.ipynb
│   │       ├── 07_complex_linear_algebra/
│   │       │   ├── 01_complex_linear_algebra.ipynb
│   │       │   ├── 02_complex_vectors.ipynb
│   │       │   ├── 03_complex_matrices.ipynb
│   │       │   ├── 04_hermitian_transpose.ipynb
│   │       │   ├── 05_complex_inner_product.ipynb
│   │       │   ├── 06_complex_norm.ipynb
│   │       │   ├── 07_unitary_matrices.ipynb
│   │       │   ├── 08_hermitian_matrices.ipynb
│   │       │   ├── 09_normal_matrices_complex.ipynb
│   │       │   ├── 10_complex_eigenvalues.ipynb
│   │       │   ├── 11_complex_svd.ipynb
│   │       │   └── 12_adjoint_operator.ipynb
│   │       └── 08_theory_problems/
│   │           ├── 01_theory_problems.ipynb
│   │           └── 02_problem_set.ipynb
│   │
│   ├── 02_Probability_and_Statistics/                        ← planned
│   ├── 03_Calculus/                                          ← planned
│   ├── 04_Calculus_of_Variations/                            ← planned
│   ├── 05_Differential_Equations/                            ← planned
│   └── 06_Integral_Transforms/                               ← planned
│
├── 02_Dynamics/
│   ├── 01_Causal_Acausal_Modeling/                           ← planned
│   ├── 02_Modeling_using_Lagrange/                           ← planned
│   ├── 03_Modeling_using_Bond_Graph/                         ← planned
│   └── 04_Modeling_using_Port_Hamiltonian/                   ← planned
│
├── 03_Optimization/
│   ├── 01_Linear_Programming_LP/                             ← planned
│   ├── 02_Convex_Quadratic_Programming_QP/                   ← planned
│   ├── 03_Convex_QCQP/                                      ← planned
│   ├── 04_Second_Order_Cone_Programming_SOCP/                ← planned
│   ├── 05_Semidefinite_Programming_SDP/                      ← planned
│   ├── 06_Mixed_Integer_Programming_MIP/                     ← planned
│   │   ├── 01_MILP/
│   │   ├── 02_MIQP/
│   │   ├── 03_MICP_MISOCP_MISDP/
│   │   └── 04_MINLP/
│   ├── 07_Global_Nonconvex_Optimization/                     ← planned
│   ├── 08_Robust_and_Stochastic_Optimization/                ← planned
│   └── 09_PDE_Constrained_Optimization/                      ← planned
│
├── 04_Optimal_Control/
│   ├── 01_Controllability_and_Observability/                 ← planned
│   ├── 02_Kalman_Filter/                                     ← planned
│   ├── 03_Full_State_Feedback_Control/                       ← planned
│   ├── 04_Linear_Quadratic_Regulator_LQR/                   ← planned
│   ├── 05_Linear_Quadratic_Gaussian_LQG/                    ← planned
│   ├── 06_Trajectory_Optimization_DDP_iLQR/                 ← planned
│   ├── 07_Model_Predictive_Control_MPC/                     ← planned
│   ├── 08_Nonlinear_MPC/                                    ← planned
│   ├── 09_Robust_and_H_infinity_Control/                    ← planned
│   ├── 10_Hybrid_and_Switched_Systems_Control/               ← planned
│   ├── 11_PDE_Constrained_Optimal_Control/                  ← planned
│   └── 13_Reinforcement_Learning_and_Approx_DP/             ← planned
│
├── 05_Machine_Learning/                                      ← planned
│
├── 07_Reinforcement_Learning/                                ← planned
│
├── Figures/                                                  ← diagrams, plots
└── Literature/                                               ← source textbooks (Markdown)
```

---

## Notebook Format

Every notebook follows a simple structure:

| Cell | Type     | Content                              |
|------|----------|--------------------------------------|
| 1    | Markdown | Section number and title             |
| 2    | Markdown | Core equation(s)                     |
| 3    | Markdown | Example(s)                           |
| 4    | Code     | Simple Python implementation         |
| 5    | Markdown | References + Previous / Next links   |

Notebooks are sequentially linked — each one points to the previous and next in the series.

---

## Current Progress

| Section | Topic | Notebooks | Status |
|---------|-------|-----------|--------|
| 01 Foundations / 01 Linear Algebra | Definitions | 10 | ✅ |
| | Vectors | 6 | ✅ |
| | Matrices | 19 | ✅ |
| | Computational Linear Algebra | 6 | ✅ |
| | Geometrical Aspects | 26 | ✅ |
| | Linear Transformations | 23 | ✅ |
| | Theoretical Linear Algebra | 65 | ✅ |
| 01 Foundations / 02–06 | Probability, Calculus, etc. | — | 📋 Planned |
| 02 Dynamics | Lagrange, Bond Graph, Port-Hamiltonian | — | 📋 Planned |
| 03 Optimization | LP → PDE-Constrained | — | 📋 Planned |
| 04 Optimal Control | LQR, MPC, H∞, RL | — | 📋 Planned |
| 05 Machine Learning | — | — | 📋 Planned |
| 07 Reinforcement Learning | — | — | 📋 Planned |

**Current Total: 155 notebooks**

---

# Getting Started 


> [!IMPORTANT]
> **You don't need to install anything to explore this hub.**
> All notebooks render directly on GitHub — Just **leave a Star** and enjoy the ride! 🚀

> The setup guide below is **only** for those who want to **run the Python code**, **modify notebooks**, or **experiment locally**. And assumes **no prior developer setup** and walks through everything from scratch.


## 1. Install Git

Git is required to download (clone) the repository.

### macOS
1. Open Terminal  
2. Run:
```bash
git --version
```
If Git is not installed, install via:
```bash
xcode-select --install
```

### Windows
1. Go to: https://git-scm.com/download/win  
2. Download and install with default settings  
3. Restart terminal after install  

### Linux (Ubuntu)
```bash
sudo apt update
sudo apt install git
```

Verify:
```bash
git --version
```

## 2. Install Python

Python 3.10 or newer is recommended.

Download from:
https://www.python.org/downloads/

During installation on Windows:
✔ Check **"Add Python to PATH"**

Verify installation:
```bash
python --version
```
or
```bash
python3 --version
```

## 3. Clone the Repository

Open a terminal (Terminal / PowerShell / Command Prompt).

Choose where you want the project folder, then run:

```bash
git clone https://github.com/AlexMunoz25/optimal-control-and-ml.git
cd optimal-control-and-ml
```

This downloads the repo and moves into it.

## 4. Create a Virtual Environment

A virtual environment keeps dependencies isolated.

```bash
python -m venv .venv
```

### Activate it

#### macOS / Linux
```bash
source .venv/bin/activate
```

#### Windows (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
```

You should now see `(.venv)` in your terminal.

## 5. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Install VS Code

Download:
https://code.visualstudio.com/

Install normally.

## 7. Install VS Code Extensions

Open VS Code → Extensions tab → install:

- Python (Microsoft)
- Jupyter (Microsoft)

Or install from terminal:
```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
```

## 8. Open the Project

From inside the repo folder:
```bash
code .
```

Or open VS Code → File → Open Folder → select the repo folder.

## 9. Select Python Interpreter

Top-right corner in VS Code:
Select interpreter → choose:

```
.venv
```

## 10. Run Notebooks

Open any `.ipynb` file and press:

- **Run All**
- or run cells individually

VS Code will automatically use the environment.

## Alternative Method — Classic Jupyter

If you prefer standard Jupyter Notebook or Anaconda, follow below.

### Option A — Using pip

```bash
git clone https://github.com/AlexMunoz25/optimal-control-and-ml.git
cd optimal-control-and-ml

python -m venv .venv
source .venv/bin/activate   # Windows equivalent if needed

pip install -r requirements.txt
pip install jupyter

jupyter notebook
```

Browser will open automatically.

### Option B — Using Anaconda

Install Anaconda:
https://www.anaconda.com/download

Then:

```bash
git clone https://github.com/AlexMunoz25/optimal-control-and-ml.git
cd optimal-control-and-ml

conda create -n ocml python=3.11
conda activate ocml

pip install -r requirements.txt
jupyter notebook
```

---

# Updating the Repo

To pull latest changes later:

```bash
git pull
```

## Deactivate Environment

When finished:
```bash
deactivate
```

### (Additional) Tutorial 

VS Code + Jupyter setup walkthrough:

https://www.youtube.com/watch?v=9FZzw9nF8Rg

---

## References so far

- Savov, I. (2016). *No Bullshit Guide to Linear Algebra*
- Aazi, M. (2024). *Mathematics For Machine Learning*
- Rozycki, P. (2020). *Computational Mechanics Course Notes, École Centrale de Nantes*