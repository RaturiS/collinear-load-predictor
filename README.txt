RF Cavity Collinear Load Predictor
Overview
This repository contains a Machine Learning surrogate model designed to bypass computationally expensive 3D full-wave electromagnetic simulations (CST Microwave Studio).
The Engineering Problem - Determining the exact coating length and radius for a collinear load to achieve a target Quality Factor (Q) and resonant frequency typically requires
hours of iterative CST simulations.

The Solution
By utilizing a decoupled polynomial regression architecture trained on an initial 80-run dataset, this model instantly predicts the required physical dimensions for a target Q.
Accuracy: Predicts coating length to within 1.4 mm wrt CST simulations.
Efficiency: Reduces simulation wait times from hours to milliseconds.
Tech Stack: Python, scikit-learn, Pandas, NumPy.
