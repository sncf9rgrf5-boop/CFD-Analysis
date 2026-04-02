# ✈️ Interactive 2D CFD Solver: NACA 0012 Virtual Wind Tunnel

A high-performance, interactive Computational Fluid Dynamics (CFD) solver built in Python. This project simulates incompressible fluid flow within a lid-driven cavity, featuring a submerged NACA 0012 airfoil with adjustable Angle of Attack (AoA).

## 🚀 Key Features
* **Physics Engine:** Solves the incompressible Navier-Stokes equations using **Chorin’s Projection Method**.
* **Numerical Stability:** Implements a **First-Order Upwind Scheme** to maintain stability at high Reynolds numbers ($Re > 10,000$).
* **Interactive UI:** Real-time sliders for Lid Speed, Reynolds Number, and Angle of Attack.
* **Dual Visualization:** * **Eulerian:** Real-time vorticity field contours and streamlines.
    * **Lagrangian:** 100+ particle tracers with bilinear interpolation for smooth advection.
* **Optimization:** Utilizes **NumPy vectorization** and **Matplotlib Artist updates** for flicker-free, 30+ FPS performance.

## 🛠️ The Math Under the Hood
The solver handles the momentum and continuity equations:
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}$$
$$\nabla \cdot \mathbf{u} = 0$$

## 📦 Installation & Usage
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/FluidFlow-Simulator.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the simulation: `python cfd_visualizer.py`