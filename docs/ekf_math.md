# EKF Mathematics

## State Vector

$$
\mathbf{x} = \begin{bmatrix} x \\ y \\ v_x \\ v_y \\ \psi \\ a_{bias} \end{bmatrix}
$$

Where:
- $x, y$ — position in the local map frame (metres)
- $v_x, v_y$ — velocity components in x and y (m/s)
- $\psi$ — heading / yaw (radians)
- $a_{bias}$ — accelerometer bias estimate (m/s²)

## Motion Model (Prediction)

Constant-acceleration model:

$$
\mathbf{x}_{k+1} = f(\mathbf{x}_k) = \begin{bmatrix}
  x_k + v_{x,k} \, \Delta t \\
  y_k + v_{y,k} \, \Delta t \\
  v_{x,k} + (a_x - a_{bias,k}) \, \Delta t \\
  v_{y,k} + (a_y - a_{bias,k}) \, \Delta t \\
  \psi_k + \dot{\psi}_k \, \Delta t \\
  a_{bias,k}
\end{bmatrix}
$$

Bias is modeled as a random walk — predicted to stay constant,
corrected over time via IMU updates.
## Jacobian of the Motion Model

$$
F = \frac{\partial f}{\partial \mathbf{x}} = \begin{bmatrix}
1 & 0 & \Delta t & 0 & 0 & 0 \\
0 & 1 & 0 & \Delta t & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & -\Delta t \\
0 & 0 & 0 & 1 & 0 & -\Delta t \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
$$
## Covariance Prediction

$$
P_{k+1|k} = F \, P_{k|k} \, F^\top + Q
$$

Where $Q$ is the process noise covariance (diagonal).

## Measurement Update (General Form)

Innovation:

$$
\mathbf{y} = \mathbf{z} - H \, \mathbf{x}
$$

Innovation covariance:

$$
S = H \, P \, H^\top + R
$$

Kalman gain:

$$
K = P \, H^\top \, S^{-1}
$$

State update:

$$
\mathbf{x} \leftarrow \mathbf{x} + K \, \mathbf{y}
$$

Covariance update:

$$
P \leftarrow (I - K \, H) \, P
$$

## GNSS Measurement Model

Observes position directly:

$$
H_{\text{GNSS}} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}
$$

$$
R_{\text{GNSS}} = \operatorname{diag}(\sigma_{x}^2, \sigma_{y}^2)
$$

## IMU Measurement Model

Observes velocity (via acceleration integration approximation) and yaw:

$$
H_{\text{IMU}} = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$

Pseudo-measurements derived from IMU readings:
- $z_v = v_{\text{current}} + a_x \cdot \Delta t$
- $z_\psi = \psi_{\text{current}} + \dot\psi \cdot \Delta t$

## Odometry Measurement Model

Observes velocity and yaw:

$$
H_{\text{odom}} = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$

$$
R_{\text{odom}} = \operatorname{diag}(\sigma_v^2, \sigma_{\dot\psi}^2)
$$

## Sensor Fusion Modes

| Mode | GNSS Update | IMU Update | Odometry Update |
|------|:-----------:|:----------:|:---------------:|
| 1    | ✓           |            |                 |
| 2    | ✓           | ✓          |                 |
| 3    | ✓           |            | ✓               |
| 4    | ✓           | ✓          | ✓               |

## GNSS Dropout Handling

During a GNSS outage, position updates stop entirely. The EKF 
continues predicting using the motion model, but covariance grows 
— the filter becomes less certain about position over time.

To limit drift during outages, two things happen:

**1. IMU + odometry take over**
Velocity and yaw updates continue via IMU pseudo-measurements and 
wheel odometry. These don't directly correct position, but they 
keep the velocity and heading estimates tight, which reduces how 
fast position error accumulates during dead reckoning.

**2. Adaptive measurement noise on recovery**
When GNSS returns after a dropout, the first few fixes can be 
noisy or jump sharply. To avoid the filter snapping hard to a 
bad measurement, I inflate $R_{\text{GNSS}}$ temporarily on 
re-acquisition — treating the returning signal with less trust 
until it stabilizes.

This is why the error plots show a graceful rise during outages 
and a smooth recovery rather than a sharp correction spike.

### Observed behavior

From the evaluation runs, EKF mean error during the 8 planned 
GNSS outages stayed within acceptable bounds, with no divergence 
or filter reset required.