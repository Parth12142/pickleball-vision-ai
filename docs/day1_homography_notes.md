# Day 1 Study Notes & Worked 4-Point Example: Homography Fundamentals

**Course Tracker Reference**: Week 1, Day 1 (Rows 2–4 of `Pickleball_AI_Tracker.xlsx`)  
**Deliverable**: Study notes and worked 4-point example (`Assignment 1: Hand-solve a 4-point mapping`)  
**Feeds Into**: Day 2 ($H$ matrix and OpenCV functions: `cv2.findHomography`, `cv2.perspectiveTransform`, `cv2.warpPerspective`)

---

## 1. Sub-Topic 1: Homogeneous Coordinates (`(x, y) → (x, y, 1)`)

### Why Cartesian Coordinates `(x, y)` Are Not Enough for Court Vision
In standard 2D Euclidean/Cartesian coordinates, a point is written as a 2-vector $\mathbf{p}_{\text{cart}} = \begin{bmatrix} x \\ y \end{bmatrix}$.
- Multiplying by a $2 \times 2$ matrix can only **rotate**, **scale**, or **shear** around the origin $(0,0)$.
- It **cannot** even represent a simple translation $(x + t_x, y + t_y)$ as a single matrix multiplication, nor can it represent **perspective foreshortening** (where objects farther from the camera shrink in pixel size).

### How Homogeneous Coordinates Work
1. **Lift 2D $\rightarrow$ 3-Vector (Add $w = 1$)**:
   $$(x, y) \longrightarrow \mathbf{p} = \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
2. **Scale Equivalence**:
   Any non-zero scalar multiple $k \neq 0$ represents the **exact same 2D point**:
   $$\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} \equiv \begin{bmatrix} kx \\ ky \\ k \end{bmatrix} = \begin{bmatrix} X' \\ Y' \\ W' \end{bmatrix}$$
3. **Project Back to 2D (Perspective Division by $W'$)**:
   After multiplying by a $3 \times 3$ matrix $M$:
   $$\mathbf{p}' = M \mathbf{p} = \begin{bmatrix} X' \\ Y' \\ W' \end{bmatrix} \implies (x', y') = \left(\frac{X'}{W'}, \frac{Y'}{W'}\right)$$

---

## 2. Sub-Topic 2: Transform Types (Translation vs. Affine vs. Projective)

Every 2D planar transform can be written as a $3 \times 3$ matrix acting on homogeneous coordinates $\begin{bmatrix} x & y & 1 \end{bmatrix}^T$. You can tell them apart immediately by inspecting the **bottom row** and the **Degrees of Freedom (DoF)**:

| Transform Type | $3 \times 3$ Matrix Structure | Bottom Row | DoF | Geometric Invariants (What Stays Preserved) | Pickleball Court Meaning |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Translation** | $\begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}$ | `[0, 0, 1]` | **2** ($t_x, t_y$) | Lengths, angles, areas, parallel lines | Sliding a 2D court diagram across the screen without rotating or resizing it. |
| **2. Affine** | $\begin{bmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \\ 0 & 0 & 1 \end{bmatrix}$ | `[0, 0, 1]` | **6** ($a_{11}, a_{12}, a_{21}, a_{22}, t_x, t_y$) | **Parallel lines remain parallel**; ratios of lengths along parallel lines stay constant ($W' = 1$ always). | Rotating, scaling, or skewing a 2D court diagram. It **cannot** model a real camera view because the far baseline and near baseline would have to be the exact same pixel width! |
| **3. Projective (Homography $H$)** | $\begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & 1 \end{bmatrix}$ | **`[h31, h32, 1]`** ($h_{31}, h_{32} \neq 0$) | **8** (9 entries minus 1 overall scale factor) | **Straight lines remain straight**, but parallel court sidelines converge toward a vanishing point! | **Real Broadcast Camera View**: Because $W' = h_{31}x + h_{32}y + 1$ depends on $(x, y)$, points on the far baseline divide by a different $W'$ than points on the near baseline—mapping a trapezoid in camera pixels to a true $20\text{ ft} \times 44\text{ ft}$ rectangle! |

---

## 3. Sub-Topic 3 & Assignment 1: Hand-Solving a 4-Point Homography Mapping

### Step 1: The Projective Equations for One Point Pair $(x, y) \rightarrow (X, Y)$
Given a camera pixel $(x, y)$ and its corresponding top-down court coordinate $(X, Y)$:
$$\begin{bmatrix} X' \\ Y' \\ W' \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11}x + h_{12}y + h_{13} \\ h_{21}x + h_{22}y + h_{23} \\ h_{31}x + h_{32}y + 1 \end{bmatrix}$$

Performing perspective division ($X = X'/W'$ and $Y = Y'/W'$):
$$X = \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + 1}, \qquad Y = \frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + 1}$$

Multiply both sides by the denominator $(h_{31}x + h_{32}y + 1)$:
$$x h_{11} + y h_{12} + h_{13} - x X h_{31} - y X h_{32} = X$$
$$x h_{21} + y h_{22} + h_{23} - x Y h_{31} - y Y h_{32} = Y$$

### Step 2: Assemble the $8 \times 8$ Linear System $A\mathbf{h} = \mathbf{b}$ for 4 Court Corners
Since $H$ has **8 unknown parameters** $\mathbf{h} = \begin{bmatrix} h_{11} & h_{12} & h_{13} & h_{21} & h_{22} & h_{23} & h_{31} & h_{32} \end{bmatrix}^T$ (with $h_{33} = 1$), **4 non-collinear court corners** ($i = 1, 2, 3, 4$) provide $4 \times 2 = 8$ linear equations:

$$\underbrace{\begin{bmatrix}
x_1 & y_1 & 1 & 0 & 0 & 0 & -x_1 X_1 & -y_1 X_1 \\
0 & 0 & 0 & x_1 & y_1 & 1 & -x_1 Y_1 & -y_1 Y_1 \\
x_2 & y_2 & 1 & 0 & 0 & 0 & -x_2 X_2 & -y_2 X_2 \\
0 & 0 & 0 & x_2 & y_2 & 1 & -x_2 Y_2 & -y_2 Y_2 \\
x_3 & y_3 & 1 & 0 & 0 & 0 & -x_3 X_3 & -y_3 X_3 \\
0 & 0 & 0 & x_3 & y_3 & 1 & -x_3 Y_3 & -y_3 Y_3 \\
x_4 & y_4 & 1 & 0 & 0 & 0 & -x_4 X_4 & -y_4 X_4 \\
0 & 0 & 0 & x_4 & y_4 & 1 & -x_4 Y_4 & -y_4 Y_4
\end{bmatrix}}_{A \ (8 \times 8)}
\underbrace{\begin{bmatrix}
h_{11} \\ h_{12} \\ h_{13} \\ h_{21} \\ h_{22} \\ h_{23} \\ h_{31} \\ h_{32}
\end{bmatrix}}_{\mathbf{h} \ (8 \times 1)}
=
\underbrace{\begin{bmatrix}
X_1 \\ Y_1 \\ X_2 \\ Y_2 \\ X_3 \\ Y_3 \\ X_4 \\ Y_4
\end{bmatrix}}_{\mathbf{b} \ (8 \times 1)}$$

### Step 3: Concrete Worked Example (Unit Square $\rightarrow$ Perspective Court Trapezoid & Inverse)
Suppose we have 4 camera pixel corners of a pickleball court in a $1280 \times 720$ broadcast frame:
1. **Far-Left Corner** $(x_1, y_1) = (420, 180) \longrightarrow$ USAP Court $(X_1, Y_1) = (0, 0)\text{ ft}$
2. **Far-Right Corner** $(x_2, y_2) = (860, 180) \longrightarrow$ USAP Court $(X_2, Y_2) = (20, 0)\text{ ft}$
3. **Near-Right Corner** $(x_3, y_3) = (1140, 640) \longrightarrow$ USAP Court $(X_3, Y_3) = (20, 44)\text{ ft}$
4. **Near-Left Corner** $(x_4, y_4) = (140, 640) \longrightarrow$ USAP Court $(X_4, Y_4) = (0, 44)\text{ ft}$

Notice how for Corner 1 where $(X_1, Y_1) = (0, 0)$, the first two rows of $A\mathbf{h} = \mathbf{b}$ simplify by hand to:
$$420 h_{11} + 180 h_{12} + h_{13} = 0$$
$$420 h_{21} + 180 h_{22} + h_{23} = 0$$
And because the far baseline $y_1 = y_2 = 180$ maps to $Y = 0$, subtracting Row 2 from Row 4 immediately gives $h_{21} = 0$, meaning the horizontal image axis is parallel to the far baseline!

Solving $\mathbf{h} = A^{-1}\mathbf{b}$ in [`court/day1_homography_basics.py`](file:///home/pce118/Desktop/pickleball-vision-ai-main/court/day1_homography_basics.py) yields:
$$H = \begin{bmatrix}
0.09055118 & 0.05511811 & -47.95275591 \\
0.00000000 & 0.43307087 & -77.95275591 \\
0.00000000 & 0.00551181 & 1.00000000
\end{bmatrix}$$
And checking the net center pixel $(640.00, 320.56)$ through $H$ maps to the exact USAP court center $(10.0\text{ ft}, 22.0\text{ ft})$ with $0.000000000000\text{ ft}$ error!


---

## 4. Plan Day 2 (Rows 5–7): OpenCV Homography Functions & Bird's-Eye Frame Warp

### 1. `cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold)` (Row 5)
- While our Day 1 solver (`solve_homography_4pt_numpy`) works when we have **4 exact, noise-free corners**, in real computer vision we often have **more than 4 keypoint matches** (e.g., 8–14 court intersections) where 1 or 2 keypoints might be false detections (*outliers*).
- **`cv2.RANSAC` (Random Sample Consensus)**:
  1. Randomly picks minimal sets of 4 point pairs and computes a candidate $H$.
  2. Projects all source points through $H$ and counts how many land within `ransacReprojThreshold` pixels/units of their target (`inliers`).
  3. Rejects the bad outliers (`mask[i] == 0`) and re-fits $H$ using **only the clean inliers** (`mask[i] == 1`).

### 2. `cv2.perspectiveTransform(pts, H)` (Row 6)
- Maps a list of $N$ 2D points through $H$ in one fast C++ call (including homogeneous conversion and perspective division by $W'$).
- **Important OpenCV Shape Rule**: `cv2.perspectiveTransform` expects a 3D `float32` or `float64` array of shape **`(N, 1, 2)`**:
  ```python
  pts_in = np.array(points_list, dtype=np.float32).reshape(-1, 1, 2)
  pts_out = cv2.perspectiveTransform(pts_in, H).reshape(-1, 2)
  ```

### 3. `cv2.warpPerspective(frame, H_img, (out_w, out_h))` (Row 7 — Assignment 2)
- Maps every pixel $(u, v)$ of a camera video frame into a **top-down bird's-eye image** of size `(out_w, out_h)` using bilinear interpolation.
- To leave a clean apron border around the $20 \times 44\text{ ft}$ court in the bird's-eye image, we map the 4 court corners to pixel coordinates inside a $500 \times 1000\text{ px}$ canvas (`margin = 60 px`, `scale = 19 px/ft`) and call `cv2.warpPerspective(frame, H_img, (500, 1000))`.
