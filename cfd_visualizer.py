import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button
import time

# --- Configuration Toggles ---
VIEW_MODE = 'Vorticity'  

# --- 1. Simulation Parameters ---
nx, ny = 40, 40
Lx, Ly = 1.0, 1.0
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
rho = 1.0

# --- 2. Initialize Arrays ---
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))
u_star = np.zeros((ny, nx))
v_star = np.zeros((ny, nx))

t = 0.0
last_dt = 1.0  

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# --- 3. Airfoil Definition (NACA 0012) ---
c = 0.4         
x_le = 0.3      
y_le = 0.5      
t_airfoil = 0.12
x_qc = x_le + 0.25 * c  
y_qc = y_le

def rotate_points(pts_x, pts_y, angle_deg, cx, cy):
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    del_x = pts_x - cx
    del_y = pts_y - cy
    
    rot_x = cx + del_x * cos_t - del_y * sin_t
    rot_y = cy + del_x * sin_t + del_y * cos_t
    return rot_x, rot_y

is_airfoil = np.zeros_like(X, dtype=bool)
cached_aoa = None

def update_airfoil_mask(aoa_deg):
    global cached_aoa, is_airfoil
    if cached_aoa == aoa_deg:
        return
    cached_aoa = aoa_deg
    
    X_loc, Y_loc = rotate_points(X, Y, -aoa_deg, x_qc, y_qc)
    
    is_airfoil.fill(False)
    valid = (X_loc >= x_le) & (X_loc <= x_le + c)
    if np.any(valid):
        x_c = (X_loc[valid] - x_le) / c
        yt = 5 * t_airfoil * c * (0.2969 * np.sqrt(x_c) - 0.1260 * x_c - 0.3516 * x_c**2 + 0.2843 * x_c**3 - 0.1015 * x_c**4)
        is_airfoil[valid] = (Y_loc[valid] >= y_le - yt) & (Y_loc[valid] <= y_le + yt)

naca_x_vals = np.linspace(0, 1, 100)
naca_yt_vals = 5 * t_airfoil * c * (0.2969 * np.sqrt(naca_x_vals) - 0.1260 * naca_x_vals - 0.3516 * naca_x_vals**2 + 0.2843 * naca_x_vals**3 - 0.1015 * naca_x_vals**4)
base_upper_x = x_le + naca_x_vals * c
base_upper_y = y_le + naca_yt_vals
base_lower_x = x_le + naca_x_vals * c
base_lower_y = y_le - naca_yt_vals

naca_base_x = np.concatenate((base_upper_x, base_lower_x[::-1]))
naca_base_y = np.concatenate((base_upper_y, base_lower_y[::-1]))

update_airfoil_mask(0.0)

# --- 4. Lagrangian Particle Tracer ---
num_particles = 100

def in_airfoil(px, py, aoa_deg):
    px_loc, py_loc = rotate_points(px, py, -aoa_deg, x_qc, y_qc)
    inside = np.zeros_like(px_loc, dtype=bool)
    valid = (px_loc >= x_le) & (px_loc <= x_le + c)
    if not np.any(valid):
        return inside
    
    x_c = (px_loc[valid] - x_le) / c
    yt = 5 * t_airfoil * c * (0.2969 * np.sqrt(x_c) - 0.1260 * x_c - 0.3516 * x_c**2 + 0.2843 * x_c**3 - 0.1015 * x_c**4)
    inside[valid] = (py_loc[valid] >= y_le - yt) & (py_loc[valid] <= y_le + yt)
    return inside

def generate_safe_particles(count, aoa_deg):
    px = np.random.uniform(low=0.0, high=Lx, size=count)
    py = np.random.uniform(low=0.0, high=Ly, size=count)
    while True:
        in_obs = in_airfoil(px, py, aoa_deg)
        if not np.any(in_obs):
            break
        px[in_obs] = np.random.uniform(0.0, Lx, size=np.sum(in_obs))
        py[in_obs] = np.random.uniform(0.0, Ly, size=np.sum(in_obs))
    return px, py

px_init, py_init = generate_safe_particles(num_particles, 0.0)
particles = np.column_stack((px_init, py_init))

# --- 5. Physics Kernel Functions ---
def calculate_dt(u, v, dx, dy, nu, U_lid):
    u_max = max(np.max(np.abs(u)), abs(U_lid)) + 1e-10
    v_max = np.max(np.abs(v)) + 1e-10
    dt_c_x = dx / u_max
    dt_c_y = dy / v_max
    dt_c_min = min(dt_c_x, dt_c_y)
    dt_d = 0.5 / nu * (dx**2 * dy**2) / (dx**2 + dy**2)
    return 0.5 * min(dt_c_min, dt_d)

def compute_intermediate_velocity(u, v, dt, dx, dy, nu):
    un = u.copy()
    vn = v.copy()
    u_star = np.copy(u)
    v_star = np.copy(v)
    
    ui = un[1:-1, 1:-1]
    vi = vn[1:-1, 1:-1]
    
    u_adv_x = np.where(ui > 0, ui * (ui - un[1:-1, 0:-2]) / dx, ui * (un[1:-1, 2:] - ui) / dx)
    u_adv_y = np.where(vi > 0, vi * (ui - un[0:-2, 1:-1]) / dy, vi * (un[2:, 1:-1] - ui) / dy)
                       
    v_adv_x = np.where(ui > 0, ui * (vi - vn[1:-1, 0:-2]) / dx, ui * (vn[1:-1, 2:] - vi) / dx)
    v_adv_y = np.where(vi > 0, vi * (vi - vn[0:-2, 1:-1]) / dy, vi * (vn[2:, 1:-1] - vi) / dy)
                       
    u_diff_x = (un[1:-1, 2:] - 2 * ui + un[1:-1, 0:-2]) / dx**2
    u_diff_y = (un[2:, 1:-1] - 2 * ui + un[0:-2, 1:-1]) / dy**2
    v_diff_x = (vn[1:-1, 2:] - 2 * vi + vn[1:-1, 0:-2]) / dx**2
    v_diff_y = (vn[2:, 1:-1] - 2 * vi + vn[0:-2, 1:-1]) / dy**2
    
    u_star[1:-1, 1:-1] = ui - dt * (u_adv_x + u_adv_y) + nu * dt * (u_diff_x + u_diff_y)
    v_star[1:-1, 1:-1] = vi - dt * (v_adv_x + v_adv_y) + nu * dt * (v_diff_x + v_diff_y)
    return u_star, v_star

def compute_pressure_rhs(b, dt, u_star, v_star, dx, dy, rho):
    b[1:-1, 1:-1] = (rho * (1.0 / dt * 
                    ((0.5 * (u_star[1:-1, 2:] - u_star[1:-1, 0:-2]) / dx) + 
                     (0.5 * (v_star[2:, 1:-1] - v_star[0:-2, 1:-1]) / dy))))
    return b

def solve_pressure_poisson(p, b, dx, dy, max_iters=500, tol=1e-4):
    pn = np.empty_like(p)
    dx2 = dx**2
    dy2 = dy**2
    denom = 2.0 * (dx2 + dy2)
    
    for _ in range(max_iters):
        pn = p.copy()
        p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy2 + 
                         (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx2 - 
                         b[1:-1, 1:-1] * dx2 * dy2) / denom
        
        p[:, -1] = p[:, -2]
        p[:, 0] = p[:, 1]
        p[-1, :] = p[-2, :]
        p[0, :] = p[1, :]
        p[0, 0] = 0.0
        
        if np.max(np.abs(p - pn)) < tol:
            break
    return p

def correct_velocity(u, v, u_star, v_star, p, dt, dx, dy, rho):
    u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt / rho * 0.5 * (p[1:-1, 2:] - p[1:-1, 0:-2]) / dx
    v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt / rho * 0.5 * (p[2:, 1:-1] - p[0:-2, 1:-1]) / dy
    return u, v

def apply_boundary_conditions(u, v, current_U_lid):
    u[:, 0] = 0.0; u[:, -1] = 0.0; u[0, :] = 0.0
    v[:, 0] = 0.0; v[:, -1] = 0.0; v[0, :] = 0.0; v[-1, :] = 0.0
    u[-1, :] = current_U_lid
    u[is_airfoil] = 0.0  
    v[is_airfoil] = 0.0
    return u, v

def compute_vorticity(u, v, dx, dy):
    omega = np.zeros_like(u)
    omega[1:-1, 1:-1] = ((v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx) - 
                         (u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy))
    omega[is_airfoil] = 0.0
    return omega

def advect_particles(particles, u, v, dt, dx, dy, Lx, Ly, aoa_deg):
    px = particles[:, 0]
    py = particles[:, 1]
    
    x_idx = px / dx
    y_idx = py / dy
    
    ix = np.clip(np.floor(x_idx).astype(int), 0, u.shape[1] - 2)
    iy = np.clip(np.floor(y_idx).astype(int), 0, u.shape[0] - 2)
    
    wx = x_idx - ix
    wy = y_idx - iy
    
    u_p = ((1 - wx) * (1 - wy) * u[iy, ix] + wx * (1 - wy) * u[iy, ix + 1] +
           (1 - wx) * wy * u[iy + 1, ix] + wx * wy * u[iy + 1, ix + 1])
           
    v_p = ((1 - wx) * (1 - wy) * v[iy, ix] + wx * (1 - wy) * v[iy, ix + 1] +
           (1 - wx) * wy * v[iy + 1, ix] + wx * wy * v[iy + 1, ix + 1])
           
    px += u_p * dt
    py += v_p * dt
    
    out_of_bounds_mask = (px < 0) | (px > Lx) | (py < 0) | (py > Ly)
    hit_airfoil_mask = in_airfoil(px, py, aoa_deg)
    respawn_mask = out_of_bounds_mask | hit_airfoil_mask
    
    num_respawn = np.sum(respawn_mask)
    if num_respawn > 0:
        new_px, new_py = generate_safe_particles(num_respawn, aoa_deg)
        px[respawn_mask] = new_px
        py[respawn_mask] = new_py
        
    particles[:, 0] = px
    particles[:, 1] = py
    return particles

# --- 6. Main Simulation, UI, and Artist Renders ---

fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.35)

ax_slider_lid = plt.axes([0.15, 0.22, 0.7, 0.03])
lid_slider = Slider(ax=ax_slider_lid, label='Lid Speed', valmin=0.0, valmax=5.0, valinit=1.0, color='steelblue')

ax_slider_re = plt.axes([0.15, 0.16, 0.7, 0.03])
re_slider = Slider(ax=ax_slider_re, label='Target Re', valmin=10.0, valmax=10000.0, valinit=5000.0, color='indianred')

ax_slider_aoa = plt.axes([0.15, 0.10, 0.7, 0.03])
aoa_slider = Slider(ax=ax_slider_aoa, label='AoA (deg)', valmin=-20.0, valmax=20.0, valinit=0.0, color='forestgreen')

def handle_slider_refresh(val):
    fig.canvas.draw_idle()

lid_slider.on_changed(handle_slider_refresh)
re_slider.on_changed(handle_slider_refresh)
aoa_slider.on_changed(handle_slider_refresh)

ax_btn_reset = plt.axes([0.4, 0.03, 0.3, 0.04])
btn_reset = Button(ax_btn_reset, 'Reset Particles', hovercolor='0.975')

def reset_particles_action(event):
    global particles
    px, py = generate_safe_particles(num_particles, aoa_slider.val)
    particles[:, 0] = px
    particles[:, 1] = py
    fig.canvas.draw_idle()

btn_reset.on_clicked(reset_particles_action)

# --- Record Logic ---
is_recording = False
frames_recorded = 0
writer = None

ax_btn_record = plt.axes([0.72, 0.03, 0.22, 0.04])
btn_record = Button(ax_btn_record, 'Record GIF', hovercolor='indianred')

def record_action(event):
    global is_recording, frames_recorded, writer
    if is_recording: return
    is_recording = True
    frames_recorded = 0
    writer = animation.PillowWriter(fps=33)
    writer.setup(fig, "airfoil_simulation.gif", dpi=100)
    btn_record.label.set_text("Recording...")
    fig.canvas.draw_idle()
    print("Recording started. Please wait (~5 seconds of data will be captured)...")

btn_record.on_clicked(record_action)

# --- High-Performance Artist Setup ---
vorticity_mesh = ax.pcolormesh(X, Y, np.zeros_like(u), cmap='RdBu', shading='gouraud', zorder=1)

particle_scatter = ax.scatter(particles[:, 0], particles[:, 1], color='white', s=16, edgecolor='black', zorder=11)
airfoil_patch = patches.Polygon(np.column_stack((naca_base_x, naca_base_y)), closed=True, color='black', zorder=10)
ax.add_patch(airfoil_patch)

warning_text = ax.text(0.5, 0.5, 'STABILITY LIMIT:\nSimulation slowed', color='red', 
                fontsize=16, fontweight='bold', ha='center', va='center',
                transform=ax.transAxes, zorder=20, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', pad=5))
warning_text.set_visible(False)

title_artist = ax.set_title("Lid-Driven Cavity Initializing...")

ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_aspect('equal')

def simulate_step(max_duration=0.030):
    global u, v, p, u_star, v_star, b, particles, t, last_dt
    
    current_U_lid = lid_slider.val
    current_Re = re_slider.val
    current_aoa = aoa_slider.val
    
    update_airfoil_mask(current_aoa)
    nu = max((abs(current_U_lid) * Lx) / current_Re, 1e-6)
    
    start_time = time.perf_counter()
    steps_this_frame = 0
    
    while (time.perf_counter() - start_time) < max_duration:
        dt = calculate_dt(u, v, dx, dy, nu, current_U_lid)
        t += dt
        last_dt = dt
        
        u_star, v_star = compute_intermediate_velocity(u, v, dt, dx, dy, nu)
        u_star, v_star = apply_boundary_conditions(u_star, v_star, current_U_lid)
        
        b = compute_pressure_rhs(b, dt, u_star, v_star, dx, dy, rho)
        p = solve_pressure_poisson(p, b, dx, dy, max_iters=500, tol=1e-4) # Early Exit
        
        u, v = correct_velocity(u, v, u_star, v_star, p, dt, dx, dy, rho)
        u, v = apply_boundary_conditions(u, v, current_U_lid)
        
        particles = advect_particles(particles, u, v, dt, dx, dy, Lx, Ly, current_aoa)
        steps_this_frame += 1
        
    return steps_this_frame

def update_plot(frame):
    global is_recording, frames_recorded, writer
    
    current_U_lid = lid_slider.val
    current_Re = re_slider.val
    current_aoa = aoa_slider.val
    live_nu = max((abs(current_U_lid) * Lx) / current_Re, 1e-6)
    
    steps_count = simulate_step(max_duration=0.030)
    
    if VIEW_MODE == 'Vorticity':
        new_vorticity_data = compute_vorticity(u, v, dx, dy)
        max_omega = np.max(np.abs(new_vorticity_data)) + 1e-5
        vorticity_mesh.set_array(new_vorticity_data.ravel())
        vorticity_mesh.set_clim(vmin=-max_omega, vmax=max_omega)
        vorticity_mesh.set_cmap('RdBu')
        title_context = "Vorticity Field"
    else:
        new_magnitude_data = np.sqrt(u**2 + v**2)
        vorticity_mesh.set_array(new_magnitude_data.ravel())
        vorticity_mesh.set_clim(vmin=0, vmax=np.max(new_magnitude_data) + 1e-5)
        vorticity_mesh.set_cmap('viridis')
        title_context = "Velocity Magnitude"
        
    new_particle_positions = particles
    particle_scatter.set_offsets(new_particle_positions)
    
    new_rotated_coords_x, new_rotated_coords_y = rotate_points(naca_base_x, naca_base_y, current_aoa, x_qc, y_qc)
    airfoil_patch.set_xy(np.column_stack((new_rotated_coords_x, new_rotated_coords_y)))
    
    warning_text.set_visible(last_dt < 0.0005)
    title_artist.set_text(f"Lid-Driven Cavity | NACA 0012 | {title_context}\nRe: {current_Re:.0f} | AoA: {current_aoa:+.1f}° | Steps: {steps_count}")
    
    # Intercept recording logic precisely saving GUI constraints implicitly checking
    if is_recording:
        writer.grab_frame()
        frames_recorded += 1
        if frames_recorded >= 165:  # ~165 frames = exactly 5 seconds bounds
            writer.finish()
            is_recording = False
            btn_record.label.set_text("Record GIF")
            fig.canvas.draw_idle()
            print("Finished capturing! Saved perfectly iteratively as: airfoil_simulation.gif")
            
    return vorticity_mesh, particle_scatter, airfoil_patch, title_artist, warning_text, btn_record.label

ani = animation.FuncAnimation(fig, update_plot, frames=200, interval=30, blit=True, cache_frame_data=False)

if __name__ == '__main__':
    print("Starting Optimized Interactive NACA 0012 Simulation (Recording Enabled).")
    plt.show()
