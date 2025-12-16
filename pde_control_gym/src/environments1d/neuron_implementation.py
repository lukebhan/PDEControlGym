from neuron_env import NeuronPDE1D
import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    env = NeuronPDE1D()
    steps = 36000000
    
    # simple logs
    ell_hist, z0_hist, z1_hist, uL2_hist, t_hist = [], [], [], [], []

    u_hist = np.zeros((90000, 201))

    for it in range(steps):
        # assuming step() still returns only u for now
        u = env.step()
        if it % 100000 == 0:
            print(f"step={it}")
        # === diagnostics (compute here since step() returns only u) ===
        # length (scalar) and Z components if accessible:
        ell = float(env.lt) if np.ndim(env.lt)==0 else float(env.lt[0])
        z0  = float(env.Z[0,0])
        z1  = float(env.Z[1,0])
        time = float(env.time_index)

        # L2 norm up to the tip index (robust to moving L)
        L = int(env.L)
        L = max(1, min(env.M-1, L))
        uL2 = float(np.sqrt(np.sum(u[:L+1,0]**2)))
        if (it % 400 == 0):
            ell_hist.append(ell); z0_hist.append(z0); z1_hist.append(z1); uL2_hist.append(uL2); t_hist.append(time)
            u_hist[int(it/400), :] = u.ravel()
        

        # # === NaN/Inf guard ===
        if (not np.isfinite(uL2)) or (not np.isfinite(z0)) or (not np.isfinite(z1)):
            print(f"[stop] non-finite at step {it}: ||u||2={uL2}, Z={[z0,z1]}")
            break

        # === early stop on convergence (tune thresholds) ===
        if it > 1000 and uL2 < 1e-12 and abs(z0) < 1e-12 and abs(z1) < 1e-12:
            print(f"[stop] converged at step {it}")
            break

        # throttled logging
        if it % 100000 == 0:
            print(f"it={it:7d}  L={L:4d}  ell={ell:.3e}  ||u||2={uL2:.3e}  Z0={z0:.3e}  Z1={z1:.3e}")

    # final summary
    print(f"Steps run: {len(uL2_hist)}")
    if uL2_hist:
        print(f"Final: ell={ell_hist[-1]:.6e}, ||u||2={uL2_hist[-1]:.3e}, Z=[{z0_hist[-1]:.3e}, {z1_hist[-1]:.3e}]")

    u_hist = np.array(u_hist)
    c_at_0 = u_hist[:, 0] + env.csubeq[0,0]
    c_at_half_ls = u_hist[:, 60] + env.csubeq[60,0]
    c_at_ls = u_hist[:, 120] + env.csubeq[120,0]

    plt.figure()
    plt.plot(t_hist, c_at_0, color='red', label="c(0,t)")
    plt.plot(t_hist, c_at_half_ls, color='blue', label="c(l_s/2,t)")
    plt.plot(t_hist, c_at_ls, color='green', label="c(l_s,t)")
    plt.axhline(env.csubeq[0,0], color='red', linestyle='--', linewidth=0.8,
            label='c_eq(0)')
    plt.axhline(env.csubeq[60,0], color='blue', linestyle='--', linewidth=0.8,
            label='c_eq(l_s/2)')
    plt.axhline(env.csubeq[120,0], color='green', linestyle='--', linewidth=0.8,
            label='c_eq(l_s)')
    plt.title("Tubulin concentration at a specific length vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Tubulin concentration [10^-3 mol/m^3]")
    plt.legend()
    plt.savefig("tubulin_concentration2.png")
    plt.show()    

if __name__ == "__main__":
    main()