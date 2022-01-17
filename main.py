import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Platoon:
    def __init__(self, n, k, cav_info, hdv_info, columnName, ite_max_u=100, ite_max_lambda=3):
        self.hdv = hdv_info
        self.n = n # platoon CAV number
        self.N = self.n + 1  # platoon vehicle number
        self.x = np.array(cav_info[0]) # platoon MPC state: location at current step k
        self.v = np.array(cav_info[1]) # platoon MPC state: velocity at current step k
        self.u = np.zeros(n) # platoon MPC input: acceleration at current step k
        self.ite_max_u = ite_max_u
        self.ite_max_l = ite_max_lambda

        self.initialize_control_variables(k)
        #self.calculate_ite_state()
        self.trajectory = pd.DataFrame(columns=columnName)  #output dataframe

    def initialize_control_variables(self, k):
        # self.k = k  # current time step
        self.x_hdv = self.hdv[0][k]
        self.v_hdv = self.hdv[1][k]
        self.u_hdv = self.hdv[2][k]
        self.x_ite = np.zeros((self.ite_max_l, self.ite_max_u, self.n))  # iteration of x at next step k+1
        self.v_ite = np.zeros((self.ite_max_l, self.ite_max_u, self.n))  # iteration of v at next step k+1
        self.u_ite = np.zeros((self.ite_max_l, self.ite_max_u, self.n))  # iteration of u at next step k+1
        self.l_ite = np.zeros((self.ite_max_l, self.n))  # iteration of lambda at next step k+1
        self.zx_ite = np.zeros((self.ite_max_l, self.ite_max_u, self.n))  # iteration of z, spacing error at next step k+1
        self.zv_ite = np.zeros((self.ite_max_l, self.ite_max_u, self.n))  # iteration of z', velocity error at next step k+1
        self.gradient_ite = np.zeros((self.ite_max_l, self.ite_max_u, self.n))  # iteration of gradient
        self.u_ite_dir = np.zeros((self.ite_max_l, self.ite_max_u, self.n))  # iteration of u improvement direction

        self.ite_u = 0  # current iteration index of u
        self.ite_l = 0  # current iteration index of lambda
        self.primal_idx = self.ite_u
        self.dual_idx = self.ite_l
        self.primal_stop = False
        self.dual_stop = False

    def mpc_algorithm(self, k):
        self.calculate_ite_state()
        while self.ite_l < self.ite_max_l and not self.dual_stop:
            # print(f"ite_lambda:{self.ite_l}")
            while self.ite_u < self.ite_max_u - 1 and not self.primal_stop:
                # print(f"ite_u:{self.ite_u}")
                self.compute_primal()
                self.primal_stop = self.check_primal_stop()

            self.dual_stop = self.check_dual_stop()
            self.primal_idx = self.ite_u
            self.compute_dual()
        self.dual_idx = self.ite_l -1
        self.update_state(k)

    def compute_primal(self):
        self.ite_u += 1
        self.compute_gradient()
        self.u_ite_dir[self.ite_l][self.ite_u] = - np.dot(np.linalg.inv(KKT), self.gradient_ite[self.ite_l][self.ite_u]) # np.linalg.inv(KKT)
        self.u_ite[self.ite_l][self.ite_u] = self.u_ite[self.ite_l][self.ite_u - 1] + \
                                             1 * self.u_ite_dir[self.ite_l][self.ite_u]
        for i in range(self.n):
            self.u_ite[self.ite_l][self.ite_u][i] = min(max(a_min, self.u_ite[self.ite_l][self.ite_u][i]), a_max)
            self.u_ite[self.ite_l][self.ite_u][i] = min(max((v_min - self.v_ite[self.ite_l][self.ite_u-1][i]) / tau,
                                                            self.u_ite[self.ite_l][self.ite_u][i]),
                                                        (v_max - self.v_ite[self.ite_l][self.ite_u-1][i]) / tau)

        self.calculate_ite_state()

    def calculate_ite_state(self):
        self.v_ite[self.ite_l][self.ite_u] = self.v[:] + tau * self.u_ite[self.ite_l][self.ite_u]  # cavs' next velocity
        self.x_ite[self.ite_l][self.ite_u] = self.x[:] + tau * self.v[:] + tau**2 /2 * self.u_ite[self.ite_l][self.ite_u] # cavs' next location
        self.zv_ite[self.ite_l][self.ite_u][0] = self.v_hdv + tau * self.u_hdv - self.v_ite[self.ite_l][self.ite_u][0] # hdv - first CAV
        self.zx_ite[self.ite_l][self.ite_u][0] = self.x_hdv + tau * self.v_hdv + tau**2/2 * self.u_hdv - self.x_ite[self.ite_l][self.ite_u][0] - sd
        for i in range(self.n - 1):
            self.zv_ite[self.ite_l][self.ite_u][i+1] = self.v_ite[self.ite_l][self.ite_u][i] - \
                                                     self.v_ite[self.ite_l][self.ite_u][i+1]
            self.zx_ite[self.ite_l][self.ite_u][i+1] = self.x_ite[self.ite_l][self.ite_u][i] - \
                                                     self.x_ite[self.ite_l][self.ite_u][i+1] - sd

    def compute_gradient(self):
        for i in range(self.n - 1):
            self.gradient_ite[self.ite_l][self.ite_u][i] = (tau**2/2) * (-self.zx_ite[self.ite_l][self.ite_u-1][i]
                                                           *alpha[i] + self.zx_ite[self.ite_l][self.ite_u-1][i+1]*alpha[i+1]) + \
                                                           tau**2 * self.u_ite[self.ite_l][self.ite_u-1][i] + \
                                                           tau * (-self.zv_ite[self.ite_l][self.ite_u-1][i]*beta[i] +
                                                                  self.zv_ite[self.ite_l][self.ite_u-1][i+1]*beta[i+1])
        self.gradient_ite[self.ite_l][self.ite_u][n-1] = -(tau**2/2) * self.zx_ite[self.ite_l][self.ite_u-1][n-1] * alpha[n-1]+\
                                                          tau**2 * self.u_ite[self.ite_l][self.ite_u-1][n-1] - \
                                                         tau * self.zv_ite[self.ite_l][self.ite_u-1][n-1]*beta[n-1]

    def check_primal_stop(self):
        for i in range(self.n):
            if abs(self.u_ite[self.ite_l][self.ite_u][i] - self.u_ite[self.ite_l][self.ite_u - 1][i]) > 0.01:
                return False
        return True

    def compute_dual(self):
        self.ite_l += 1
        self.l_ite[self.ite_l] = self.l_ite[self.ite_l - 1] + 1

    def check_dual_stop(self):
        for i in range(1, self.n):
            if self.x_ite[self.ite_l][self.ite_u][i-1] - self.x_ite[self.ite_l][self.ite_u][i] \
                    < L + tau * self.v_ite[self.ite_l][self.ite_u][i]:
                return False
        return True

    def update_state(self, k):
        self.u = self.u_ite[self.dual_idx][self.primal_idx]
        self.record_trajectory(k)

        self.x = self.x_ite[self.dual_idx][self.primal_idx]  # prev platoon MPC state: location
        self.v = self.v_ite[self.dual_idx][self.primal_idx]  # prev platoon MPC state: velocity
        self.u = np.zeros(self.n)  # prev platoon MPC input: acceleration
        self.initialize_control_variables(k+1)


    def record_trajectory(self, k):
        cur_df = {}
        cur_df["hdv_x"] = self.hdv[0][k]
        cur_df["hdv_v"] = self.hdv[1][k]
        cur_df["hdv_u"] = self.hdv[2][k]
        # cur_df["iteration_u"] = self.u_ite
        # cur_df["iteration_lambda"] = self.l_ite

        u_ite_record = [[] for _ in range(self.n)]
        for idx in range(self.primal_idx + 1):
            for i in range(self.n):
                u_ite_record[i].append(self.u_ite[self.dual_idx][idx][i])

        for i in range(self.n):
            cur_df[f"cav{i+1}_u"] = self.u[i]
            cur_df[f"cav{i+1}_v"] = self.v[i]
            cur_df[f"cav{i+1}_x"] = self.x[i]
            cur_df[f"cav{i+1}_l_ite"] = self.l_ite[self.ite_l-1][i]
            cur_df[f"cav{i+1}_u_ite"] = u_ite_record[i]

        cur_df[f"cav1_s"] = self.hdv[0][k] - self.x[0]
        for i in range(1, self.n):
            cur_df[f"cav{i+1}_s"] = self.x[i-1] - self.x[i]

        self.trajectory = self.trajectory.append(cur_df, ignore_index = True)

def output_to_plot():
    global i
    """Write the output in excel and figures"""
    # platoon.trajectory.to_excel("output.xlsx")
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1 = plt.gca()
    platoon.trajectory.plot(kind='line', x="time_step", y='hdv_x', ax=ax1)
    for i in range(1, n + 1):
        platoon.trajectory.plot(kind='line', x="time_step", y=f'cav{i}_x', ax=ax1)
    ax1.set_ylabel("distance (m)")
    ax1.set_xlabel("time (sec)")
    ax1.set_ylim([-500, 1500])
    ax1.set_title("distance-time plot")
    plt.savefig("distance-time.png")
    plt.show()
    ax2 = plt.gca()
    platoon.trajectory.plot(kind='line', x="time_step", y='hdv_v', ax=ax2)
    for i in range(1, n + 1):
        platoon.trajectory.plot(kind='line', x="time_step", y=f'cav{i}_v', ax=ax2)
    ax2.set_ylabel("velocity (m/s)")
    ax2.set_xlabel("time (sec)")
    ax2.set_ylim([0, 40])
    ax2.set_title("velocity-time plot")
    plt.savefig("velocity-time.png")
    plt.show()
    ax3 = plt.gca()
    platoon.trajectory.plot(kind='line', x="time_step", y='hdv_u', ax=ax3)
    for i in range(1, n + 1):
        platoon.trajectory.plot(kind='line', x="time_step", y=f'cav{i}_u', ax=ax3)
    ax3.set_ylabel("acceleration (m/s2)")
    ax3.set_xlabel("time (sec)")
    ax3.set_ylim([-6, 6])
    ax3.set_title("acceleration-time plot")
    plt.savefig("acceleration-time.png")
    plt.show()
    ax4 = plt.gca()
    for i in range(1, n + 1):
        platoon.trajectory.plot(kind='line', x="time_step", y=f'cav{i}_s', ax=ax4)
    ax4.set_ylabel("spacing (m)")
    ax4.set_xlabel("time (sec)")
    ax4.set_ylim([40, 60])
    ax4.set_title("spacing-time plot")
    plt.savefig("spacing-time.png")
    plt.show()


"""Parameter settings"""
n = 8
T = 50
tau = 1
L = 3
a_min = -6
a_max = 5
v_min = 0
v_max = 32
sd = 50
init_vel = 20
alpha = [0.3*n**2 - 0.6*(n-i) for i in range(1, n+1)]
beta = [0.4*n**2 - 1.2*(n-i) for i in range(1, n+1)]
Q_x = np.diag(alpha)
Q_v = np.diag(beta)
hessian = [((alpha[i] + alpha[i + 1]) * (tau ** 2 / 2) ** 2 + tau ** 2 * (1 + beta[i] + beta[i + 1])) for i in range(n - 1)]
hessian.append(alpha[n-1] * (tau ** 2 / 2) ** 2 + tau ** 2 + beta[n-1] * tau ** 2)
KKT = np.diag(hessian)

"""Simulated HDV acceleration/deceleration data"""
hdv = np.zeros((3, T)) # (x, v, u)

for t in range(10):
    hdv[2][t] = 1
for t in range(10, 20):
    hdv[2][t] = 0
for t in range(20, 30):
    hdv[2][t] = -2
for t in range(30, 35):
    hdv[2][t] = 3
for t in range(35, 40):
    hdv[2][t] = -3
for t in range(40, T):
    hdv[2][t] = 0

hdv[1][0] = init_vel # initial velocity = 20m/s
"""Note the dynamics be: x(t) = x(t-1) + tau*v(t-1) + tau^2/2*u(t)"""
for t in range(1, T):
    hdv[0][t] = hdv[0][t-1] + tau * hdv[1][t-1] + tau ** 2 / 2 * hdv[2][t-1]
    hdv[1][t] = hdv[1][t-1] + tau * hdv[2][t-1]

"""platoon initial information data: first is one HDV, then follow n many CAVs"""
cav_info = [[_ * (-sd) for _ in range(1, n+1)], [init_vel for _ in range(1, n+1)]]

columnName = ["hdv_x", "hdv_v", "hdv_u"] #, "iteration_u", "iteration_lambda"
for i in range(1, n+1):
    columnName += [f"cav{i}_x", f"cav{i}_s", f"cav{i}_v", f"cav{i}_u", f"cav{i}_u_ite", f"cav{i}_l_ite"]

platoon = Platoon(n=n, k=0, cav_info=cav_info, hdv_info=hdv, columnName=columnName)
trajectory = pd.DataFrame(columns=columnName)


"""Run the platoon controller"""
for k in range(0, T-1):
    platoon.mpc_algorithm(k)
platoon.trajectory["time_step"] = platoon.trajectory.index

# output_to_plot()