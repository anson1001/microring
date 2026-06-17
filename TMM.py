import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

def fit(
    # =========================================================
    # 0. 用户参数区：主要修改这里
    # =========================================================

    # ---------- 输入数据 ----------
    file_path = "multiplot_data.csv",           # 输入 CSV 文件路径
    wavelength_col = "wavelength_nm",           # 波长列名
    transmission_col = "transmission_dB",       # transmission 列名（单位 dB）
    input_already_normalized = False,           # True: 输入已是相对baseline的dB比值，不再做全局baseline归一化


    # ---------- baseline 提取参数 ----------
    baseline_savgol_window = 7001,              # SG 平滑窗口长度，尽量取奇数
    baseline_savgol_poly = 3,                   # SG 多项式阶数

    # ---------- resonance 搜索参数 ----------
    min_prominence_db = 4.0,                    # dip 至少多深（dB）才算有效 resonance
    min_distance_pts = 2000,                    # 相邻 resonance 最小间隔（点数）
    fit_half_window_pts = 2000,                 # 每个 resonance 局部拟合窗口半宽（点数）

    # ---------- TMM 拟合参数 ----------
    # 注意：在当前这版代码中，FSR 不再作为自由拟合参数，
    # 而是由“当前 resonance 与相邻 resonance 的波长间隔”固定下来。
    # 所以 fsr_guess_nm 仅在以下情况作为兜底值：
    #   1. 只检测到一个 resonance；
    #   2. 检测到的间隔非正或异常。
    fsr_guess_nm = 2.5,
    max_fit_trials = 3,

    # ---------- 器件物理参数 ----------
    ring_radius_um = 26.0,                     # microring 半径（um）
    ng_for_Q = 4.05,                            # 对应 mode 的 group index，用于计算 Q

    # ---------- 裸数据输出文件 ----------
    raw_data_csv = "raw_data.csv",

    # ---------- 全谱输出文件 ----------
    fit_results_csv = "microring_fit_results.csv",

    # ---------- 局部拟合图显示数量 ----------
    n_local_plots = 30,                          # 显示 loaded Q 最高的前几个 resonance 的局部拟合图

    # ---------- 统计图参数 ----------
    plot_statistics = True,                     # 是否绘制统计图
    hist_bins = 20,                             # Q 分布图柱子数

    # ---------- 不合理的 resonance 决定保留与否 ----------
    distil_resonances = True,                   # 统计地剔除不合理的 resonance（例如 Ql<1000 或 a>0.99 的），只保留合理的结果
    check_resonances = True,                   # 逐个resonance检查

    # ---------- 单个 resonance 精修功能 ----------
    # 逻辑：
    # 1) 先做全谱拟合
    # 2) 再手动指定一个小波长范围
    # 3) 从该范围内已拟合到的 resonance 中，选择一个峰重新精修拟合
    refit_single_resonance = False,
    single_fit_min_nm = 1260,                # 小范围下限（nm）
    single_fit_max_nm = 1360,                # 小范围上限（nm）

    single_fit_results_csv = "single_resonance_refit_result.csv",

    # ---------- 图片保存设置 ----------
    save_plots = True,                          # 是否保存图片
    show_plots = False,                          # 是否显示图片
    plot_dpi = 300,                             # 保存图片分辨率
):

    # =========================================================
    # 0.1 自动创建结果输出文件夹
    # =========================================================
    # 例如 normalized_TEK00184.csv -> 输出文件夹 normalized_TEK00184
    input_path = Path(file_path)
    output_dir = input_path.stem
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)

    # 把 CSV 输出路径也放进该文件夹
    raw_data_csv = str(Path(output_dir) / raw_data_csv)
    fit_results_csv = str(Path(output_dir) / fit_results_csv)
    single_fit_results_csv = str(Path(output_dir) / single_fit_results_csv)

    print(f"Begin TMM fitting for file: {file_path}")
    print(f"All results will be saved in folder: {output_dir}")


    # =========================================================
    # 工具函数：统一保存并显示图片
    # =========================================================
    def save_and_show(fig, filename):
        """
        统一管理图片保存与显示。
        """
        full_path = Path(output_dir) / filename
        if save_plots:
            fig.savefig(full_path, dpi=plot_dpi, bbox_inches="tight")
            print(f"Saved figure: {full_path}")
        if show_plots:
            plt.show()
        else:
            plt.close(fig)


    # =========================================================
    # 1. 读取数据
    # =========================================================
    df = pd.read_csv(file_path)

    wavelength = df[wavelength_col].to_numpy(dtype=float)
    transmission_db = df[transmission_col].to_numpy(dtype=float)

    # 按波长升序排序
    sort_idx = np.argsort(wavelength)
    wavelength = wavelength[sort_idx]
    transmission_db = transmission_db[sort_idx]

    # dB -> 线性功率 transmission
    # 若 transmission_dB = 10 * log10(T_lin)
    # 则
    #       T_lin = 10^(transmission_dB / 10)
    transmission_lin = 10 ** (transmission_db / 10.0)

    # 裸数据输出
    df.to_csv(raw_data_csv, index=False)
    print(f"Saved raw data to: {raw_data_csv}")

    # =========================================================
    # 2. baseline 提取函数
    # =========================================================
    def estimate_baseline_linear(x, y, window_length=401, polyorder=3):
        """
        估计慢变 baseline（在线性功率域中进行）。

        物理上，测得的 transmission 可近似写成：
            T_measured(lambda) = B(lambda) * T_ring(lambda)

        其中：
            B(lambda) : 系统慢变背景（fiber coupling、插损、源功率包络等）
            T_ring(lambda) : microring 的快速 resonance 响应

        因此本函数的目标是估计 B(lambda)。

        思路：
        1. 用 Savitzky-Golay 滤波做粗平滑；
        2. 仅保留高于 baseline0 的点做上包络修正；
        3. 再轻微平滑，得到更稳定的 baseline。

        这样可以尽量避免 resonance dips 把 baseline 拉低。
        """
        window_length = int(window_length)

        if window_length % 2 == 0:
            window_length += 1

        if window_length >= len(y):
            window_length = len(y) - 1
            if window_length % 2 == 0:
                window_length -= 1

        baseline0 = savgol_filter(y, window_length=window_length, polyorder=polyorder)

        # # 上包络修正：
        # # 理想情况下 baseline 应该更接近谱线“上沿”，
        # # 而不是被 resonances 的 dip 拖到下面。
        # mask_upper = y >= baseline0
        # if np.sum(mask_upper) > 10:
        #     spline = UnivariateSpline(
        #         x[mask_upper],
        #         y[mask_upper],
        #         s=len(x[mask_upper]) * 1e-4
        #     )
        #     baseline1 = spline(x)

        #     wl2 = min(window_length, len(y) - (1 - len(y) % 2))
        #     if wl2 % 2 == 0:
        #         wl2 -= 1
        #     if wl2 >= 5:
        #         baseline1 = savgol_filter(
        #             baseline1,
        #             window_length=wl2,
        #             polyorder=min(polyorder, 3)
        #         )
        #     baseline = baseline1
        # else:
        #     baseline = baseline0
        baseline = baseline0

        baseline = np.clip(baseline, 1e-12, None)
        return baseline

    if input_already_normalized:
        # 输入已是 P/P_baseline 的dB比值，转线性后直接作为归一化谱使用
        baseline_lin = np.ones_like(transmission_lin)
        transmission_norm_lin = transmission_lin.copy()
        transmission_norm_db = transmission_db.copy()
    else:
        baseline_lin = estimate_baseline_linear(
            wavelength,
            transmission_lin,
            window_length=baseline_savgol_window,
            polyorder=baseline_savgol_poly
        )
        # baseline 归一化
        # 根据
        #       T_measured(lambda) = B(lambda) * T_ring(lambda)
        # 归一化后
        #       T_norm(lambda) = T_measured(lambda) / B(lambda)
        # 理想情况下就接近纯 ring 响应 T_ring(lambda)
        transmission_norm_lin = transmission_lin / baseline_lin
        transmission_norm_db = 10 * np.log10(np.clip(transmission_norm_lin, 1e-12, None))

    # =========================================================
    # 3. 全谱自动寻找 resonance dips
    # =========================================================
    # resonance 在 transmission 中表现为 dip（谷），
    # 而 scipy.signal.find_peaks 默认找“峰”，
    # 所以我们对 -transmission_norm_db 找峰：
    #
    # 如果原始 dip_signal = -T_norm_db
    # 则 transmission 的“谷”变成了 dip_signal 中的“峰”
    dip_signal = -transmission_norm_db

    peak_indices, peak_props = find_peaks(
        dip_signal,
        prominence=min_prominence_db,
        distance=min_distance_pts
    )

    # print(f"Found {len(peak_indices)} candidate resonances in the whole spectrum.")

    peak_wavelengths = wavelength[peak_indices]

    # =========================================================
    # 3.1 为每个 resonance 计算固定 FSR
    #     规则：
    #     - 非最后一个峰：FSR = 当前峰与下一个峰之间的间隔
    #     - 最后一个峰：FSR = 当前峰与前一个峰之间的间隔
    #     - 若只找到一个峰：FSR = fsr_guess_nm
    # =========================================================
    def compute_fixed_fsr_from_detected_peaks(peak_wavelengths, fallback_fsr):
        """
        根据检测到的 resonance 中心波长，为每个峰分配一个固定 FSR。

        对 ring 来说，相邻 resonance 满足 round-trip phase 增加 2*pi。
        因此相邻峰之间的波长间隔就是局部 FSR：
            FSR_i ≈ lambda_{i+1} - lambda_i

        这里按你的要求：
        - 对第 i 个峰，用它与下一个峰的间隔作为固定 FSR；
        - 对最后一个峰，用它与前一个峰的间隔；
        - 若只有一个峰，使用 fallback_fsr。

        Parameters
        ----------
        peak_wavelengths : ndarray
            所有检测到的 resonance 中心波长（已升序）
        fallback_fsr : float
            如果峰数量不足时使用的兜底 FSR

        Returns
        -------
        fsr_list : ndarray
            与 peak_wavelengths 一一对应的固定 FSR
        """
        n = len(peak_wavelengths)
        fsr_list = np.full(n, fallback_fsr, dtype=float)

        if n == 0:
            return fsr_list

        if n == 1:
            fsr_list[0] = fallback_fsr
            return fsr_list

        for i in range(n):
            if i < n - 1:
                fsr_list[i] = peak_wavelengths[i + 1] - peak_wavelengths[i]
            else:
                fsr_list[i] = peak_wavelengths[i] - peak_wavelengths[i - 1]

        # 防止异常非正值
        fsr_list = np.where(fsr_list > 0, fsr_list, fallback_fsr)
        return fsr_list


    fixed_fsr_list = compute_fixed_fsr_from_detected_peaks(peak_wavelengths, fsr_guess_nm)


    # =========================================================
    # 4. 定义 double-bus microring through-port TMM 模型（FSR固定版本）
    # =========================================================
    def doublebus_through_ring_linear_fixed_fsr(lambda_nm, t, a, lambda0, B0, B1, fsr_nm):
        """
        Double-bus (add-drop) microring 的 through-port 局部线性功率 transmission 模型。
        这里 FSR 不是拟合参数，而是外部固定传入。

        ---------------------------------------------------------
        1. all-pass ring 的场传输函数
        ---------------------------------------------------------
        对于 double-bus ring，through-port 的场传输函数可写为：

            E_through / E_in = (t - a*t*exp(-j*phi)) / (1 - a*t^2*exp(-j*phi))

        其中：
            t    : 两个耦合器共用的 self-coupling amplitude coefficient (t1=t2=t)
            a    : 腔内一圈 intrinsic amplitude transmission
            phi  : round-trip phase

        ---------------------------------------------------------
        2. 功率 transmission
        ---------------------------------------------------------
        实验测到的是功率 transmission，因此：

            T_ring(lambda) = |E_out / E_in|^2

        展开后得到：

            numerator   = t^2 + (a*t)^2 - 2*a*t^2*cos(phi)
            denominator = 1 + (a*t^2)^2 - 2*a*t^2*cos(phi)

            T_ring = numerator / denominator

        ---------------------------------------------------------
        3. 局部相位线性化
        ---------------------------------------------------------
        round-trip phase 严格写法是：

            phi(lambda) = beta(lambda) * L
                        = 2*pi*n_eff(lambda)*L / lambda

        在单个 resonance 附近做局部线性化后，可写成：

            phi(lambda) ≈ 2*pi*(lambda - lambda0)/FSR

        这里 FSR 由相邻 resonance 间隔固定给出，而不是再作为拟合参数。

        ---------------------------------------------------------
        4. 局部背景项
        ---------------------------------------------------------
        由于全局 baseline 去除后仍可能残留局部缓慢背景，
        这里再乘一个一阶局部背景：

            Background(lambda) = B0 + B1*(lambda - lambda0)

        因此最终模型为：

            T_local(lambda) = Background(lambda) * T_ring(lambda)
        """
        phi = 2 * np.pi * (lambda_nm - lambda0) / fsr_nm

        numerator = t**2 + (a * t)**2 - 2 * a * t**2 * np.cos(phi)
        denominator = 1 + (a * t**2)**2 - 2 * a * t**2 * np.cos(phi)
        T_ring = numerator / denominator

        background = B0 + B1 * (lambda_nm - lambda0)
        return background * T_ring


    def build_model_with_fixed_fsr(fsr_nm):
        """
        返回一个把 FSR 固定住的拟合函数，供 curve_fit 调用。

        这样 curve_fit 只拟合：
            t, a, lambda0, B0, B1
        而 FSR 通过闭包固定为指定值 fsr_nm。
        """
        def model(lambda_nm, t, a, lambda0, B0, B1):
            return doublebus_through_ring_linear_fixed_fsr(lambda_nm, t, a, lambda0, B0, B1, fsr_nm)
        return model


    # =========================================================
    # 5. 一些物理量计算函数
    # =========================================================
    def calc_kappa_from_t(t):
        """
        在假设 coupler 无损时：

            t^2 + kappa^2 = 1

        因此：
            kappa^2 = 1 - t^2
            kappa   = sqrt(1 - t^2)

        这里：
            t       : self-coupling amplitude coefficient
            kappa   : cross-coupling amplitude coefficient
            kappa^2 : power coupling ratio
        """
        t = np.clip(t, 0.0, 1.0)
        kappa_sq = max(0.0, 1.0 - t**2)
        kappa = np.sqrt(kappa_sq)
        return kappa, kappa_sq


    def calc_roundtrip_length_cm(radius_um):
        """
        计算 ring 一圈长度（cm）：

            L = 2*pi*R

        其中 R 为 ring 半径。
        """
        radius_cm = radius_um * 1e-4
        return 2 * np.pi * radius_cm


    def calc_Q_from_amp_factor(amp_factor, lambda0_nm, ng, radius_um):
        """
        根据 amplitude round-trip factor 计算 Q。

        ---------------------------------------------------------
        1. round-trip time
        ---------------------------------------------------------
        光在 ring 中走一圈的时间为：

            tau_rt = n_g * L / c

        其中：
            n_g : group index
            L   : ring 一圈长度
            c   : 光速

        ---------------------------------------------------------
        2. 若每一圈功率乘上一个因子 amp_factor^2
        ---------------------------------------------------------
        则有：

            P_{m+1} = P_m * amp_factor^2

        把离散 round-trip 衰减等效成连续指数衰减，可得到：

            Q = - 2*pi*n_g*L / [ lambda * ln(amp_factor^2) ]

        ---------------------------------------------------------
        3. 在本代码中的四种用法（double-bus, 且 t1=t2=t）
        ---------------------------------------------------------
            amp_factor = a          -> intrinsic Q,  记作 Qi
            amp_factor = t          -> 单个耦合器的 coupling Q, 记作 Qc_single
            两个耦合器总耦合满足：1/Qc_total = 1/Qc1 + 1/Qc2 = 2/Qc_single
            amp_factor = a*t^2      -> loaded Q,     记作 Ql

        这对应：
            Qi  = -2*pi*n_g*L / [lambda0 * ln(a^2)]
            Qc_single = -2*pi*n_g*L / [lambda0 * ln(t^2)]
            Ql       = -2*pi*n_g*L / [lambda0 * ln((a*t^2)^2)]

        并满足：
            1/Ql = 1/Qi + 2/Qc_single
        """
        amp_factor = np.clip(amp_factor, 1e-15, 1 - 1e-15)
        L_m = 2 * np.pi * radius_um * 1e-6
        lambda0_m = lambda0_nm * 1e-9

        Q = -2 * np.pi * ng * L_m / (lambda0_m * np.log(amp_factor**2))
        return Q


    def calc_prop_loss_db_per_cm(a, radius_um):
        """
        根据 round-trip amplitude transmission a 计算传播损耗（dB/cm）。

        若一圈后的振幅因子为：
            E -> aE

        则功率因子为：
            P -> a^2 P

        功率损耗对应的 dB 表示为：
            loss_roundtrip_dB = -10*log10(a^2) = -20*log10(a)

        若 round-trip length 为 L_cm，则传播损耗（dB/cm）为：

            propagation loss (dB/cm) = -20*log10(a) / L_cm
        """
        a = np.clip(a, 1e-15, 1.0)
        L_cm = calc_roundtrip_length_cm(radius_um)
        loss_db_per_cm = -20 * np.log10(a) / L_cm
        return loss_db_per_cm


    # =========================================================
    # 6. 全谱逐个 resonance 做局部 TMM 拟合（FSR固定）
    # =========================================================
    fit_results = []

    for i, idx0 in enumerate(peak_indices):
        left = max(0, idx0 - fit_half_window_pts)
        right = min(len(wavelength), idx0 + fit_half_window_pts + 1)

        x_fit = wavelength[left:right]
        y_fit = transmission_norm_lin[left:right]

        if len(x_fit) < 20:
            continue

        lambda0_guess = wavelength[idx0]
        fsr_fixed = fixed_fsr_list[i]

        edge_n = max(3, len(x_fit) // 8)
        B0_guess = np.mean(np.r_[y_fit[:edge_n], y_fit[-edge_n:]])
        B1_guess = 0.0

        # 因为 FSR 已固定，这里只拟合：
        #   t, a, lambda0, B0, B1
        initial_guesses = [
            [0.95, 0.95, lambda0_guess, B0_guess, B1_guess],
            [0.90, 0.98, lambda0_guess, B0_guess, B1_guess],
            [0.98, 0.90, lambda0_guess, B0_guess, B1_guess],
        ]

        # 参数边界：
        # t, a 一般约束在 [0, 1]
        # lambda0 约束在当前局部窗口内
        # B0, B1 给一个较宽松范围
        lower_bounds = [0.0, 0.0, x_fit.min(), 0.5, -100.0]
        upper_bounds = [1.0, 1.0, x_fit.max(), 1.5, 100.0]

        popt_best = None
        pcov_best = None
        rss_best = np.inf

        model_fixed = build_model_with_fixed_fsr(fsr_fixed)

        for p0 in initial_guesses[:max_fit_trials]:
            try:
                popt, pcov = curve_fit(
                    model_fixed,
                    x_fit,
                    y_fit,
                    p0=p0,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=50000
                )

                y_model = model_fixed(x_fit, *popt)

                # 残差平方和：
                # RSS = Σ (y_data - y_model)^2
                rss = np.sum((y_fit - y_model) ** 2)

                if rss < rss_best:
                    rss_best = rss
                    popt_best = popt
                    pcov_best = pcov

            except Exception:
                continue

        if popt_best is None:
            print(f"[{i}] Fit failed near {lambda0_guess:.6f} nm")
            continue

        t_fit, a_fit, lambda0_fit, B0_fit, B1_fit = popt_best
        perr = np.sqrt(np.diag(pcov_best)) if pcov_best is not None else [np.nan] * 5

        # 在共振中心处的 transmission（拟合值）
        T_res_lin = doublebus_through_ring_linear_fixed_fsr(
            np.array([lambda0_fit]), t_fit, a_fit, lambda0_fit, B0_fit, B1_fit, fsr_fixed
        )[0]
        T_res_db = 10 * np.log10(max(T_res_lin, 1e-12))

        # 对称双耦合器情形下不再区分 t1/t2 的耦合区。
        regime = "symmetric-double-bus"

        kappa_fit, kappa_sq_fit = calc_kappa_from_t(t_fit)

        # intrinsic Q：由 a 计算
        try:
            Qi_fit = calc_Q_from_amp_factor(a_fit, lambda0_fit, ng_for_Q, ring_radius_um)
        except Exception:
            Qi_fit = np.nan

        # 单个耦合器 Qc_single：由 t 计算
        try:
            Qc_single_fit = calc_Q_from_amp_factor(t_fit, lambda0_fit, ng_for_Q, ring_radius_um)
        except Exception:
            Qc_single_fit = np.nan

        # loaded Q：由 a*t^2 计算
        try:
            Ql_fit = calc_Q_from_amp_factor(a_fit * t_fit**2, lambda0_fit, ng_for_Q, ring_radius_um)
        except Exception:
            Ql_fit = np.nan

        # 一致性检查：
        #   1/Ql = 1/Qi + 2/Qc_single
        if (
            np.isfinite(Qi_fit) and np.isfinite(Qc_single_fit)
            and Qi_fit > 0 and Qc_single_fit > 0
        ):
            Ql_from_sum = 1.0 / (1.0 / Qi_fit + 2.0 / Qc_single_fit)
        else:
            Ql_from_sum = np.nan

        # propagation loss（dB/cm）
        try:
            prop_loss_db_cm = calc_prop_loss_db_per_cm(a_fit, ring_radius_um)
        except Exception:
            prop_loss_db_cm = np.nan

        # 用 loaded Q 估算 linewidth：
        #   Ql = lambda0 / Delta_lambda
        # => Delta_lambda = lambda0 / Ql
        if np.isfinite(Ql_fit) and Ql_fit > 0:
            linewidth_nm = lambda0_fit / Ql_fit
        else:
            linewidth_nm = np.nan

        fit_results.append({
            "peak_index": idx0,
            "lambda0_nm": lambda0_fit,
            "lambda0_err_nm": perr[2],
            "t": t_fit,
            "t_err": perr[0],
            "a": a_fit,
            "a_err": perr[1],
            "kappa": kappa_fit,
            "kappa_sq": kappa_sq_fit,
            "FSR_nm": fsr_fixed,          # 固定值
            "FSR_err_nm": 0.0,            # 固定参数，不参与拟合
            "B0": B0_fit,
            "B1": B1_fit,
            "T_res_linear": T_res_lin,
            "T_res_dB": T_res_db,
            "Qi": Qi_fit,
            "Qc_single": Qc_single_fit,
            "Ql": Ql_fit,
            "Ql_from_sum_rule": Ql_from_sum,
            "linewidth_nm_est": linewidth_nm,
            "prop_loss_dB_per_cm": prop_loss_db_cm,
            "RSS": rss_best,
            "coupling_regime": regime,
        })

    print(f"Successfully fitted {len(fit_results)} resonances.")
    results_df = pd.DataFrame(fit_results)
    results_df = results_df.replace([np.inf, -np.inf], np.nan)
    results_df = results_df[np.isfinite(results_df["Ql"]) & (results_df["Ql"] > 0)]

    # =========================================================
    # 7. 剔除不合理的 resonance
    # =========================================================

    def on_key(event):
        global user_decision # Tell the function to use the variable outside
        if event.key == 'y':
            user_decision = True
            plt.close()
        elif event.key == 'n':
            user_decision = False
            plt.close()

    # Omit bad results based on loaded Q, coupling coefficient, and extinction ratio:
    if distil_resonances == True:
        original_peak_no = len(results_df['Ql'])
        # List of columns to clean
        cols = ['Ql', 'a', 't']

        # --- STAGE 1: Top/Bottom 10% Quantile Trim ---
        for col in cols:
            lower_quant = results_df[col].quantile(0.10)
            upper_quant = results_df[col].quantile(0.90)
            results_df = results_df[(results_df[col] >= lower_quant) & (results_df[col] <= upper_quant)]

        # --- STAGE 2: 1 Standard Deviation Filter ---
        for col in cols:
            m = results_df[col].median()
            std = results_df[col].std()
        
        # Keep only data within [m - std, m + std]
        results_df = results_df[(results_df[col] >= (m - std)) & (results_df[col] <= (m + std))]
        remaining_peak_no = len(results_df['Ql'])
        print(f"Omitting Bad Results: \nOriginal Peak Number: {original_peak_no}\nRemaining Peak Number: {remaining_peak_no}")
    
    # manually check each resonance and decide whether to keep it:
    # if check_resonances == True:
    #     plot_df = results_df.copy()
    #     # plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    #     # plot_df = plot_df[np.isfinite(plot_df["Ql"]) & (plot_df["Ql"] > 0)]
    #     # plot_df = plot_df.sort_values("Ql", ascending=False).head(n_local_plots).reset_index(drop=True)

    #     print("Now checking each resonance fit result.")
    #     print("For each plot, press 'y' to keep this resonance, or 'n' to discard it.")
    #     for k in range(len(plot_df)):
    #         row = plot_df.iloc[k]
    #         lambda0 = row["lambda0_nm"]
    #         fsr_fixed = row["FSR_nm"]

    #         idx_center = np.argmin(np.abs(wavelength - lambda0))
    #         left = max(0, idx_center - fit_half_window_pts)
    #         right = min(len(wavelength), idx_center + fit_half_window_pts + 1)

    #         x_fit = wavelength[left:right]
    #         y_fit = transmission_norm_lin[left:right]

    #         y_model_lin = doublebus_through_ring_linear_fixed_fsr(
    #             x_fit, row["t"], row["a"], row["lambda0_nm"], row["B0"], row["B1"], fsr_fixed
    #         )

    #         fig = plt.figure(figsize=(7, 4))
    #         plt.plot(
    #             x_fit,
    #             y_fit,
    #             'o',
    #             ms=4,
    #             label="data"
    #         )
    #         plt.plot(
    #             x_fit,
    #             y_model_lin,
    #             '-',
    #             lw=2,
    #             label="TMM fit"
    #         )
    #         plt.xlabel("Wavelength (nm)")
    #         plt.ylabel("Normalized Transmission (linear)")
    #         plt.title(f"Top-Ql local fit #{k+1}: {lambda0:.4f} nm, Ql={row['Ql']:.3e}")
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show(block=False)
    #         if event.key == 'y':
    #             continue
    #         elif event.key == 'n':
    #             results_df = results_df.drop(plot_df.index[k])
    #         plt.close()

    if check_resonances == True:
        plot_df = results_df.copy()
        print("Now checking each resonance fit result.")
        print("Keys: [y] Keep, [n] Discard, [q] Quit Review")

        # 1. Define a place to store the key pressed
        key_pressed = [None] 

        # 2. Define the callback function
        def on_key(event):
            if event.key in ['y', 'n', 'q']:
                key_pressed[0] = event.key
                plt.close() # This breaks the plt.show() block

        indices_to_drop = []

        for k in range(len(plot_df)):
            key_pressed[0] = None # Reset for this iteration
            row = plot_df.iloc[k]
            
            # ... (your existing data selection and y_model_lin math) ...
            lambda0 = row["lambda0_nm"]
            fsr_fixed = row["FSR_nm"]
            idx_center = np.argmin(np.abs(wavelength - lambda0))
            left = max(0, idx_center - fit_half_window_pts)
            right = min(len(wavelength), idx_center + fit_half_window_pts + 1)
            x_fit = wavelength[left:right]
            y_fit = transmission_norm_lin[left:right]
            y_model_lin = doublebus_through_ring_linear_fixed_fsr(
                x_fit, row["t"], row["a"], row["lambda0_nm"], row["B0"], row["B1"], fsr_fixed
            )

            fig = plt.figure(figsize=(7, 4))
            
            # 3. Connect the event handler to this specific figure
            fig.canvas.mpl_connect('key_press_event', on_key)

            plt.plot(x_fit, y_fit, 'o', ms=4, label="data")
            plt.plot(x_fit, y_model_lin, '-', lw=2, label="TMM fit")
            plt.title(f"Fit #{k+1}: {lambda0:.4f} nm, Ql={row['Ql']:.3e}\n[y] Keep  [n] Discard")
            plt.legend()
            plt.tight_layout()

            # 4. Use block=True here! 
            # Because plt.close() is in our function, it will "unblock" when you hit a key.
            plt.show(block=True) 

            # 5. Act on the decision
            if key_pressed[0] == 'n':
                print(f"Discarding resonance at {lambda0:.2f} nm")
                indices_to_drop.append(plot_df.index[k])
            elif key_pressed[0] == 'q':
                print("Stopping review early...")
                break
            else:
                print(f"Kept resonance at {lambda0:.2f} nm")

        # Drop the rejected ones from the original dataframe
        results_df = results_df.drop(indices_to_drop)

    # =========================================================
    # 8. 保存全谱拟合结果 / 原始谱 / baseline / 归一化谱 / 检测到的 resonance / loaded Q 最高的前几个 resonance 的局部拟合图
    # =========================================================
    if len(results_df) == 0:
        raise RuntimeError("No resonance was successfully fitted. Please check your parameters.")

    results_df = results_df.sort_values("lambda0_nm").reset_index(drop=True)
    results_df.to_csv(fit_results_csv, index=False)
    # print(f"Saved full-spectrum fit results to: {fit_results_csv}")

    fig = plt.figure(figsize=(10, 4))
    plt.plot(wavelength, transmission_db, label="Measured (dB)")
    plt.plot(
        wavelength,
        10 * np.log10(np.clip(baseline_lin, 1e-12, None)),
        label="Estimated baseline (dB)"
    )
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission (dB)")
    plt.title("Measured spectrum and estimated baseline")
    plt.legend()
    plt.tight_layout()
    save_and_show(fig, "01_measured_spectrum_and_baseline.png")

    fig = plt.figure(figsize=(10, 4))
    plt.plot(wavelength, transmission_norm_db, label="Baseline-normalized (dB)")
    plt.plot(
        wavelength[peak_indices],
        transmission_norm_db[peak_indices],
        'ro',
        ms=4,
        label="Detected resonances"
    )
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Transmission (dB)")
    plt.title("Normalized spectrum and detected resonances")
    plt.legend()
    plt.tight_layout()
    save_and_show(fig, "02_normalized_spectrum_and_detected_resonances.png")

    # =========================================================
    # 9. 绘制 loaded Q 最高的前几个 resonance 的局部拟合图
    # =========================================================
    plot_df = results_df.copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df[np.isfinite(plot_df["Ql"]) & (plot_df["Ql"] > 0)]
    plot_df = plot_df.sort_values("Ql", ascending=False).head(n_local_plots).reset_index(drop=True)

    for k in range(len(plot_df)):
        row = plot_df.iloc[k]
        lambda0 = row["lambda0_nm"]
        fsr_fixed = row["FSR_nm"]

        idx_center = np.argmin(np.abs(wavelength - lambda0))
        left = max(0, idx_center - fit_half_window_pts)
        right = min(len(wavelength), idx_center + fit_half_window_pts + 1)

        x_fit = wavelength[left:right]
        y_fit = transmission_norm_lin[left:right]

        y_model_lin = doublebus_through_ring_linear_fixed_fsr(
            x_fit, row["t"], row["a"], row["lambda0_nm"], row["B0"], row["B1"], fsr_fixed
        )

        fig = plt.figure(figsize=(7, 4))
        plt.plot(
            x_fit,
            y_fit,
            'o',
            ms=4,
            label="data"
        )
        plt.plot(
            x_fit,
            y_model_lin,
            '-',
            lw=2,
            label="TMM fit"
        )
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Normalized Transmission (linear)")
        plt.title(f"Top-Ql local fit #{k+1}: {lambda0:.4f} nm, Ql={row['Ql']:.3e}")
        plt.legend()
        plt.tight_layout()
        save_and_show(fig, f"03_topQl_local_fit_{k+1:02d}_{lambda0:.4f}nm.png")


    # =========================================================
    # 10. 绘制统计图：中心波长 vs Ql/Qi + Ql/Qi 分布图
    # =========================================================
    if plot_statistics:
        valid_q = results_df.copy()
        valid_q = valid_q.replace([np.inf, -np.inf], np.nan)

        valid_Ql = valid_q[np.isfinite(valid_q["Ql"]) & (valid_q["Ql"] > 0)]
        valid_Qi = valid_q[np.isfinite(valid_q["Qi"]) & (valid_q["Qi"] > 0)]

        fig = plt.figure(figsize=(7, 4))
        plt.plot(valid_Ql["lambda0_nm"], valid_Ql["Ql"], 'o-', ms=4)
        plt.xlabel("Center wavelength (nm)")
        plt.ylabel("Loaded Q (Ql)")
        plt.title("Center wavelength vs Loaded Q")
        plt.tight_layout()
        save_and_show(fig, "04_center_wavelength_vs_Ql.png")

        fig = plt.figure(figsize=(7, 4))
        plt.plot(valid_Qi["lambda0_nm"], valid_Qi["Qi"], 'o-', ms=4)
        plt.xlabel("Center wavelength (nm)")
        plt.ylabel("Intrinsic Q (Qi)")
        plt.title("Center wavelength vs Intrinsic Q")
        plt.tight_layout()
        save_and_show(fig, "05_center_wavelength_vs_Qi.png")

        fig = plt.figure(figsize=(7, 4))
        plt.hist(valid_Ql["Ql"], bins=hist_bins)
        plt.xlabel("Loaded Q (Ql)")
        plt.ylabel("Counts")
        plt.title("Distribution of Loaded Q")
        plt.tight_layout()
        save_and_show(fig, "06_distribution_of_Ql.png")

        fig = plt.figure(figsize=(7, 4))
        plt.hist(valid_Qi["Qi"], bins=hist_bins)
        plt.xlabel("Intrinsic Q (Qi)")
        plt.ylabel("Counts")
        plt.title("Distribution of Intrinsic Q")
        plt.tight_layout()
        save_and_show(fig, "07_distribution_of_Qi.png")


    # =========================================================
    # 11. 单个 resonance 精修模块（FSR 同样固定）
    # =========================================================
    if refit_single_resonance:
        print("\n========== Single-resonance refit mode ==========")
        print(f"Selected wavelength range: [{single_fit_min_nm:.6f}, {single_fit_max_nm:.6f}] nm")

        candidate_rows = results_df[
            (results_df["lambda0_nm"] >= single_fit_min_nm) &
            (results_df["lambda0_nm"] <= single_fit_max_nm)
        ].copy()

        if len(candidate_rows) == 0:
            print("No fitted resonance center found in the selected range.")
            print("Please enlarge the selected range or check the full-spectrum fit results.")

        else:
            # 若该范围内有多个 resonance，默认选 dip 最深的一个
            candidate_rows = candidate_rows.sort_values("T_res_dB")
            best_row = candidate_rows.iloc[0]

            lambda0_seed = best_row["lambda0_nm"]
            fsr_fixed = best_row["FSR_nm"]
            print(f"Chosen resonance center for refit: {lambda0_seed:.6f} nm")
            print(f"Fixed FSR used in refit: {fsr_fixed:.6f} nm")

            range_mask = (wavelength >= single_fit_min_nm) & (wavelength <= single_fit_max_nm)
            x_single = wavelength[range_mask]
            y_single_lin = transmission_norm_lin[range_mask]
            y_single_db = transmission_norm_db[range_mask]

            if len(x_single) < 20:
                print("Too few data points in the selected range. Please enlarge the range.")
            else:
                local_min_idx = np.argmin(y_single_lin)
                lambda0_guess = x_single[local_min_idx]

                edge_n = max(3, len(x_single) // 8)
                B0_guess = np.mean(np.r_[y_single_lin[:edge_n], y_single_lin[-edge_n:]])
                B1_guess = 0.0

                # 单峰精修时仍然固定 FSR，仅拟合：
                #   t, a, lambda0, B0, B1
                initial_guesses_single = [
                    [best_row["t"], best_row["a"], lambda0_guess, B0_guess, B1_guess],
                    [0.95, 0.95, lambda0_guess, B0_guess, B1_guess],
                    [0.90, 0.98, lambda0_guess, B0_guess, B1_guess],
                    [0.98, 0.90, lambda0_guess, B0_guess, B1_guess],
                ]

                lower_bounds = [0.0, 0.0, x_single.min(), 0.5, -100.0]
                upper_bounds = [1.0, 1.0, x_single.max(), 1.5, 100.0]

                popt_best = None
                pcov_best = None
                rss_best = np.inf

                model_fixed = build_model_with_fixed_fsr(fsr_fixed)

                for p0 in initial_guesses_single:
                    try:
                        popt, pcov = curve_fit(
                            model_fixed,
                            x_single,
                            y_single_lin,
                            p0=p0,
                            bounds=(lower_bounds, upper_bounds),
                            maxfev=50000
                        )

                        y_model = model_fixed(x_single, *popt)
                        rss = np.sum((y_single_lin - y_model) ** 2)

                        if rss < rss_best:
                            rss_best = rss
                            popt_best = popt
                            pcov_best = pcov

                    except Exception:
                        continue

                if popt_best is None:
                    print("Single-resonance refit failed.")

                else:
                    t_fit, a_fit, lambda0_fit, B0_fit, B1_fit = popt_best
                    perr = np.sqrt(np.diag(pcov_best)) if pcov_best is not None else [np.nan] * 5

                    T_res_lin = doublebus_through_ring_linear_fixed_fsr(
                        np.array([lambda0_fit]), t_fit, a_fit, lambda0_fit, B0_fit, B1_fit, fsr_fixed
                    )[0]
                    T_res_db = 10 * np.log10(max(T_res_lin, 1e-12))

                    regime = "symmetric-double-bus"

                    kappa_fit, kappa_sq_fit = calc_kappa_from_t(t_fit)

                    try:
                        Qi_fit = calc_Q_from_amp_factor(a_fit, lambda0_fit, ng_for_Q, ring_radius_um)
                    except Exception:
                        Qi_fit = np.nan

                    try:
                        Qc_single_fit = calc_Q_from_amp_factor(t_fit, lambda0_fit, ng_for_Q, ring_radius_um)
                    except Exception:
                        Qc_single_fit = np.nan

                    try:
                        Ql_fit = calc_Q_from_amp_factor(a_fit * t_fit**2, lambda0_fit, ng_for_Q, ring_radius_um)
                    except Exception:
                        Ql_fit = np.nan

                    if (
                        np.isfinite(Qi_fit) and np.isfinite(Qc_single_fit)
                        and Qi_fit > 0 and Qc_single_fit > 0
                    ):
                        Ql_from_sum = 1.0 / (1.0 / Qi_fit + 2.0 / Qc_single_fit)
                    else:
                        Ql_from_sum = np.nan

                    try:
                        prop_loss_db_cm = calc_prop_loss_db_per_cm(a_fit, ring_radius_um)
                    except Exception:
                        prop_loss_db_cm = np.nan

                    if np.isfinite(Ql_fit) and Ql_fit > 0:
                        linewidth_nm = lambda0_fit / Ql_fit
                    else:
                        linewidth_nm = np.nan

                    single_result = pd.DataFrame([{
                        "selected_range_min_nm": single_fit_min_nm,
                        "selected_range_max_nm": single_fit_max_nm,
                        "lambda0_nm": lambda0_fit,
                        "lambda0_err_nm": perr[2],
                        "t": t_fit,
                        "t_err": perr[0],
                        "a": a_fit,
                        "a_err": perr[1],
                        "kappa": kappa_fit,
                        "kappa_sq": kappa_sq_fit,
                        "FSR_nm": fsr_fixed,
                        "FSR_err_nm": 0.0,
                        "B0": B0_fit,
                        "B1": B1_fit,
                        "T_res_linear": T_res_lin,
                        "T_res_dB": T_res_db,
                        "Qi": Qi_fit,
                        "Qc_single": Qc_single_fit,
                        "Ql": Ql_fit,
                        "Ql_from_sum_rule": Ql_from_sum,
                        "linewidth_nm_est": linewidth_nm,
                        "prop_loss_dB_per_cm": prop_loss_db_cm,
                        "RSS": rss_best,
                        "coupling_regime": regime,
                    }])

                    single_result.to_csv(single_fit_results_csv, index=False)
                    print(f"Saved single-resonance refit result to: {single_fit_results_csv}")

                    print("\nSingle-resonance refit result:")
                    print(single_result.T)

                    y_model_single_lin = doublebus_through_ring_linear_fixed_fsr(
                        x_single, t_fit, a_fit, lambda0_fit, B0_fit, B1_fit, fsr_fixed
                    )

                    fig = plt.figure(figsize=(7, 4))
                    plt.plot(
                        x_single,
                        y_single_lin,
                        'o',
                        ms=4,
                        label="data"
                    )
                    plt.plot(
                        x_single,
                        y_model_single_lin,
                        '-',
                        lw=2,
                        label="single-resonance refit"
                    )
                    plt.xlabel("Wavelength (nm)")
                    plt.ylabel("Normalized Transmission (linear)")
                    plt.title(
                        f"Single resonance refit in [{single_fit_min_nm:.4f}, {single_fit_max_nm:.4f}] nm"
                    )
                    plt.legend()
                    plt.tight_layout()
                    save_and_show(fig, "08_single_resonance_refit.png")


    # =========================================================
    # 12. 结束时输出参数平均值（全谱拟合结果）
    # =========================================================
    avg_df = results_df.replace([np.inf, -np.inf], np.nan).copy()

    def safe_mean(series, positive_only=False):
        s = pd.to_numeric(series, errors="coerce")
        if positive_only:
            s = s[s > 0]
        s = s[np.isfinite(s)]
        if len(s) == 0:
            return np.nan
        return float(np.mean(s))

    avg_a = safe_mean(avg_df["a"])
    avg_t = safe_mean(avg_df["t"])
    avg_Qi = safe_mean(avg_df["Qi"], positive_only=True)
    avg_Qc_single = safe_mean(avg_df["Qc_single"], positive_only=True)
    avg_Ql = safe_mean(avg_df["Ql"], positive_only=True)

    # print("\n========== Average fitted parameters (full spectrum) ==========")
    # print(f"Average a          : {avg_a:.6f}" if np.isfinite(avg_a) else "Average a          : nan")
    # print(f"Average t          : {avg_t:.6f}" if np.isfinite(avg_t) else "Average t          : nan")
    # print(f"Average Qi         : {avg_Qi:.3e}" if np.isfinite(avg_Qi) else "Average Qi         : nan")
    # print(
    #     f"Average Qc_single  : {avg_Qc_single:.3e}"
    #     if np.isfinite(avg_Qc_single)
    #     else "Average Qc_single  : nan"
    # )
    # print(f"Average Ql         : {avg_Ql:.3e}" if np.isfinite(avg_Ql) else "Average Ql         : nan")
    print(f"Complete! All results saved to: {fit_results_csv}")

    return results_df #, avg_a, avg_t, avg_Qi, avg_Qc_single, avg_Ql