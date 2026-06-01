# Gate10 Ac-225 到 CUDA/OSEM 适配说明

本目录是独立适配目录，不修改 `/Users/chen/Documents/OSEM-origin` 原始代码。原始代码快照保留在 `original_snapshot/` 用于对照。

## 当前正式路径

- Gate10 模型：`/Users/chen/Documents/gate10_Ac225_Compton_PET/gate10-geoonly`
- Gate10 Python 包：`/Users/chen/Documents/gate10_Ac225_Compton_PET/gate10-geoonly/gate10_model`
- 正式 Gate10 ROOT：`/Users/chen/Documents/gate10_Ac225_Compton_PET/gate10-geoonly/gate10_output/run_trackaware_8threads_40min_ac225_fixed/gate10_output/run_trackaware_8threads_40min_ac225_fixed/ac225_hits.root`
- 正式 raw hit 分析：`/Users/chen/Documents/gate10_Ac225_Compton_PET/gate10-geoonly/gate10_output/run_trackaware_8threads_40min_ac225_fixed_analysis`
- 正式 detector response：`/Users/chen/Documents/gate10_Ac225_Compton_PET/gate10-geoonly/gate10_output/run_trackaware_8threads_40min_ac225_fixed_detector_response_gamma`
- OSEM 适配目录：`/Users/chen/Documents/OSEM-ac225-adapter`
- 正式 OSEM 输入：`/Users/chen/Documents/OSEM-ac225-adapter/data`
- Colab CUDA/OSEM 输出：`/Users/chen/Documents/OSEM-ac225-adapter/cuda_outputs`
- 论文 Fig.7/Fig.10 风格评价图：`/Users/chen/Documents/OSEM-ac225-adapter/evaluation`

## Gate10 正式模拟

正式模拟命令：

```bash
source /Users/chen/Documents/gate10/activate_gate10.sh
cd /Users/chen/Documents/gate10_Ac225_Compton_PET/gate10-geoonly
python -m gate10_model.run_ac225 \
  --output-dir gate10_output/run_trackaware_8threads_40min_ac225_fixed \
  --events 0 --activity-bq 45 --duration-s 2400 --threads 8 \
  --seed 2400703 --no-overlap-check
```

实际运行统计：

- simulated time：`0-40 min`
- threads：`8`
- events：`864105`
- wall time：`21.686 min`
- ROOT size：约 `768 MB`

物理设置检查：

- 源为中心点源 `ion 89 225`，`enable_decay=True`，Geant4 负责 Ac-225 衰变链和 gamma 发射。
- EM physics list 为 `G4EmStandardPhysics_option4`，覆盖 Compton、photoelectric、Rayleigh 等 gamma 电磁相互作用。
- 启用 atomic deexcitation、fluorescence、Auger、PIXE。
- Hits actor 已输出 `TrackID`、`ParentID`、`PDGCode`、`PreKineticEnergy`、`PostKineticEnergy`、`ProcessDefinedStep`，用于 gamma-track 级事件构建。

## Detector Response

正式 detector response 命令：

```bash
source /Users/chen/Documents/gate10/activate_gate10.sh
cd /Users/chen/Documents/gate10_Ac225_Compton_PET/gate10-geoonly
python -m gate10_model.detector_response \
  --root-file gate10_output/run_trackaware_8threads_40min_ac225_fixed/gate10_output/run_trackaware_8threads_40min_ac225_fixed/ac225_hits.root \
  --output-dir gate10_output/run_trackaware_8threads_40min_ac225_fixed_detector_response_gamma \
  --energy-mode ideal-gamma --grouping-mode gamma-track --pairing-mode gamma-track \
  --light-yield-per-mev 56000 --collection-efficiency 0.4 --pde 0.35 \
  --microcell-count 14400 --electronics-noise-pe 10 --energy-threshold-mev 0.005 \
  --random-seed 2400703
```

关键结果：

- total singles：`151101`
- thresholded singles：`135412`
- scatter singles：`123350`
- absorber singles：`27751`
- Compton candidate pairs：`583`
- valid-angle pairs：`472`
- mean total energy：`0.27543049 MeV`

这里最关键的修正是：不再用单纯 `EventID` 组合 scatter/absorber。Ac-225 一个 EventID 内包含衰变链产生的多条 gamma 和大量电子，EventID-only 会错误拼接不同 gamma 的 hit，导致 OSEM peak 偏到远处。当前使用 `gamma-track` 配对，并用 `PreKineticEnergy - PostKineticEnergy` 得到 gamma 的理想能量损失。

## 当前几何参数

这些参数来自当前 Gate10 几何，并写入 CUDA summary：

- sectors：`8`
- sector angle：`45 deg`
- scatter center radius：`39.5 mm`
- scatter thin thickness：`1.5 mm`
- scatter thick thickness：`4.0 mm`
- scatter thin inner radius：`38.75 mm`
- scatter thick inner radius：`37.5 mm`
- absorber inner radius：`62.5 mm`
- absorber center radius：`67.0 mm`
- absorber thickness：`9.0 mm`
- pixel grid：`8 x 8`
- pixel pitch：`3.2 mm`
- pixel size：`2.5 mm`
- crystal gap / reflector：`BaSO4`

原始 OSEM 中类似 `scarad=43.7`、`absrad=68.7` 的旧实验参数不再用于位置计算。本适配版 CUDA 直接读取 Gate10 输出的 scatter/absorber 三维位置。

## Gate10 到 OSEM 输入转换

正式转换命令：

```bash
cd /Users/chen/Documents/OSEM-ac225-adapter
python scripts/gate_to_osem_input.py \
  --input /Users/chen/Documents/gate10_Ac225_Compton_PET/gate10-geoonly/gate10_output/run_trackaware_8threads_40min_ac225_fixed_detector_response_gamma/compton_events.csv \
  --output-dir data --prefix ac225 \
  --energy-window-frac 0.15 --min-scatter-kev 1
```

输出：

- `data/ac225_osem_events_all.csv`：472 events，使用每个事件的 `scatter + absorber` 作为 incident energy。
- `data/ac225_osem_events_sourceqa20.csv`：253 events，已知源点 ARM <= 20 deg，只用于模拟 QA。
- `data/ac225_osem_events_line218.csv`：157 events，218 keV ±15% 能窗。
- `data/ac225_osem_events_line218_sourceqa20.csv`：109 events，218 keV 能窗加 source-QA。
- `data/ac225_osem_events_line440.csv`：156 events，440.446 keV ±15% 能窗。
- `data/ac225_osem_events_line440_sourceqa20.csv`：110 events，440.446 keV 能窗加 source-QA。
- `data/ac225_osem_conversion_summary.json`：转换统计和几何参数。

## Colab CUDA/OSEM

Colab 中运行：

```bash
cd /content/OSEM-ac225-adapter_formal_trackaware
bash run_colab_ac225_osem.sh
```

默认参数：

- grid：`40 x 40 x 25`
- FOV：`100 x 100 x 50 mm`
- iterations：`48`
- subsets：`4`
- sigma：`6 deg`
- sensitivity correction：关闭

`40 x 40` 的 XY 网格对应 `2.5 mm` voxel，和当前晶体 pixel size 一致。验证发现 `80 x 80` 在当前事件数下容易把单条 Compton cone 交线噪声强化成远端最高 voxel；`40 x 40` 更适合当前统计量。需要高分辨率时可在 Colab 前设置 `NX=80 NY=80`，但必须增加事件数或加入正则化/平滑评价。

## CUDA/OSEM 输出结果

正式 Colab 输出已回传到：

```text
/Users/chen/Documents/OSEM-ac225-adapter/cuda_outputs
/Users/chen/Documents/OSEM-ac225-adapter/evaluation
/Users/chen/Documents/OSEM-ac225-adapter/logs
```

关键 peak 结果：

| dataset | events | peak [mm] | centroid [mm] | comment |
|---|---:|---|---|---|
| all variable energy | 472 | `(-1.25, -36.25, -4)` | `(1.75, -1.84, -0.13)` | 盲重建最高 voxel 被伪交点主导 |
| 218 keV | 157 | `(13.75, -28.75, -2)` | `(-0.54, -5.58, -0.09)` | 盲重建仍受低质事件影响 |
| 440.446 keV | 156 | `(-1.25, 1.25, 0)` | `(3.23, -0.75, -1.26)` | peak 已接近中心 |
| sourceqa20 | 253 | `(1.25, -1.25, 0)` | `(-0.56, -0.58, 0.01)` | 几何闭合正确 |
| 218 keV sourceqa20 | 109 | `(1.25, -1.25, 0)` | `(2.15, -2.95, 1.36)` | 218 keV QA peak 正确 |
| 440.446 keV sourceqa20 | 110 | `(1.25, 1.25, 0)` | `(1.10, 4.12, 0.28)` | 440 keV QA peak 正确 |

结论：当前模型几何没有整体平移错误；如果用 source-QA 筛选，218 keV 和 440.446 keV 都能重建到原点附近。未经 QA 的 all/218 盲重建 peak 偏远，原因是 Ac-225 多能线、低能散射和低统计事件混合后，最高 voxel 容易被少数伪 cone 交点支配。

## 论文风格评价图

已生成：

- Fig.7 风格：`evaluation/fig7_spatial_resolution_vs_energy.png`
- Fig.10 风格：`evaluation/fig10_scatter_absorber_energy.png`
- 数值表：`evaluation/fig7_spatial_resolution_summary.csv`
- 能窗计数：`evaluation/fig10_energy_window_counts.csv`

Fig.7 当前数值：

- 218 keV source-QA：`FWHM mean = 3.865 mm`
- 440.446 keV source-QA：`FWHM mean = 2.932 mm`

这些 FWHM 是当前 OSEM 输出的 source-center profile 指标。它们可用于检查流程是否跑通和几何是否闭合，但还不能直接作为论文级 detector performance，因为当前 MPPC/能量响应是后处理近似。

Fig.10 当前能窗：

- 218 keV ±15%：约 `158 / 472` events
- 440.446 keV ±15%：约 `156 / 472` events

## 修改/新增文件

Gate10 模型侧：

- `gate10_model/simulation.py`：Hits actor 增加 track/parent/energy/process 分支。
- `gate10_model/detector_response.py`：增加 `ideal-gamma`、`gamma-track` grouping/pairing 和输出 `gamma_track_id`。
- `gate10_model/run_ac225.py`：修正 `--seed` 解析，数字 seed 转成 int。
- `tests/test_detector_response.py`：更新 detector response 测试。

OSEM 适配侧：

- `scripts/gate_to_osem_input.py`：Gate10 CSV 到直接位置 OSEM 输入；增加 line-specific source-QA 输出。
- `src/ac225_osem_cuda.cu`：直接位置 CUDA OSEM，几何参数改为当前 Gate10，能量按事件/能线可配置，默认关闭旧 sensitivity division。
- `scripts/plot_cuda_recon.py`：输出 central slice、MIP 和 profile PNG。
- `scripts/evaluate_ac225_results.py`：生成 Fig.7/Fig.10 风格评价，Fig.7 优先使用 line-specific source-QA。
- `scripts/osem_cpu_diagnostic.py`：CPU 对照诊断，用于定位 CUDA 和参数问题。
- `run_colab_ac225_osem.sh`：Colab 编译运行脚本，默认 `40 x 40 x 25`，运行 6 组输入。

## 可信度和限制

可信的部分：

- Gate10 确实模拟了 Ac-225 衰变链和 gamma 电磁相互作用。
- 当前 detector response 已避免 EventID-only 错误配对。
- 当前 CUDA/OSEM 与 CPU 诊断在相同网格/参数下结果一致。
- source-QA 和 line-specific source-QA 的 peak 已回到原点附近，说明几何和坐标适配正确。

仍需谨慎的部分：

- MPPC 光产额、PDE、microcell 饱和、电子学噪声是后处理近似，不是 optical photon transport。
- `sourceqa20` 使用已知模拟源点做 ARM 筛选，只能作为模拟 QA；真实未知源不能这样筛。
- all/218 盲重建最高 voxel 仍可能偏离中心，说明当前事件筛选和响应模型还不足以做稳定盲重建。
- 论文 Fig.7/Fig.10 的图形风格已对齐，但数值不能直接声称优于论文；当前 detector、source、能量线、统计量和响应模型都不同。
- 原始 OSEM 中的 `CS_crystal.txt` 衰减表当前未纳入，因为现有原始目录没有完整依赖文件；适配版保留 forward/backprojection/OSEM 更新思想，但不是逐行复刻原始实验重建环境。
