# 探测器响应后处理说明

这个目录里的 `detector_response.py` 是对 GATE hits 的后处理，不是完整 optical photon tracking。

它把：

```text
Geant4 step hits
→ crystal singles
→ 简化 MPPC 响应
→ scatter/absorber Compton candidates
```

串起来，目的是让输出比原始 `ac225_hits.root` 更接近探测器读出。

## 1. 运行命令

先运行 GATE 模拟，得到 hits：

```bash
cd /Users/chen/Documents/gate10_Ac225_Compton_PET/gate10-geoonly
source /Users/chen/Documents/gate10/activate_gate10.sh
python -m gate10_model.run_ac225 --events 1000 --threads 1 --output-dir gate10_output/small_run --no-overlap-check
```

再运行探测器响应后处理：

```bash
python -m gate10_model.detector_response \
  --root-file gate10_output/small_run/ac225_hits.root \
  --output-dir gate10_output/small_run_detector_response
```

如果要修改光产额、PDE、阈值等参数：

```bash
python -m gate10_model.detector_response \
  --root-file gate10_output/small_run/ac225_hits.root \
  --output-dir gate10_output/small_run_detector_response \
  --light-yield-per-mev 50000 \
  --collection-efficiency 0.4 \
  --pde 0.35 \
  --microcell-count 14400 \
  --electronics-noise-pe 10 \
  --energy-threshold-mev 0.02
```

## 2. 默认探测器响应模型

默认参数：

```text
light_yield_per_mev: 50000 photons/MeV
collection_efficiency: 0.40
pde: 0.35
microcell_count: 14400
gain: 1.0
electronics_noise_pe: 10 pe
energy_threshold_mev: 0.02 MeV
```

计算链：

```text
N_optical_mean = Edep_MeV * light_yield_per_mev
N_collected_mean = N_optical_mean * collection_efficiency
N_pe_mean = N_collected_mean * pde
N_pe_sample = Poisson(N_pe_mean)
N_fired = microcell_count * (1 - exp(-N_pe_sample / microcell_count))
N_fired += Gaussian(0, electronics_noise_pe)
E_reco = N_fired / (light_yield_per_mev * collection_efficiency * pde)
```

注意：

- `light_yield_per_mev` 是闪烁晶体参数，不是 MPPC 参数。
- `pde`、`microcell_count`、噪声、阈值是 MPPC/电子学响应参数。
- 这是快速近似模型，没有追踪真实光学光子传播。

## 3. 输出文件

后处理输出目录：

```text
gate10_output/small_run_detector_response
```

主要文件：

```text
crystal_singles.csv
compton_events.csv
detector_response.root
detector_response_summary.txt
reconstructed_energy_spectrum.png
photoelectron_spectrum.png
scatter_absorber_energy.png
compton_angle_spectrum.png
compton_total_energy_spectrum.png
layer_summary.png
```

`detector_response.root` 里有两个 tree：

```text
CrystalSingles
ComptonEvents
```

## 4. CrystalSingles 如何理解

`CrystalSingles` 是把同一个事件、同一个晶体内的所有 step hits 合并后的结果。

合并键：

```text
EventID + PreStepUniqueVolumeID
```

主要字段：

```text
event_id
crystal_id
layer
sector
pixel_index
energy_deposit_mev
energy_reco_mev
x, y, z
time
hit_count
optical_photons_mean
collected_photons_mean
photoelectrons_mean
photoelectrons
charge
passed_threshold
```

重点看：

- `energy_deposit_mev`：Geant4 在这个晶体里真实沉积的能量。
- `energy_reco_mev`：经过简化 MPPC 响应后的重建能量。
- `photoelectrons`：考虑 PDE、Poisson 涨落、饱和和电子学噪声后的信号。
- `passed_threshold`：是否超过当前电子学阈值。

评价方法：

1. `energy_reco_mev` 应与 `energy_deposit_mev` 总体相关。
2. `photoelectrons` 不应大量为 0，除非阈值或 PDE 设置过低。
3. scatter 和 absorber 的 singles 数量应随几何和能量窗变化合理。
4. 如果 `energy_reco_mev` 明显低于 `energy_deposit_mev`，通常是 microcell 饱和、PDE/collection 参数或噪声模型导致。

## 5. ComptonEvents 如何理解

`ComptonEvents` 是同一个 `event_id` 下的 scatter 和 absorber singles 配对。

主要字段：

```text
event_id
scatter_crystal_id
absorber_crystal_id
energy_scatter_mev
energy_absorber_mev
energy_total_mev
scatter_x, scatter_y, scatter_z
absorber_x, absorber_y, absorber_z
delta_time
distance_mm
compton_cos_theta
compton_angle_deg
valid_compton_angle
```

康普顿角计算：

```text
E_before = E_scatter + E_absorber
E_after = E_absorber
cos(theta) = 1 - 0.511 * (1/E_after - 1/E_before)
```

能量单位是 MeV。

评价方法：

1. `valid_compton_angle` 为 True 才表示能量组合给出了物理允许的康普顿角。
2. 如果 valid 数量很少，可能是统计太少、能量阈值不合适、scatter/absorber 配对不够干净，或当前 Ac-225 gamma 能量不适合当前简化选择。
3. `energy_total_mev` 应该结合 Ac-225 相关 gamma 能线和能量窗口看，不要直接把所有候选混在一起判断。
4. `delta_time` 目前使用 ROOT 里的 `GlobalTime` 原始单位。做真实符合窗前，需要先确认时间单位和衰变链延迟处理。

## 6. 如何评价一组结果

建议按这个顺序看：

1. `detector_response_summary.txt`
   - 看 singles 总数、过阈值 singles、scatter/absorber 数量、Compton candidates 数量。

2. `layer_summary.png`
   - 看 scatter 和 absorber 的计数及重建能量是否极端失衡。

3. `reconstructed_energy_spectrum.png`
   - 看 MPPC 响应后的晶体能谱。
   - 调整 `pde`、`collection_efficiency`、`microcell_count`、噪声后，这张图会变化。

4. `photoelectron_spectrum.png`
   - 看 MPPC 信号规模。
   - 如果大量信号贴近 0，阈值/PDE/collection 可能太低。
   - 如果大量信号接近 `microcell_count`，说明饱和严重。

5. `scatter_absorber_energy.png`
   - 看 scatter 能量和 absorber 能量是否形成合理分布。
   - Compton PET 后续需要在这张图上定义更合适的选择条件。

6. `compton_angle_spectrum.png`
   - 只包含物理允许的康普顿角。
   - 小统计下为空或很少不一定是代码错误，可能只是候选数不足。

7. `compton_total_energy_spectrum.png`
   - 看候选的总能量是否落在你关心的 gamma 能线附近。

## 7. 当前模型的限制

这个脚本没有做：

- 真实 optical photon tracking；
- 晶体表面反射/包裹材料；
- 光导/耦合胶；
- MPPC 几何面积耦合；
- dark count、afterpulse、crosstalk 的详细模型；
- 真实电子学波形；
- 严格的时间符合；
- 成像重建。

它适合先做快速、可调参的探测器响应近似。等你有具体 MPPC 型号、PDE 曲线、晶体光产额、能量分辨率和时间分辨率数据后，再把这些参数替换成实测或厂家数据。
