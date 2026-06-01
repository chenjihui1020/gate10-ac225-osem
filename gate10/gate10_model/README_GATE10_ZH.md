# GATE 10 Ac-225 Compton PET 重写版

本目录用 GATE 10 Python API 重写了原项目中的几何和 Ac-225 源项。旧的 GATE 宏和 Geant4 C++ 代码没有被覆盖。

## 1. 打开环境

先进入项目目录，并激活之前安装好的 GATE 10 环境：

```bash
cd /Users/chen/Documents/gate10_Ac225_Compton_PET/gate10-geoonly
source /Users/chen/Documents/gate10/activate_gate10.sh
```

之后所有命令都从 `/Users/chen/Documents/gate10_Ac225_Compton_PET/gate10-geoonly` 运行。

## 2. 运行一个小模拟

```bash
python -m gate10_model.run_ac225 --events 1000 --threads 1
```

默认输出在：

```text
/Users/chen/Documents/gate10_Ac225_Compton_PET/gate10-geoonly/gate10_output/smoke
```

其中：

- `stats.txt`：运行统计。
- `ac225_hits.root`：GATE 10 `DigitizerHitsCollectionActor` 输出的 hit ROOT 文件。

如果要按活度和时间运行，而不是固定事件数：

```bash
python -m gate10_model.run_ac225 --events 0 --activity-bq 1000000 --duration-s 0.01 --threads 1
```

注意：`1 MBq x 2 h` 对应约 `7.2e9` 次衰变，不能直接作为日常测试运行。正式大统计应分块运行。

## 3. 可视化

推荐先用稳定的非 Qt 预览图：

```bash
python -m gate10_model.plot_geometry --open
```

该命令会生成：

```text
gate10_output/geometry_plots/geometry_xy.png
gate10_output/geometry_plots/geometry_3d.png
gate10_output/geometry_plots/geometry_pixel_faces_8x8.png
```

其中 `geometry_pixel_faces_8x8.png` 是检查像素排布最直接的图：每个模块正面都是 `8 x 8` 个 `2.5 mm x 2.5 mm` GAGG 像素，pitch 为 `3.2 mm`。

如果一定要尝试 Geant4 Qt 可视化窗口：

```bash
python -m gate10_model.visualize_geometry --type qt
```

这个命令会占住当前终端，直到你关闭 Qt 窗口。不要加 `--subprocess`，macOS 上 GATE 10 的 Qt 子进程模式可能出现 `The queue is empty`。在部分 macOS + Python 3.14 + GATE 10 wheel 组合上，Qt 本身可能触发 Python 原生崩溃，此时请使用上面的 `plot_geometry`、GDML 或 VRML 方式。

只导出 GDML 几何文件，不打开窗口：

```bash
python -m gate10_model.visualize_geometry --type gdml
```

如果终端提示 `No module named 'pyg4ometry'`，这只表示当前环境不能直接用 Python 预览 GDML；GDML 文件本身仍会生成。

只导出 VRML 文件：

```bash
python -m gate10_model.visualize_geometry --type vrml_file_only
```

GDML/VRML 文件会写入：

```text
/Users/chen/Documents/gate10_Ac225_Compton_PET/gate10-geoonly/gate10_output/visualization
```

## 4. 几何对应关系

重写代码位于 `geometry.py`：

- world：`3000 x 3000 x 3000 mm` 空气盒。
- source phantom：中心半径 `2 mm` 水球。
- scatter：8 个扇区，偶数扇区 `1.5 mm` GAGG，内表面半径 `38.75 mm`；奇数扇区 `4.0 mm` GAGG，内表面半径 `37.5 mm`。两种厚度的晶体中心半径均为 `39.5 mm`。
- absorber：8 个扇区，`9.0 mm` GAGG，内表面半径 `62.5 mm`。
- 每个模块：`8 x 8` 个 `2.5 x 2.5 mm` 像素，pitch `3.2 mm`；晶体之间 `0.7 mm` 间隙按 `BaSO4` 硫酸钡处理；外侧 `2 mm` MPPC。
- 材料数据库位于本目录 `config/GateMaterials.db`，包含 `CeGAGG`、`BaSO4`、`PLA_3DPrintedFrame`、`Water`、`Air`、`Silicon`。

## 5. 源项对应关系

重写代码位于 `source.py`：

```python
source.particle = "ion 89 225"
source.energy.mono = 0
source.position.type = "point"
source.position.translation = [0, 0, 0]
source.direction.type = "iso"
```

这对应旧宏：

```text
/gate/source/ac225/gps/particle ion
/gate/source/ac225/gps/ion 89 225 0 0
/gate/source/ac225/gps/pos/type Point
/gate/source/ac225/gps/pos/centre 0 0 0 mm
/gate/source/ac225/gps/ang/type iso
```

物理设置启用了 radioactive decay：

```python
sim.physics_manager.enable_decay = True
sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
```

因此正式 Ac-225 模拟由 Geant4 radioactive decay 产生衰变链和 gamma/electron，不在源中手写 218 keV、440.446 keV 或 511 keV 单能 gamma。

## 6. 代码入口

- `geometry.py`：几何。
- `source.py`：Ac-225 ion source。
- `simulation.py`：把几何、源、物理、actor 组装成一个 `gate.Simulation()`。
- `run_ac225.py`：命令行运行模拟。
- `visualize_geometry.py`：Qt/GDML/VRML 可视化入口。
