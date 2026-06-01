// Ac-225 Gate10 direct-position Compton OSEM adapter.
//
// This file intentionally does not overwrite the original OSEM-origin CUDA code.
// It keeps the same OSEM idea: subset forward projection, backprojection, and
// multiplicative image update using a Compton-cone response. The input and
// geometry are adapted to Gate10 simulation output, where event positions are
// already available in millimetres.

#include "cuda_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

static constexpr double kPi = 3.14159265358979323846;
static constexpr double kElectronRestKev = 510.99895;

// Gate10 geometry values used to replace the original scarad/absrad-style
// constants. Direct event positions are used in the kernels, but these numbers
// are reported with every run so the reconstruction parameters are auditable.
static constexpr int kSectors = 8;
static constexpr double kSectorAngleDeg = 45.0;
static constexpr double kScatterCenterRadiusMm = 39.5;
static constexpr double kScatterInnerRadiusThinMm = 38.75;
static constexpr double kScatterInnerRadiusThickMm = 37.5;
static constexpr double kScatterThinThicknessMm = 1.5;
static constexpr double kScatterThickThicknessMm = 4.0;
static constexpr double kAbsorberInnerRadiusMm = 62.5;
static constexpr double kAbsorberCenterRadiusMm = 67.0;
static constexpr double kAbsorberThicknessMm = 9.0;
static constexpr double kPixelPitchMm = 3.2;
static constexpr double kPixelSizeMm = 2.5;

struct Event {
    double sx;
    double sy;
    double sz;
    double ax;
    double ay;
    double az;
    double e_scatter_kev;
    double e_absorber_kev;
    double e_incident_kev;
    double theta_rad;
};

struct Options {
    std::string input;
    std::string outdir = "cuda_outputs/ac225";
    int nx = 80;
    int ny = 80;
    int nz = 25;
    int iterations = 8;
    int subsets = 8;
    int max_events = 0;
    double fov_x_mm = 100.0;
    double fov_y_mm = 100.0;
    double fov_z_mm = 50.0;
    double sigma_deg = 8.0;
    double source_x_mm = 0.0;
    double source_y_mm = 0.0;
    double source_z_mm = 0.0;
    bool use_sensitivity = false;
};

static void check_cuda(cudaError_t status, const char* label) {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << label << " failed: " << cudaGetErrorString(status);
        throw std::runtime_error(oss.str());
    }
}

static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string item;
    std::stringstream ss(line);
    while (std::getline(ss, item, ',')) {
        if (!item.empty() && item.back() == '\r') {
            item.pop_back();
        }
        out.push_back(item);
    }
    return out;
}

static double stod_field(
    const std::vector<std::string>& fields,
    const std::map<std::string, int>& header,
    const std::string& name
) {
    auto it = header.find(name);
    if (it == header.end()) {
        throw std::runtime_error("Missing CSV column: " + name);
    }
    return std::stod(fields.at(static_cast<size_t>(it->second)));
}

static double clamp_host(double value, double low, double high) {
    return std::max(low, std::min(high, value));
}

static double theta_from_scatter_energy(double e_scatter_kev, double e_incident_kev) {
    if (e_scatter_kev <= 0.0 || e_incident_kev <= e_scatter_kev) {
        return -1.0;
    }
    const double e_after = e_incident_kev - e_scatter_kev;
    const double c = 1.0 - kElectronRestKev * (1.0 / e_after - 1.0 / e_incident_kev);
    if (c < -1.0 || c > 1.0) {
        return -1.0;
    }
    return std::acos(clamp_host(c, -1.0, 1.0));
}

static std::vector<Event> read_events(const std::string& path, int max_events) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open input: " + path);
    }

    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("Input CSV is empty: " + path);
    }
    const auto header_fields = split_csv(line);
    std::map<std::string, int> header;
    for (int i = 0; i < static_cast<int>(header_fields.size()); ++i) {
        header[header_fields[static_cast<size_t>(i)]] = i;
    }

    std::vector<Event> events;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto fields = split_csv(line);
        if (fields.size() < header_fields.size()) {
            continue;
        }
        Event e{};
        e.sx = stod_field(fields, header, "scatter_x_mm");
        e.sy = stod_field(fields, header, "scatter_y_mm");
        e.sz = stod_field(fields, header, "scatter_z_mm");
        e.ax = stod_field(fields, header, "absorber_x_mm");
        e.ay = stod_field(fields, header, "absorber_y_mm");
        e.az = stod_field(fields, header, "absorber_z_mm");
        e.e_scatter_kev = stod_field(fields, header, "scatter_energy_keV");
        e.e_absorber_kev = stod_field(fields, header, "absorber_energy_keV");
        e.e_incident_kev = stod_field(fields, header, "incident_energy_keV");
        e.theta_rad = theta_from_scatter_energy(e.e_scatter_kev, e.e_incident_kev);
        if (e.theta_rad < 0.0 || !std::isfinite(e.theta_rad)) {
            continue;
        }
        events.push_back(e);
        if (max_events > 0 && static_cast<int>(events.size()) >= max_events) {
            break;
        }
    }
    return events;
}

__device__ double clamp_device(double value, double low, double high) {
    return fmax(low, fmin(high, value));
}

__device__ double klein_nishina_shape(double e_incident_kev, double e_after_kev, double cos_theta) {
    if (e_incident_kev <= 0.0 || e_after_kev <= 0.0) {
        return 0.0;
    }
    const double ratio = e_after_kev / e_incident_kev;
    const double term = ratio + 1.0 / ratio - (1.0 - cos_theta * cos_theta);
    return fmax(0.0, ratio * ratio * term);
}

__device__ double event_weight(
    const Event& e,
    double vx,
    double vy,
    double vz,
    double sigma_rad
) {
    const double ux = e.sx - vx;
    const double uy = e.sy - vy;
    const double uz = e.sz - vz;
    const double wx = e.ax - e.sx;
    const double wy = e.ay - e.sy;
    const double wz = e.az - e.sz;

    const double d1sq = ux * ux + uy * uy + uz * uz;
    const double d2sq = wx * wx + wy * wy + wz * wz;
    if (d1sq < 1.0e-9 || d2sq < 1.0e-9) {
        return 0.0;
    }

    const double costh = clamp_device(
        (ux * wx + uy * wy + uz * wz) / sqrt(d1sq * d2sq),
        -1.0,
        1.0
    );
    const double theta = acos(costh);
    const double delta = theta - e.theta_rad;
    const double angular = exp(-0.5 * delta * delta / (sigma_rad * sigma_rad));
    const double e_after = e.e_incident_kev - e.e_scatter_kev;
    const double kn = klein_nishina_shape(e.e_incident_kev, e_after, costh);

    return angular * kn / (d1sq * d2sq + 1.0e-12);
}

__global__ void forward_kernel(
    double* sums,
    const double* image,
    const Event* events,
    int nevents,
    int nx,
    int ny,
    int nz,
    double fov_x,
    double fov_y,
    double fov_z,
    double sigma_rad
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    const int nvox = nx * ny * nz;
    if (q >= nvox) {
        return;
    }
    const double img = image[q];
    if (img <= 0.0) {
        return;
    }
    const int k = q / (nx * ny);
    const int j = (q - k * nx * ny) / nx;
    const int i = q - k * nx * ny - j * nx;

    const double vx = -fov_x / 2.0 + fov_x * (static_cast<double>(i) + 0.5) / nx;
    const double vy = -fov_y / 2.0 + fov_y * (static_cast<double>(j) + 0.5) / ny;
    const double vz = -fov_z / 2.0 + fov_z * (static_cast<double>(k) + 0.5) / nz;

    for (int p = 0; p < nevents; ++p) {
        const double w = event_weight(events[p], vx, vy, vz, sigma_rad);
        if (w > 0.0) {
            atomicAdd(&sums[p], img * w);
        }
    }
}

__global__ void backward_update_kernel(
    double* image,
    const double* sums,
    const Event* events,
    int nevents,
    int nx,
    int ny,
    int nz,
    double fov_x,
    double fov_y,
    double fov_z,
    double sigma_rad,
    bool use_sensitivity
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    const int nvox = nx * ny * nz;
    if (q >= nvox || image[q] <= 0.0) {
        return;
    }
    const int k = q / (nx * ny);
    const int j = (q - k * nx * ny) / nx;
    const int i = q - k * nx * ny - j * nx;

    const double vx = -fov_x / 2.0 + fov_x * (static_cast<double>(i) + 0.5) / nx;
    const double vy = -fov_y / 2.0 + fov_y * (static_cast<double>(j) + 0.5) / ny;
    const double vz = -fov_z / 2.0 + fov_z * (static_cast<double>(k) + 0.5) / nz;

    double back = 0.0;
    double sens = 0.0;
    for (int p = 0; p < nevents; ++p) {
        const double w = event_weight(events[p], vx, vy, vz, sigma_rad);
        sens += w;
        if (sums[p] > 1.0e-300) {
            back += w / sums[p];
        }
    }
    if (use_sensitivity && sens > 1.0e-300) {
        image[q] = image[q] * back / sens;
    } else if (!use_sensitivity && back > 1.0e-300) {
        image[q] = image[q] * back;
    } else {
        image[q] = 0.0;
    }
}

static void make_dir(const std::string& path) {
    std::string command = "mkdir -p '" + path + "'";
    const int rc = std::system(command.c_str());
    if (rc != 0) {
        throw std::runtime_error("Cannot create directory: " + path);
    }
}

static void normalize_image(std::vector<double>& image) {
    const double maxv = *std::max_element(image.begin(), image.end());
    if (maxv <= 0.0 || !std::isfinite(maxv)) {
        return;
    }
    for (double& v : image) {
        v /= maxv;
    }
}

static double coord(int index, int n, double fov) {
    return -fov / 2.0 + fov * (static_cast<double>(index) + 0.5) / n;
}

static void write_matrix_csv(const std::string& path, const std::vector<std::vector<double>>& matrix) {
    std::ofstream out(path);
    out << std::setprecision(10);
    for (const auto& row : matrix) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i) {
                out << ",";
            }
            out << row[i];
        }
        out << "\n";
    }
}

static void write_image_text(const std::string& path, const std::vector<double>& image, const Options& opt) {
    std::ofstream out(path);
    out << std::setprecision(10);
    for (int k = 0; k < opt.nz; ++k) {
        out << "# z_index " << k << "\n";
        for (int j = 0; j < opt.ny; ++j) {
            for (int i = 0; i < opt.nx; ++i) {
                out << image[i + opt.nx * (j + opt.ny * k)];
                if (i + 1 < opt.nx) {
                    out << "\t";
                }
            }
            out << "\n";
        }
        out << "\n";
    }
}

static double fwhm_from_profile(const std::vector<double>& coords, const std::vector<double>& values) {
    if (values.empty()) {
        return -1.0;
    }
    const int n = static_cast<int>(values.size());
    int peak = 0;
    for (int i = 1; i < n; ++i) {
        if (values[i] > values[peak]) {
            peak = i;
        }
    }
    if (values[peak] <= 0.0) {
        return -1.0;
    }
    const double half = 0.5 * values[peak];
    int left = peak;
    while (left > 0 && values[left] >= half) {
        --left;
    }
    int right = peak;
    while (right < n - 1 && values[right] >= half) {
        ++right;
    }
    if (left == peak || right == peak || left + 1 >= n || right - 1 < 0) {
        return -1.0;
    }
    auto interp = [&](int a, int b) {
        const double y0 = values[a];
        const double y1 = values[b];
        if (y1 == y0) {
            return coords[a];
        }
        return coords[a] + (half - y0) * (coords[b] - coords[a]) / (y1 - y0);
    };
    const double xl = interp(left, left + 1);
    const double xr = interp(right, right - 1);
    return xr - xl;
}

static void write_outputs(const std::vector<double>& image, const Options& opt, int event_count) {
    const int central_k = opt.nz / 2;
    std::vector<std::vector<double>> central(static_cast<size_t>(opt.ny), std::vector<double>(static_cast<size_t>(opt.nx), 0.0));
    std::vector<std::vector<double>> mip(static_cast<size_t>(opt.ny), std::vector<double>(static_cast<size_t>(opt.nx), 0.0));
    for (int j = 0; j < opt.ny; ++j) {
        for (int i = 0; i < opt.nx; ++i) {
            central[static_cast<size_t>(j)][static_cast<size_t>(i)] =
                image[i + opt.nx * (j + opt.ny * central_k)];
            double maxv = 0.0;
            for (int k = 0; k < opt.nz; ++k) {
                maxv = std::max(maxv, image[i + opt.nx * (j + opt.ny * k)]);
            }
            mip[static_cast<size_t>(j)][static_cast<size_t>(i)] = maxv;
        }
    }
    write_matrix_csv(opt.outdir + "/central_slice.csv", central);
    write_matrix_csv(opt.outdir + "/mip_xy.csv", mip);
    write_image_text(opt.outdir + "/image_final.txt", image, opt);

    double sum = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double cz = 0.0;
    double inside5 = 0.0;
    int peak_index = 0;
    for (int q = 0; q < opt.nx * opt.ny * opt.nz; ++q) {
        if (image[q] > image[peak_index]) {
            peak_index = q;
        }
        const int k = q / (opt.nx * opt.ny);
        const int j = (q - k * opt.nx * opt.ny) / opt.nx;
        const int i = q - k * opt.nx * opt.ny - j * opt.nx;
        const double x = coord(i, opt.nx, opt.fov_x_mm);
        const double y = coord(j, opt.ny, opt.fov_y_mm);
        const double z = coord(k, opt.nz, opt.fov_z_mm);
        const double v = image[q];
        sum += v;
        cx += v * x;
        cy += v * y;
        cz += v * z;
        const double dx = x - opt.source_x_mm;
        const double dy = y - opt.source_y_mm;
        const double dz = z - opt.source_z_mm;
        if (std::sqrt(dx * dx + dy * dy + dz * dz) <= 5.0) {
            inside5 += v;
        }
    }
    if (sum > 0.0) {
        cx /= sum;
        cy /= sum;
        cz /= sum;
    }

    const int peak_k = peak_index / (opt.nx * opt.ny);
    const int peak_j = (peak_index - peak_k * opt.nx * opt.ny) / opt.nx;
    const int peak_i = peak_index - peak_k * opt.nx * opt.ny - peak_j * opt.nx;

    std::vector<double> xcoords(static_cast<size_t>(opt.nx));
    std::vector<double> ycoords(static_cast<size_t>(opt.ny));
    for (int i = 0; i < opt.nx; ++i) {
        xcoords[static_cast<size_t>(i)] = coord(i, opt.nx, opt.fov_x_mm);
    }
    for (int j = 0; j < opt.ny; ++j) {
        ycoords[static_cast<size_t>(j)] = coord(j, opt.ny, opt.fov_y_mm);
    }
    const int source_i = static_cast<int>(std::min_element(
        xcoords.begin(), xcoords.end(),
        [&](double a, double b) { return std::abs(a - opt.source_x_mm) < std::abs(b - opt.source_x_mm); }
    ) - xcoords.begin());
    const int source_j = static_cast<int>(std::min_element(
        ycoords.begin(), ycoords.end(),
        [&](double a, double b) { return std::abs(a - opt.source_y_mm) < std::abs(b - opt.source_y_mm); }
    ) - ycoords.begin());
    const int source_k = central_k;

    std::vector<double> profile_x_source(static_cast<size_t>(opt.nx));
    std::vector<double> profile_y_source(static_cast<size_t>(opt.ny));
    std::vector<double> profile_x_peak(static_cast<size_t>(opt.nx));
    std::vector<double> profile_y_peak(static_cast<size_t>(opt.ny));
    for (int i = 0; i < opt.nx; ++i) {
        profile_x_source[static_cast<size_t>(i)] = image[i + opt.nx * (source_j + opt.ny * source_k)];
        profile_x_peak[static_cast<size_t>(i)] = image[i + opt.nx * (peak_j + opt.ny * peak_k)];
    }
    for (int j = 0; j < opt.ny; ++j) {
        profile_y_source[static_cast<size_t>(j)] = image[source_i + opt.nx * (j + opt.ny * source_k)];
        profile_y_peak[static_cast<size_t>(j)] = image[peak_i + opt.nx * (j + opt.ny * peak_k)];
    }

    std::ofstream summary(opt.outdir + "/summary.txt");
    summary << std::setprecision(10);
    summary << "input_file: " << opt.input << "\n";
    summary << "valid_events_used: " << event_count << "\n";
    summary << "nx: " << opt.nx << "\n";
    summary << "ny: " << opt.ny << "\n";
    summary << "nz: " << opt.nz << "\n";
    summary << "fov_x_mm: " << opt.fov_x_mm << "\n";
    summary << "fov_y_mm: " << opt.fov_y_mm << "\n";
    summary << "fov_z_mm: " << opt.fov_z_mm << "\n";
    summary << "iterations: " << opt.iterations << "\n";
    summary << "subsets: " << opt.subsets << "\n";
    summary << "sigma_deg: " << opt.sigma_deg << "\n";
    summary << "use_sensitivity: " << (opt.use_sensitivity ? "true" : "false") << "\n";
    summary << "gate_geometry_sectors: " << kSectors << "\n";
    summary << "gate_sector_angle_deg: " << kSectorAngleDeg << "\n";
    summary << "gate_scatter_center_radius_mm: " << kScatterCenterRadiusMm << "\n";
    summary << "gate_scatter_inner_radius_thin_mm: " << kScatterInnerRadiusThinMm << "\n";
    summary << "gate_scatter_inner_radius_thick_mm: " << kScatterInnerRadiusThickMm << "\n";
    summary << "gate_scatter_thin_thickness_mm: " << kScatterThinThicknessMm << "\n";
    summary << "gate_scatter_thick_thickness_mm: " << kScatterThickThicknessMm << "\n";
    summary << "gate_absorber_inner_radius_mm: " << kAbsorberInnerRadiusMm << "\n";
    summary << "gate_absorber_center_radius_mm: " << kAbsorberCenterRadiusMm << "\n";
    summary << "gate_absorber_thickness_mm: " << kAbsorberThicknessMm << "\n";
    summary << "gate_pixel_pitch_mm: " << kPixelPitchMm << "\n";
    summary << "gate_pixel_size_mm: " << kPixelSizeMm << "\n";
    summary << "peak_x_mm: " << coord(peak_i, opt.nx, opt.fov_x_mm) << "\n";
    summary << "peak_y_mm: " << coord(peak_j, opt.ny, opt.fov_y_mm) << "\n";
    summary << "peak_z_mm: " << coord(peak_k, opt.nz, opt.fov_z_mm) << "\n";
    summary << "centroid_x_mm: " << cx << "\n";
    summary << "centroid_y_mm: " << cy << "\n";
    summary << "centroid_z_mm: " << cz << "\n";
    summary << "activity_fraction_inside_5mm: " << (sum > 0.0 ? inside5 / sum : 0.0) << "\n";
    summary << "source_center_profile_fwhm_x_mm: " << fwhm_from_profile(xcoords, profile_x_source) << "\n";
    summary << "source_center_profile_fwhm_y_mm: " << fwhm_from_profile(ycoords, profile_y_source) << "\n";
    summary << "peak_profile_fwhm_x_mm: " << fwhm_from_profile(xcoords, profile_x_peak) << "\n";
    summary << "peak_profile_fwhm_y_mm: " << fwhm_from_profile(ycoords, profile_y_peak) << "\n";
}

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };
        if (arg == "--input") {
            opt.input = require_value(arg);
        } else if (arg == "--outdir") {
            opt.outdir = require_value(arg);
        } else if (arg == "--nx") {
            opt.nx = std::stoi(require_value(arg));
        } else if (arg == "--ny") {
            opt.ny = std::stoi(require_value(arg));
        } else if (arg == "--nz") {
            opt.nz = std::stoi(require_value(arg));
        } else if (arg == "--iterations") {
            opt.iterations = std::stoi(require_value(arg));
        } else if (arg == "--subsets") {
            opt.subsets = std::stoi(require_value(arg));
        } else if (arg == "--max-events") {
            opt.max_events = std::stoi(require_value(arg));
        } else if (arg == "--fov-x-mm") {
            opt.fov_x_mm = std::stod(require_value(arg));
        } else if (arg == "--fov-y-mm") {
            opt.fov_y_mm = std::stod(require_value(arg));
        } else if (arg == "--fov-z-mm") {
            opt.fov_z_mm = std::stod(require_value(arg));
        } else if (arg == "--sigma-deg") {
            opt.sigma_deg = std::stod(require_value(arg));
        } else if (arg == "--source-x-mm") {
            opt.source_x_mm = std::stod(require_value(arg));
        } else if (arg == "--source-y-mm") {
            opt.source_y_mm = std::stod(require_value(arg));
        } else if (arg == "--source-z-mm") {
            opt.source_z_mm = std::stod(require_value(arg));
        } else if (arg == "--use-sensitivity") {
            opt.use_sensitivity = true;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (opt.input.empty()) {
        throw std::runtime_error("--input is required");
    }
    if (opt.nx <= 0 || opt.ny <= 0 || opt.nz <= 0 || opt.iterations <= 0 || opt.subsets <= 0) {
        throw std::runtime_error("Grid, iteration, and subset values must be positive");
    }
    return opt;
}

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);
        make_dir(opt.outdir);
        std::vector<Event> events = read_events(opt.input, opt.max_events);
        if (events.empty()) {
            throw std::runtime_error("No valid events after input filtering");
        }

        int device = 0;
        check_cuda(cudaSetDevice(device), "cudaSetDevice");
        cudaDeviceProp props{};
        check_cuda(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties");
        std::cout << "CUDA device: " << props.name << "\n";
        std::cout << "Events: " << events.size() << "\n";

        const int nvox = opt.nx * opt.ny * opt.nz;
        std::vector<double> image(static_cast<size_t>(nvox), 1.0);
        for (int k = 0; k < opt.nz; ++k) {
            for (int j = 0; j < opt.ny; ++j) {
                for (int i = 0; i < opt.nx; ++i) {
                    const double x = coord(i, opt.nx, opt.fov_x_mm);
                    const double y = coord(j, opt.ny, opt.fov_y_mm);
                    const double r = std::sqrt(x * x + y * y);
                    if (r > std::min(opt.fov_x_mm, opt.fov_y_mm) / 2.0) {
                        image[i + opt.nx * (j + opt.ny * k)] = 0.0;
                    }
                }
            }
        }

        double* d_image = nullptr;
        check_cuda(cudaMalloc(&d_image, sizeof(double) * static_cast<size_t>(nvox)), "cudaMalloc image");
        check_cuda(cudaMemcpy(d_image, image.data(), sizeof(double) * static_cast<size_t>(nvox), cudaMemcpyHostToDevice), "cudaMemcpy image");

        const int threads = 256;
        const int blocks = (nvox + threads - 1) / threads;
        const double sigma_rad = opt.sigma_deg * kPi / 180.0;

        for (int iter = 0; iter < opt.iterations; ++iter) {
            for (int subset = 0; subset < opt.subsets; ++subset) {
                std::vector<Event> subset_events;
                for (size_t idx = static_cast<size_t>(subset); idx < events.size(); idx += static_cast<size_t>(opt.subsets)) {
                    subset_events.push_back(events[idx]);
                }
                if (subset_events.empty()) {
                    continue;
                }

                Event* d_events = nullptr;
                double* d_sums = nullptr;
                check_cuda(
                    cudaMalloc(&d_events, sizeof(Event) * subset_events.size()),
                    "cudaMalloc events"
                );
                check_cuda(
                    cudaMalloc(&d_sums, sizeof(double) * subset_events.size()),
                    "cudaMalloc sums"
                );
                check_cuda(
                    cudaMemcpy(d_events, subset_events.data(), sizeof(Event) * subset_events.size(), cudaMemcpyHostToDevice),
                    "cudaMemcpy events"
                );
                check_cuda(cudaMemset(d_sums, 0, sizeof(double) * subset_events.size()), "cudaMemset sums");

                forward_kernel<<<blocks, threads>>>(
                    d_sums,
                    d_image,
                    d_events,
                    static_cast<int>(subset_events.size()),
                    opt.nx,
                    opt.ny,
                    opt.nz,
                    opt.fov_x_mm,
                    opt.fov_y_mm,
                    opt.fov_z_mm,
                    sigma_rad
                );
                check_cuda(cudaGetLastError(), "forward_kernel");
                check_cuda(cudaDeviceSynchronize(), "forward sync");

                backward_update_kernel<<<blocks, threads>>>(
                    d_image,
                    d_sums,
                    d_events,
                    static_cast<int>(subset_events.size()),
                    opt.nx,
                    opt.ny,
                    opt.nz,
                    opt.fov_x_mm,
                    opt.fov_y_mm,
                    opt.fov_z_mm,
                    sigma_rad,
                    opt.use_sensitivity
                );
                check_cuda(cudaGetLastError(), "backward_update_kernel");
                check_cuda(cudaDeviceSynchronize(), "backward sync");

                cudaFree(d_sums);
                cudaFree(d_events);
            }
            check_cuda(cudaMemcpy(image.data(), d_image, sizeof(double) * static_cast<size_t>(nvox), cudaMemcpyDeviceToHost), "copy image to host");
            normalize_image(image);
            check_cuda(cudaMemcpy(d_image, image.data(), sizeof(double) * static_cast<size_t>(nvox), cudaMemcpyHostToDevice), "copy image to device");
            std::cout << "iteration " << (iter + 1) << " complete\n";
        }

        check_cuda(cudaMemcpy(image.data(), d_image, sizeof(double) * static_cast<size_t>(nvox), cudaMemcpyDeviceToHost), "final copy");
        normalize_image(image);
        cudaFree(d_image);
        write_outputs(image, opt, static_cast<int>(events.size()));
        std::cout << "Output: " << opt.outdir << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
