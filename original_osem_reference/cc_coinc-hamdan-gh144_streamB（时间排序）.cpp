// cc_coinc-hamdan-gh144_stream_logicB.cpp
// 基于 stream 版的基础，将几何符合逻辑修改为 "Logic B" (Intra-module)
// 即：同一 Module 内，前 64 通道为 Scatter，后 64 通道为 Absorber
//时间排序
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <string>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <array>

using namespace std;

struct Event {
    double t_in;   // ns
    int    module;
    int    ch;
    double E;      // keV
};

// hamdan 原来的 chconv0 映射（保持不变）
int chconv0[144] = {
    33, 41, 49, 57, 4, 12, 20, 28, 129, 130, 131, 132, 38, 46, 54, 62,
    1, 9, 17, 25, 6, 14, 22, 30, 39, 47, 55, 63, 5, 13, 21, 29,
    34, 42, 50, 58, 36, 44, 52, 60, 2, 10, 18, 26, 7, 15, 23, 31,
    40, 48, 56, 64, 35, 43, 51, 59, 3, 11, 19, 27, 133, 134, 135, 136,
    8, 16, 24, 32, 37, 45, 53, 61, 97, 105, 113, 121, 68, 76, 84, 92,
    137, 138, 139, 140, 102, 110, 118, 126, 65, 73, 81, 89, 70, 78, 86, 94,
    103, 111, 119, 127, 69, 77, 85, 93, 98, 106, 114, 122, 100, 108, 116, 124,
    66, 74, 82, 90, 71, 79, 87, 95, 104, 112, 120, 128, 99, 107, 115, 123,
    67, 75, 83, 91, 141, 142, 143, 144, 72, 80, 88, 96, 101, 109, 117, 125
};

// [修改点 1] 移除了 Logic A 需要的 ModulePair 和 validPairs，Logic B 不需要它们

// ========= gh144 能量反解（保持不变） =========
static inline double gh144_energy(double tot, int ch, const vector<vector<double>> &calib)
{
    if (ch < 0 || ch >= 144) return -1.0;
    double c1 = calib[ch][0];
    double c2 = calib[ch][1];
    double c3 = calib[ch][2];
    if (c1 == 0.0 || c2 == 0.0) return -1.0;

    double arg = c2 * tot + c3;
    if (arg <= 0.0) return -1.0;

    double E = c1 * std::log(arg);
    if (!std::isfinite(E) || E <= 0.0) return -1.0;
    return E;
}
// =====================================================

class DetectorStream {
public:
    DetectorStream(int p, const string* inames65)
        : p_(p), inames_(inames65), file_idx_(0),
          t_s_(0), t_ms_(0), time_(0), accepted_(0) {}

    bool init() {
        // 读 calibinfo(p+1).txt
        calib_.assign(144, vector<double>(3, 0.0));
        string fname = "Calib_Compton/calibinfo" + to_string(p_ + 1) + ".txt";
        ifstream fin(fname);
        if (!fin.is_open()) {
            cerr << "[Detector " << (p_ + 1) << "] Cannot open " << fname << endl;
            return false;
        }
        for (int ch = 0; ch < 144; ++ch) {
            fin >> calib_[ch][0] >> calib_[ch][1] >> calib_[ch][2];
        }
        fin.close();

        return open_next_file();
    }

    // 取下一个有效 Event
    bool next(Event &ev) {
        const double multi = 8.0;

        while (true) {
            if (!ifs_.is_open()) {
                if (!open_next_file()) return false;
            }

            unsigned char tem[1];
            unsigned char tem1[7];

            if (!ifs_.read(reinterpret_cast<char*>(tem), sizeof(tem))) {
                ifs_.close();
                continue;
            }

            int header = tem[0];
            if (header != 0x69 && header != 0x6a) {
                continue;
            }

            if (!ifs_.read(reinterpret_cast<char*>(tem1), sizeof(tem1))) {
                ifs_.close();
                continue;
            }

            unsigned char out[8];
            out[0] = tem[0];
            for (int i = 0; i < 7; i++) out[i + 1] = tem1[i];

            if (header == 0x69) {
                t_s_  = (uint32_t(out[1]) << 24) | (uint32_t(out[2]) << 16) | (uint32_t(out[3]) << 8) | uint32_t(out[4]);
                t_ms_ = (uint32_t(out[5]) << 2)  | ((uint32_t(out[6]) & 0xC0u) >> 6);
                uint32_t time32 = (t_s_ << 10) | t_ms_;
                time_ = (int64_t)time32;
                continue;
            }

            // 0x6a event 包
            uint32_t tof_ns_u  = (uint32_t(out[1]) << 16) | (uint32_t(out[2]) << 8) | uint32_t(out[3]);
            uint32_t t_width_u = (uint32_t(out[4]) << 12) | (uint32_t(out[5]) << 4) | ((uint32_t(out[6]) & 0xF0u) >> 4);
            int module         = int(uint32_t(out[6]) & 0x0Fu);
            int ch_raw         = int(out[7]);

            if (module < 0 || module >= 8) continue;
            if (ch_raw < 0 || ch_raw >= 144) continue;

            int t_width = int(t_width_u) - 1;
            if (t_width <= 0) continue;

            int ch = chconv0[ch_raw] - 1;
            if (ch < 0 || ch >= 144) continue;

            double t_in = (1000000.0 * 8.0 * double(time_) + double(tof_ns_u) - double(t_width)) / multi;
            double tot = double(t_width) / 8.0;

            double E = gh144_energy(tot, ch, calib_);
            if (!std::isfinite(E) || E <= 5.0 || E > 1500.0) continue;

            ev = Event{t_in, module, ch, E};
            ++accepted_;
            return true;
        }
    }

    uint64_t accepted_count() const { return accepted_; }

private:
    bool open_next_file() {
        while (file_idx_ < 65) {
            const string &fn = inames_[file_idx_++];
            ifs_.open(fn, ios::in | ios::binary);
            if (ifs_.is_open()) return true;
        }
        return false;
    }

private:
    int p_;
    const string* inames_;
    int file_idx_;
    ifstream ifs_;
    uint32_t t_s_;
    uint32_t t_ms_;
    int64_t  time_;
    vector<vector<double>> calib_;
    uint64_t accepted_;
};

struct HeapItem {
    Event ev;
    int   sid;
};
struct HeapCmp {
    bool operator()(const HeapItem &a, const HeapItem &b) const {
        return a.ev.t_in > b.ev.t_in;
    }
};

int main()
{
    string inname[8][65];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 65; j++) {
            inname[i][j] = "PSD000005_00_00" + to_string(i) + "_00" + to_string(j) + ".edb";
        }
    }

    vector<DetectorStream> streams;
    streams.reserve(8);
    for (int i = 0; i < 8; ++i) {
        streams.emplace_back(i, inname[i]);
        if (!streams.back().init()) {
            cerr << "[Detector " << (i + 1) << "] init failed\n";
        }
    }

    priority_queue<HeapItem, vector<HeapItem>, HeapCmp> pq;
    for (int i = 0; i < 8; ++i) {
        Event e;
        if (streams[i].next(e)) {
            pq.push(HeapItem{e, i});
        }
    }

    const double time_window = 400.0;
    uint64_t coinc_count = 0;

    ofstream fout("coin-cc3.txt");
    fout << fixed << setprecision(3);

    Event prev{};
    bool has_prev = false;

    while (!pq.empty()) {
        HeapItem it = pq.top();
        pq.pop();

        const Event &curr = it.ev;

        if (has_prev) {
            double dt = curr.t_in - prev.t_in;
            // 只要在时间窗口内，就进行几何判断
            if (dt <= time_window) {
                int m1 = prev.module;
                int m2 = curr.module;

                // [修改点 2] 几何逻辑改为 B (Intra-module)
                // 必须在同一个 Module 内部发生
                if (m1 == m2) {
                    // 定义 helper lambda 方便判断
                    auto isScatterCh  = [](int c){ return c >= 0  && c < 64;  };
                    auto isAbsorberCh = [](int c){ return c >= 64 && c < 128; };

                    // Case 1: prev=Scatter, curr=Absorber (正常顺序)
                    if (isScatterCh(prev.ch) && isAbsorberCh(curr.ch)) {
                        int cam_s = m1;
                        int cam_a = m1; // 逻辑层面上是同一个 module
                        int ch_s  = prev.ch;
                        int ch_a  = curr.ch - 64; // 归一化

                        double t_sec = curr.t_in * 1e-9;

                        // Absorber Module ID 依然 +8 以示区分
                        fout << cam_s << " "
                             << ch_s  << " "
                             << (cam_a + 8) << " "
                             << ch_a  << " "
                             << prev.E << " "
                             << curr.E << " "
                             << dt     << " "
                             << setprecision(10) << t_sec << "\n";

                        ++coinc_count;
                        fout << fixed << setprecision(3);
                    }
                    // Case 2: prev=Absorber, curr=Scatter (反向顺序，交换 E，dt 取负)
                    else if (isAbsorberCh(prev.ch) && isScatterCh(curr.ch)) {
                        int cam_s = m1;
                        int cam_a = m1;
                        int ch_s  = curr.ch;
                        int ch_a  = prev.ch - 64;

                        double t_sec = curr.t_in * 1e-9;

                        fout << cam_s << " "
                             << ch_s  << " "
                             << (cam_a + 8) << " "
                             << ch_a  << " "
                             << curr.E << " "
                             << prev.E << " "
                             << -dt    << " "
                             << setprecision(10) << t_sec << "\n";

                        ++coinc_count;
                        fout << fixed << setprecision(3);
                    }
                }
            }
        }

        prev = curr;
        has_prev = true;

        Event next_e;
        if (streams[it.sid].next(next_e)) {
            pq.push(HeapItem{next_e, it.sid});
        }
    }

    for (int i = 0; i < 8; ++i) {
        cout << "Detector " << (i + 1) << " Counts: " << streams[i].accepted_count() << "\n";
    }

    cout << "Total coincidence events: " << coinc_count << "\n";
    cout << "Saved to coin-cc3.txt (Logic B: Intra-module)\n";

    return 0;
}
