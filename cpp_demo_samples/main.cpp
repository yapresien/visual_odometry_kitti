#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

// ═══════════════════════════════════════════════════════════════════════════
// CONFIG
// ═══════════════════════════════════════════════════════════════════════════
static const bool   USE_OPENCL       = true;   // set false to force CPU-only
static const bool   SHOW_TIMING      = true;
static const bool   SHOW_DISP        = false;
static const bool   SHOW_POOL        = false;
static const bool   USE_Y_CLAMP      = false;
static const int    RENDER_EVERY_N   = 1;
static const int    MAX_FRAMES       = -1;     // -1 = all

// Accuracy constants (mirror Python)
static const int    NUM_DISP         = 128;
static const int    P1_SGBM          = 200;
static const int    P2_SGBM          = 400;
static const int    UNIQUENESS       = 3;
static const int    BLOCK_SIZE       = 5;
static const cv::Size LK_WIN         = cv::Size(21, 21);
static const int    LK_LEVELS        = 4;
static const int    LK_ITERS         = 40;
static const double LK_EPS           = 0.01;
static const double Z_MIN            = 3.0;
static const double Z_MAX            = 200.0;
static const int    RANSAC_ITERS     = 300;
static const double RANSAC_REPROJ    = 3.5;
static const double RANSAC_CONF      = 0.99;
static const int    RANSAC_FLAGS     = cv::SOLVEPNP_SQPNP;
static const double MERGE_THRESHOLD  = 15.0;
static const int    POOL_MIN_POINTS  = 100;
static const int    FRAME_BUFFER     = 16;

// Paths
static const std::string BASEDIR       = "/home/master/myspace/projects/dataset_kitti/sequences/00";
static const std::string CALIB_FILE    = BASEDIR + "/calib.txt";
static const std::string GT_POSES_FILE = "/home/master/myspace/projects/dataset_kitti/poses/00.txt";

// ═══════════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════════
static inline double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static std::vector<std::string> globSorted(const std::string& dir, const std::string& ext = ".png") {
    std::vector<std::string> out;
    if (!fs::exists(dir)) return out;
    for (auto& e : fs::directory_iterator(dir))
        if (e.is_regular_file() && e.path().extension() == ext)
            out.push_back(e.path().string());
    std::sort(out.begin(), out.end());
    return out;
}

static void loadKittiCalib(const std::string& path, cv::Matx33d& K, double& baseline) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open calib " + path);
    cv::Matx34d P0, P1;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string tag; ss >> tag;
        if (tag != "P0:" && tag != "P1:") continue;
        cv::Matx34d M;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c)
                ss >> M(r, c);
        if (tag == "P0:") P0 = M; else P1 = M;
    }
    K = cv::Matx33d(P0(0,0), P0(0,1), P0(0,2),
                    P0(1,0), P0(1,1), P0(1,2),
                    P0(2,0), P0(2,1), P0(2,2));
    baseline = std::abs(P1(0,3) - P0(0,3)) / P0(0,0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Async frame loader
// ═══════════════════════════════════════════════════════════════════════════
struct FramePair { cv::Mat left, right; };

class FrameLoader {
public:
    FrameLoader(const std::vector<std::string>& lp, const std::vector<std::string>& rp, int cap)
        : left_(lp), right_(rp), cap_(cap) {
        worker_ = std::thread([this]{ run(); });
    }
    ~FrameLoader() { stop(); }

    void stop() {
        if (stopped_.exchange(true)) return;
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    bool get(FramePair& out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this]{ return stopped_ || !q_.empty(); });
        if (q_.empty()) return false;
        out = std::move(q_.front()); q_.pop();
        cv_.notify_all();
        return true;
    }

private:
    void run() {
        for (size_t i = 0; i < left_.size(); ++i) {
            if (stopped_) return;
            FramePair p;
            p.left  = cv::imread(left_[i],  cv::IMREAD_GRAYSCALE);
            p.right = cv::imread(right_[i], cv::IMREAD_GRAYSCALE);
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [this]{ return stopped_ || (int)q_.size() < cap_; });
            if (stopped_) return;
            q_.push(std::move(p));
            cv_.notify_all();
        }
    }

    std::vector<std::string> left_, right_;
    int cap_;
    std::queue<FramePair> q_;
    std::mutex m_;
    std::condition_variable cv_;
    std::thread worker_;
    std::atomic<bool> stopped_{false};
};

// ═══════════════════════════════════════════════════════════════════════════
// Pipeline
// ═══════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    bool useOcl = true;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--no-ocl") useOcl = false;

    if (useOcl) {
        cv::ocl::setUseOpenCL(true);
    } else {
        cv::ocl::setUseOpenCL(false);
    }
    std::cout << "[OPENCL] haveOpenCL=" << cv::ocl::haveOpenCL()
              << " useOpenCL="   << cv::ocl::useOpenCL() << "\n";
    if (cv::ocl::useOpenCL()) {
        cv::ocl::Context ctx = cv::ocl::Context::getDefault();
        std::cout << "[OPENCL] device=" << ctx.device(0).name()
                  << " vendor="           << ctx.device(0).vendorName() << "\n";
    }

    // ── Calibration + file lists ────────────────────────────────────────────
    cv::Matx33d K; double baseline = 0.0;
    loadKittiCalib(CALIB_FILE, K, baseline);
    double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);

    auto img0_files = globSorted(BASEDIR + "/image_0");
    auto img1_files = globSorted(BASEDIR + "/image_1");
    if (img0_files.empty() || img1_files.size() != img0_files.size()) {
        std::cerr << "[ERR] image lists\n"; return 1;
    }

    // ── GT ──────────────────────────────────────────────────────────────────
    std::vector<cv::Vec3d> gt_xyz;
    {
        std::ifstream f(GT_POSES_FILE);
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            double v[12]; for (double& x : v) ss >> x;
            gt_xyz.emplace_back(v[3], v[7], v[11]);
        }
    }

    std::cout << "[INIT] baseline=" << baseline << "m  "
              << "frames=" << img0_files.size() << "\n";

    cv::namedWindow("KITTI Stereo Tracking", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Stereo Trajectory Map", cv::WINDOW_AUTOSIZE);

    // ── Stereo matcher (CPU — OpenCL SGBM does not exist) ───────────────────
    auto sgbm = cv::StereoSGBM::create(
        0, NUM_DISP, BLOCK_SIZE,
        P1_SGBM, P2_SGBM,
        1, 63, UNIQUENESS, 100, 32,
        cv::StereoSGBM::MODE_SGBM_3WAY);

    // ── Bootstrap ───────────────────────────────────────────────────────────
    cv::Mat prev_l_mat = cv::imread(img0_files[0], cv::IMREAD_GRAYSCALE);
    cv::Mat prev_r_mat = cv::imread(img1_files[0], cv::IMREAD_GRAYSCALE);
    if (prev_l_mat.empty()) { std::cerr << "[ERR] cannot read frame 0\n"; return 1; }
    int h = prev_l_mat.rows, w = prev_l_mat.cols;

    // These two are used by the accelerated stages and stay on device
    cv::UMat prev_l_umat = prev_l_mat.getUMat(cv::ACCESS_READ);
    cv::UMat prev_r_umat = prev_r_mat.getUMat(cv::ACCESS_READ);

    // Feature detection (may run on OpenCL)
    std::vector<cv::Point2f> pts_prev;
    cv::goodFeaturesToTrack(prev_l_umat, pts_prev, 1500, 0.01, 10);

    // ── Async loader ────────────────────────────────────────────────────────
    std::vector<std::string> lpaths(img0_files.begin()+1, img0_files.end());
    std::vector<std::string> rpaths(img1_files.begin()+1, img1_files.end());
    FrameLoader loader(lpaths, rpaths, FRAME_BUFFER);
    std::cout << "[LOADER] prefetch started, buffer=" << FRAME_BUFFER << "\n";

    // ── State ───────────────────────────────────────────────────────────────
    int gt_drawn_upto = 0;
    cv::Mat R_global = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_global = cv::Mat::zeros(3, 1, CV_64F);
    std::vector<cv::Mat> traj_log;
    { cv::Mat r; cv::hconcat(R_global, t_global, r); traj_log.push_back(r.reshape(1,1).clone()); }

    cv::Mat map_canvas = cv::Mat::zeros(600, 600, CV_8UC3);

    const std::vector<std::string> STAGES = {"read","upload","detect","stereo","lk","pnp","merge","render"};
    std::map<std::string, std::vector<double>> stage_times;
    for (auto& s : STAGES) stage_times[s];
    std::vector<double> total_times;

    const int nFrames = (int)img0_files.size();
    const int lastFrame = (MAX_FRAMES > 0) ? std::min(MAX_FRAMES, nFrames - 1) : nFrames - 1;

    // ═══════════════════════════════════════════════════════════════════════
    // MAIN LOOP
    // ═══════════════════════════════════════════════════════════════════════
    for (int frame_idx = 1; frame_idx <= lastFrame; ++frame_idx) {
        auto t_frame = Clock::now();

        // ── read ────────────────────────────────────────────────────────────
        auto t = Clock::now();
        FramePair fp;
        if (!loader.get(fp)) { std::cerr << "[WARN] loader stopped\n"; break; }
        cv::Mat curr_l_mat = std::move(fp.left);
        cv::Mat curr_r_mat = std::move(fp.right);
        stage_times["read"].push_back(msSince(t));
        if (curr_l_mat.empty() || curr_r_mat.empty()) continue;

        // ── upload (Mat → UMat; cheap if already in OpenCL memory) ─────────
        t = Clock::now();
        cv::UMat curr_l_umat = curr_l_mat.getUMat(cv::ACCESS_READ);
        cv::UMat curr_r_umat = curr_r_mat.getUMat(cv::ACCESS_READ);
        stage_times["upload"].push_back(msSince(t));

        // ── detect (only if pool small) ────────────────────────────────────
        t = Clock::now();
        if ((int)pts_prev.size() < POOL_MIN_POINTS) {
            cv::goodFeaturesToTrack(prev_l_umat, pts_prev, 1500, 0.01, 10);
            if (pts_prev.empty()) {
                prev_l_mat = curr_l_mat; prev_r_mat = curr_r_mat;
                prev_l_umat = curr_l_umat; prev_r_umat = curr_r_umat;
                stage_times["detect"].push_back(msSince(t));
                continue;
            }
        }
        stage_times["detect"].push_back(msSince(t));

        // ── stereo (CPU; output is Mat for PnP pixel access) ───────────────
        t = Clock::now();
        cv::Mat disp16;
        sgbm->compute(prev_l_mat, prev_r_mat, disp16);
        cv::Mat disp_float;
        disp16.convertTo(disp_float, CV_32F, 1.0 / 16.0);
        stage_times["stereo"].push_back(msSince(t));

        if (SHOW_DISP) {
            double sum = 0; int n = 0;
            for (int y = 0; y < disp_float.rows; ++y) {
                const float* row = disp_float.ptr<float>(y);
                for (int x = 0; x < disp_float.cols; ++x) {
                    float d = row[x];
                    if (d > 0.5f && d < NUM_DISP - 1) { sum += d; ++n; }
                }
            }
            if (n) std::cout << "[DISP] mean=" << sum/n
                             << " valid=" << 100.0*n/disp_float.total() << "%\n";
        }

        // ── LK (may run on OpenCL) ──────────────────────────────────────────
        t = Clock::now();
        std::vector<cv::Point2f> next_pts;
        std::vector<uchar> st;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(
            prev_l_umat, curr_l_umat, pts_prev, next_pts, st, err,
            LK_WIN, LK_LEVELS,
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                             LK_ITERS, LK_EPS));
        stage_times["lk"].push_back(msSince(t));

        int n = std::min({(int)pts_prev.size(), (int)next_pts.size(), (int)st.size()});
        std::vector<cv::Point2f> good_prev, good_curr;
        good_prev.reserve(n); good_curr.reserve(n);
        for (int i = 0; i < n; ++i)
            if (st[i]) { good_prev.push_back(pts_prev[i]); good_curr.push_back(next_pts[i]); }

        // ── PnP (CPU) ──────────────────────────────────────────────────────
        t = Clock::now();
        std::vector<uchar> inlier_mask(good_prev.size(), 0);
        std::vector<int>   valid_idx;
        if ((int)good_prev.size() > 15) {
            std::vector<cv::Point3f> P3; std::vector<cv::Point2f> P2;
            std::vector<int> idx;
            for (int i = 0; i < (int)good_prev.size(); ++i) {
                int x = (int)std::lround(good_prev[i].x);
                int y = (int)std::lround(good_prev[i].y);
                if (x < 0 || x >= w || y < 0 || y >= h) continue;
                float d = disp_float.at<float>(y, x);
                if (d <= 0.5f || d >= NUM_DISP - 1) continue;
                double Z = fx * baseline / d;
                if (Z <= Z_MIN || Z > Z_MAX) continue;
                P3.emplace_back((float)((x - cx) * Z / fx),
                                (float)((y - cy) * Z / fy),
                                (float)Z);
                P2.push_back(good_curr[i]);
                idx.push_back(i);
            }
            if ((int)P3.size() > 6) {
                cv::Mat rvec, tvec, inl;
                cv::Mat Kmat(K);
                bool ok = cv::solvePnPRansac(P3, P2, Kmat, cv::noArray(),
                                             rvec, tvec, false, RANSAC_ITERS,
                                             (float)RANSAC_REPROJ, RANSAC_CONF,
                                             inl, RANSAC_FLAGS);
                if (ok && !inl.empty()) {
                    cv::Mat Rm; cv::Rodrigues(rvec, Rm);
                    cv::Mat Rd = Rm.t();
                    cv::Mat td = -Rm.t() * tvec;
                    t_global = t_global + R_global * td;
                    R_global = R_global * Rd;
                    if (USE_Y_CLAMP && std::abs(t_global.at<double>(1,0)) > 3.0)
                        t_global.at<double>(1,0) *= 0.98;

                    cv::Mat row; cv::hconcat(R_global, t_global, row);
                    traj_log.push_back(row.reshape(1,1).clone());

                    for (int k = 0; k < inl.rows; ++k) {
                        int ii = idx[inl.at<int>(k,0)];
                        inlier_mask[ii] = 1;
                        valid_idx.push_back(ii);
                    }
                }
            }
        }
        stage_times["pnp"].push_back(msSince(t));

        // ── merge ──────────────────────────────────────────────────────────
        t = Clock::now();
        std::vector<cv::Point2f> kept;
        int nIn = 0; for (auto v : inlier_mask) nIn += v;
        if (nIn > 10) {
            for (int i = 0; i < (int)inlier_mask.size(); ++i)
                if (inlier_mask[i]) kept.push_back(good_curr[i]);
        } else if (!valid_idx.empty()) {
            for (int i : valid_idx) kept.push_back(good_curr[i]);
        } else {
            kept = good_curr;
        }

        // Fresh detection on current frame (UMat in, may be OpenCL)
        std::vector<cv::Point2f> fresh;
        cv::goodFeaturesToTrack(curr_l_umat, fresh, 1500, 0.01, 10);

        std::vector<cv::Point2f> merged;
        if (!kept.empty() && !fresh.empty()) {
            merged = kept;
            const float thr2 = (float)(MERGE_THRESHOLD * MERGE_THRESHOLD);
            for (const auto& f : fresh) {
                float best = thr2;
                for (const auto& k : kept) {
                    float dx = f.x - k.x, dy = f.y - k.y;
                    float d2 = dx*dx + dy*dy;
                    if (d2 < best) best = d2;
                }
                if (best >= thr2) merged.push_back(f);  // no kept point within threshold
            }
        } else if (!kept.empty()) {
            merged = kept;
        } else if (!fresh.empty()) {
            merged = fresh;
        }

        if ((int)merged.size() < POOL_MIN_POINTS) {
            std::vector<cv::Point2f> fb;
            cv::goodFeaturesToTrack(curr_l_umat, fb, 1500, 0.01, 10);
            if (!fb.empty()) merged = fb;
        }
        stage_times["merge"].push_back(msSince(t));

        pts_prev = std::move(merged);
        prev_l_mat = std::move(curr_l_mat);
        prev_r_mat = std::move(curr_r_mat);
        prev_l_umat = std::move(curr_l_umat);
        prev_r_umat = std::move(curr_r_umat);

        if (SHOW_POOL)
            std::cout << "[POOL] kept=" << kept.size() << " merged=" << pts_prev.size() << "\n";

        // ── render ─────────────────────────────────────────────────────────
        t = Clock::now();
        if (frame_idx % RENDER_EVERY_N == 0) {
            cv::Mat side;
            cv::hconcat(prev_l_mat, prev_r_mat, side);
            cv::Mat img; cv::cvtColor(side, img, cv::COLOR_GRAY2BGR);
            for (auto& p : good_curr)
                cv::circle(img, p, 2, cv::Scalar(0,255,0), -1);
            cv::imshow("KITTI Stereo Tracking", img);

            double x_coord = t_global.at<double>(0,0);
            double z_coord = t_global.at<double>(2,0);
            cv::circle(map_canvas,
                { (int)(-x_coord*0.5) + 300, (int)(z_coord*0.5) + 100 },
                2, cv::Scalar(0,165,255), -1);

            int gStart = std::max(1, gt_drawn_upto + 1);
            int gEnd   = std::min(frame_idx + 1, (int)gt_xyz.size());
            for (int g = gStart; g < gEnd; ++g) {
                cv::Point p1{ (int)(-gt_xyz[g-1][0]*0.5)+300, (int)(gt_xyz[g-1][2]*0.5)+100 };
                cv::Point p2{ (int)(-gt_xyz[g  ][0]*0.5)+300, (int)(gt_xyz[g  ][2]*0.5)+100 };
                cv::line(map_canvas, p1, p2, cv::Scalar(0,0,255), 1);
            }
            cv::imshow("Stereo Trajectory Map", map_canvas);
            gt_drawn_upto = frame_idx;

            if ((cv::waitKey(1) & 0xFF) == 'q') {
                std::cout << "[EXIT] 'q'\n"; break;
            }
        }
        stage_times["render"].push_back(msSince(t));
        total_times.push_back(msSince(t_frame));

        // Make sure queued OpenCL work is actually done before we measure end
        if (cv::ocl::useOpenCL()) cv::ocl::finish();
    }

    // ── Shutdown ────────────────────────────────────────────────────────────
    loader.stop();
    {
        std::ofstream ofs("est_traj.txt");
        for (auto& m : traj_log) {
            cv::Mat r = m.reshape(1, 1);
            for (int i = 0; i < r.cols; ++i) {
                if (i) ofs << ' ';
                ofs << r.at<double>(0, i);
            }
            ofs << '\n';
        }
        std::cout << "[SAVE] " << traj_log.size() << " rows → est_traj.txt\n";
    }
    cv::destroyAllWindows();

    // ── Profiling ───────────────────────────────────────────────────────────
    if (SHOW_TIMING && !total_times.empty()) {
        auto stat = [](const std::vector<double>& v,
                       double& mean, double& mn, double& mx, double& sum) {
            mean = mn = mx = sum = 0;
            if (v.empty()) return;
            sum = 0; mn = 1e30; mx = -1e30;
            for (double x : v) { sum += x; mn = std::min(mn, x); mx = std::max(mx, x); }
            mean = sum / v.size();
        };
        std::cout << "\n" << std::string(72, '=') << "\n";
        std::cout << "PROFILE  OpenCL=" << (cv::ocl::useOpenCL() ? "ON" : "OFF")
                  << "  —  " << total_times.size() << " frames\n";
        std::cout << std::string(72, '-') << "\n";
        printf("  %-9s  %7s  %7s  %7s  %8s\n", "stage", "mean", "min", "max", "total");
        for (auto& s : STAGES) {
            auto& v = stage_times[s];
            double mean, mn, mx, sum;
            stat(v, mean, mn, mx, sum);
            if (!v.empty())
                printf("  %-9s  %6.2f  %6.2f  %6.2f  %6.1fs\n",
                       s.c_str(), mean, mn, mx, sum / 1000.0);
        }
        double mean, mn, mx, sum;
        stat(total_times, mean, mn, mx, sum);
        std::cout << std::string(72, '-') << "\n";
        printf("  %-9s  %6.2f  %6.2f  %6.2f  %6.1fs\n",
               "TOTAL", mean, mn, mx, sum / 1000.0);
        printf("  Throughput: %.1f fps\n", 1000.0 / mean);
        std::cout << std::string(72, '=') << "\n";
    }
    return 0;
}