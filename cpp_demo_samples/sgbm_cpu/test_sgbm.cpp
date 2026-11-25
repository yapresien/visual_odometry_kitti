#include "sgbm_cpu.hpp"
#include <iostream>
#include <random>
#include <cassert>

static void makeSyntheticStereo(std::vector<uint8_t> &left,
                                std::vector<uint8_t> &right,
                                int w, int h, int disp = 16)
{
    left.assign((size_t)w * h, 0);
    // Non-periodic intensity ramp with mild noise, restricted to a rectangle.
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> jitter(-6, 6);
    for (int y = 30; y < 90; ++y)
    {
        for (int x = 80; x < 200; ++x)
        {
            int val = 40 + (x - 80) * 2 + jitter(rng); // unique per column
            val = std::min(250, std::max(5, val));
            left[(size_t)y * w + x] = (uint8_t)val;
        }
    }
    right.assign((size_t)w * h, 0);
    for (int y = 30; y < 90; ++y)
        for (int x = 80; x < 200; ++x)
        {
            int xr = x - disp;
            if (xr >= 0)
                right[(size_t)y * w + xr] = left[(size_t)y * w + x];
        }
}

static float meanValidInROI(const std::vector<float> &disp, int w, int h,
                            int rx, int ry, int rw, int rh)
{
    double sum = 0;
    int n = 0;
    for (int y = ry; y < ry + rh; ++y)
        for (int x = rx; x < rx + rw; ++x)
        {
            float v = disp[(size_t)y * w + x];
            if (v >= 0)
            {
                sum += v;
                ++n;
            }
        }
    return n ? (float)(sum / n) : -1.0f;
}

int main()
{
    const int w = 320, h = 120, disp = 16;

    std::vector<uint8_t> left, right;
    makeSyntheticStereo(left, right, w, h, disp);

    sgbm::Params p;
    p.numDisparities = 64;
    p.blockSize = 5;
    p.disp12MaxDiff = -1;
    p.speckleWindowSize = 0;
    p.uniquenessRatio = 3;
    p.P1 = 200;
    p.P2 = 400;
    sgbm::SgbmCpu sgbm(p);

    auto outputs = sgbm.compute(left, right, w, h);

    // Test 1: output size
    assert(outputs.dispL.size() == (size_t)w * h);

    assert(outputs.dispR.size() == (size_t)w * h);

    assert(outputs.dispLRaw.size() == (size_t)w * h);

    assert(outputs.dispRRaw.size() == (size_t)w * h);

    // Test 2: no NaN/Inf
    for (float v : outputs.dispL)
    {
        assert(
            std::isfinite(v) ||
            v == sgbm::SgbmCpu::kInvalidDisparity);
    }

    for (float v : outputs.dispR)
    {
        assert(
            std::isfinite(v) ||
            v == sgbm::SgbmCpu::kInvalidDisparity);
    }

    for (float v : outputs.dispLRaw)
    {
        assert(
            std::isfinite(v) ||
            v == sgbm::SgbmCpu::kInvalidDisparity);
    }

    for (float v : outputs.dispRRaw)
    {
        assert(
            std::isfinite(v) ||
            v == sgbm::SgbmCpu::kInvalidDisparity);
    }

    // Test 3: mean disparity in textured ROI should be ~16
    float mean = meanValidInROI(
        outputs.dispLRaw,
        w,
        h,
        100,
        40,
        80,
        40);

    std::cout << "[test] ROI mean disparity = " << mean << " (expected ~" << disp << ")\n";

    assert(std::abs(mean - disp) < 4.0f);

    // Test 4: invalid input throws
    size_t validRaw = 0;
    size_t validChecked = 0;

    for (float v : outputs.dispLRaw)
    {
        if (v != sgbm::SgbmCpu::kInvalidDisparity)
            ++validRaw;
    }

    for (float v : outputs.dispL)
    {
        if (v != sgbm::SgbmCpu::kInvalidDisparity)
            ++validChecked;
    }

    assert(validChecked <= validRaw);

    assert(outputs.aggL.size() == outputs.costL.size());

    assert(outputs.aggR.size() == outputs.costR.size());

    assert(!outputs.aggL.empty());
    assert(!outputs.aggR.empty());

    bool thrown = false;

    try
    {
        sgbm.compute(
            {},
            {},
            w,
            h);
    }
    catch (const std::invalid_argument &)
    {
        thrown = true;
    }

    assert(thrown);

    float meanL =         meanValidInROI(
            outputs.dispLRaw,
            w,
            h,
            100,
            40,
            80,
            40);

    float meanR =         meanValidInROI(
            outputs.dispRRaw,
            w,
            h,
            100,
            40,
            80,
            40);

    std::cout << "[test] meanL=" << meanL << " meanR=" << meanR << '\n';
    assert(meanL > 10.0f);
    assert(meanL < 22.0f);
    assert(std::abs(meanL - disp) < 4.0f);

    std::cout << "All SGBM tests passed.\n";
    return 0;
}