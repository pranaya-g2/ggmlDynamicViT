// preprocess.cpp — image -> normalized float32 CHW [3, 224, 224]
//
// Reproduces the PyTorch eval transform:
//   Resize(256, interpolation=BICUBIC)   (PIL backend -> Pillow's resampler)
//   CenterCrop(224)
//   ToTensor()                            (u8 / 255 in float32)
//   Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
//
// The resize is a direct port of Pillow's libImaging/Resample.c (8bpc path):
//   - bicubic kernel with a = -0.5, support 2.0
//   - kernel support widened by the downscale factor (this is the antialiasing)
//   - weights normalized, then converted to fixed point with 22 fractional bits
//   - horizontal pass FIRST, rounded + clamped to uint8, then vertical pass
// Skipping any of these gives +-1..3 LSB differences vs PyTorch.

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "stb_image.h"

#include "dynvit-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace dynvit {

namespace {

constexpr int    PRECISION_BITS  = 32 - 8 - 2;   // Pillow: 22
constexpr double BICUBIC_SUPPORT = 2.0;

double bicubic_filter(double x) {
    constexpr double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

inline uint8_t clip8(int32_t v) {
    v >>= PRECISION_BITS;                       // arithmetic shift, same as Pillow
    return uint8_t(v < 0 ? 0 : (v > 255 ? 255 : v));
}

struct resample_coeffs {
    int ksize = 0;
    std::vector<int>     bounds;  // per output pixel: [xmin, count]
    std::vector<int32_t> kk;      // out_size * ksize, fixed point
};

// Pillow: precompute_coeffs() + normalize_coeffs_8bpc(), with box = [0, in_size].
resample_coeffs precompute_coeffs(int in_size, int out_size) {
    const double scale       = double(in_size) / double(out_size);
    const double filterscale = std::max(scale, 1.0);
    const double support     = BICUBIC_SUPPORT * filterscale;

    resample_coeffs c;
    c.ksize = int(std::ceil(support)) * 2 + 1;
    c.bounds.resize(size_t(out_size) * 2);
    c.kk.assign(size_t(out_size) * c.ksize, 0);

    std::vector<double> k(c.ksize);
    const double ss = 1.0 / filterscale;

    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale;

        int xmin = int(center - support + 0.5);   // truncation toward zero, as in C
        if (xmin < 0) xmin = 0;
        int xmax = int(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;                             // now a count

        double ww = 0.0;
        for (int x = 0; x < xmax; ++x) {
            const double w = bicubic_filter((x + xmin - center + 0.5) * ss);
            k[x] = w;
            ww  += w;
        }
        for (int x = 0; x < xmax; ++x) {
            if (ww != 0.0) k[x] /= ww;
        }
        for (int x = 0; x < xmax; ++x) {
            const double v = k[x] * double(1 << PRECISION_BITS);
            c.kk[size_t(xx) * c.ksize + x] = v < 0 ? int32_t(-0.5 + v) : int32_t(0.5 + v);
        }
        c.bounds[2 * xx + 0] = xmin;
        c.bounds[2 * xx + 1] = xmax;
    }
    return c;
}

image_u8 resample_horizontal(const image_u8 & in, int out_w) {
    const resample_coeffs c = precompute_coeffs(in.w, out_w);

    image_u8 out;
    out.w = out_w;
    out.h = in.h;
    out.data.resize(size_t(out.w) * out.h * 3);

    for (int y = 0; y < in.h; ++y) {
        const uint8_t * row  = in.data.data()  + size_t(y) * in.w  * 3;
        uint8_t       * orow = out.data.data() + size_t(y) * out.w * 3;
        for (int xx = 0; xx < out_w; ++xx) {
            const int       xmin = c.bounds[2 * xx + 0];
            const int       cnt  = c.bounds[2 * xx + 1];
            const int32_t * k    = c.kk.data() + size_t(xx) * c.ksize;

            int32_t s0 = 1 << (PRECISION_BITS - 1);
            int32_t s1 = s0, s2 = s0;
            for (int x = 0; x < cnt; ++x) {
                const uint8_t * p = row + size_t(xmin + x) * 3;
                s0 += int32_t(p[0]) * k[x];
                s1 += int32_t(p[1]) * k[x];
                s2 += int32_t(p[2]) * k[x];
            }
            orow[xx * 3 + 0] = clip8(s0);
            orow[xx * 3 + 1] = clip8(s1);
            orow[xx * 3 + 2] = clip8(s2);
        }
    }
    return out;
}

image_u8 resample_vertical(const image_u8 & in, int out_h) {
    const resample_coeffs c = precompute_coeffs(in.h, out_h);

    image_u8 out;
    out.w = in.w;
    out.h = out_h;
    out.data.resize(size_t(out.w) * out.h * 3);

    for (int yy = 0; yy < out_h; ++yy) {
        const int       ymin = c.bounds[2 * yy + 0];
        const int       cnt  = c.bounds[2 * yy + 1];
        const int32_t * k    = c.kk.data() + size_t(yy) * c.ksize;
        uint8_t       * orow = out.data.data() + size_t(yy) * out.w * 3;

        for (int x = 0; x < in.w; ++x) {
            int32_t s0 = 1 << (PRECISION_BITS - 1);
            int32_t s1 = s0, s2 = s0;
            for (int y = 0; y < cnt; ++y) {
                const uint8_t * p = in.data.data() + (size_t(ymin + y) * in.w + x) * 3;
                s0 += int32_t(p[0]) * k[y];
                s1 += int32_t(p[1]) * k[y];
                s2 += int32_t(p[2]) * k[y];
            }
            orow[x * 3 + 0] = clip8(s0);
            orow[x * 3 + 1] = clip8(s1);
            orow[x * 3 + 2] = clip8(s2);
        }
    }
    return out;
}

// torchvision Resize(int): shorter side -> s, longer side -> int(s * long / short).
void resize_short_side_dims(int w, int h, int s, int & ow, int & oh) {
    if (w <= h) {
        ow = s;
        oh = int(double(s) * h / w);
    } else {
        oh = s;
        ow = int(double(s) * w / h);
    }
}

} // namespace

bool load_image_rgb(const std::string & path, image_u8 & out) {
    int w = 0, h = 0, n = 0;
    unsigned char * px = stbi_load(path.c_str(), &w, &h, &n, 3);
    if (!px) {
        std::fprintf(stderr, "dynvit: failed to load '%s': %s\n", path.c_str(), stbi_failure_reason());
        return false;
    }
    out.w = w;
    out.h = h;
    out.data.assign(px, px + size_t(w) * h * 3);
    stbi_image_free(px);
    return true;
}

image_u8 resize_bicubic_pil(const image_u8 & in, int out_w, int out_h) {
    // Pillow skips a pass when that dimension is unchanged. Horizontal goes first.
    image_u8 tmp = (out_w != in.w) ? resample_horizontal(in, out_w) : in;
    if (out_h != tmp.h) {
        return resample_vertical(tmp, out_h);
    }
    return tmp;
}

image_u8 center_crop(const image_u8 & in, int crop_w, int crop_h) {
    // Python round() is round-half-to-even; nearbyint in the default FP mode matches.
    const int top  = int(std::nearbyint((in.h - crop_h) / 2.0));
    const int left = int(std::nearbyint((in.w - crop_w) / 2.0));

    image_u8 out;
    out.w = crop_w;
    out.h = crop_h;
    out.data.resize(size_t(crop_w) * crop_h * 3);
    for (int y = 0; y < crop_h; ++y) {
        const uint8_t * src = in.data.data() + (size_t(top + y) * in.w + left) * 3;
        std::copy(src, src + size_t(crop_w) * 3, out.data.data() + size_t(y) * crop_w * 3);
    }
    return out;
}

void preprocess_u8(const image_u8 & in, const preprocess_params & p, float * out) {
    int rw = 0, rh = 0;
    resize_short_side_dims(in.w, in.h, p.resize_short, rw, rh);

    const image_u8 resized = resize_bicubic_pil(in, rw, rh);
    const image_u8 cropped = center_crop(resized, p.crop, p.crop);

    // HWC u8 -> CHW f32. Same float32 op order as torchvision: (x / 255 - mean) / std.
    const size_t plane = size_t(p.crop) * p.crop;
    for (int c = 0; c < 3; ++c) {
        float * dst = out + c * plane;
        for (size_t i = 0; i < plane; ++i) {
            const float v = float(cropped.data[i * 3 + c]) / 255.0f;
            dst[i] = (v - p.mean[c]) / p.std[c];
        }
    }
}

bool preprocess_image(const std::string & path, const preprocess_params & p, std::vector<float> & out) {
    image_u8 img;
    if (!load_image_rgb(path, img)) {
        return false;
    }
    out.resize(size_t(3) * p.crop * p.crop);
    preprocess_u8(img, p, out.data());
    return true;
}

} // namespace dynvit