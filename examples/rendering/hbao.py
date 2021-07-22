from PIL import Image
import numpy as np
from numpy.core.fromnumeric import shape

import taichi as ti

ti.init(arch=ti.gpu, kernel_profiler=True)
res = 2048
DIR_NUM = 16
STEP_NUM = 8
STEP_NUM_SD = 11
PI = 3.1415926
INV_2PI = 1.0 / (2 * PI)

in_tex = ti.field(ti.f32, shape=(res, res))
out_tex = ti.field(ti.f32, shape=(res, res))

noise_tex = ti.Vector.field(3, ti.f32, shape=(16, 16))

source = Image.open('source.png')
source = source.rotate(-90)
source_np = np.array(source) / 255.0
source_np = np.float32(source_np)

in_tex.from_numpy(source_np)


# W(θ)
@ti.func
def falloff(distance: ti.f32, radius: ti.f32):
    return ti.max(0, 1.0 - distance / radius)


@ti.func
def rand_uv(distance: ti.f32, radius: ti.f32):
    return ti.max(0, 1.0 - distance / radius)


@ti.kernel
def init_noise():
    for P in ti.grouped(noise_tex):
        noise_tex[P] = ti.Vector([ti.random(), ti.random(), ti.random()])


@ti.kernel
def hbao(radius: ti.f32, height_depth: ti.f32, src: ti.template(),
         dst: ti.template()):
    ti.static_assert(dst.shape == src.shape,
                     "needs src and dst fields to be same shape")
    resx = src.shape[0]
    radius_px = radius * resx
    height = height_depth * resx
    step_vec = ti.Vector([radius_px, radius_px]) / STEP_NUM
    step_size = ti.sqrt(step_vec.dot(step_vec))
    for P in ti.grouped(src):
        # for each sample direction
        accum_ao = 0.0
        pixel_depth = src[P]
        # rand = ti.random()
        # rand = noise_tex[P & 0x11]
        for dir_id in ti.static(range(DIR_NUM)):
            # may need some jitter here, but with jitter dramaticly slow down the kernel
            # dir_ang = (dir_id + rand.x) * 2.0 * PI / DIR_NUM
            dir_ang = dir_id * 2.0 * PI / DIR_NUM
            dir_vec = ti.Vector([ti.cos(dir_ang), ti.sin(dir_ang)]) * step_vec
            # for each step
            max_ao = 0.0
            for step_id in ti.static(range(1, STEP_NUM + 1)):
                # round to pixel
                sample_offset = step_id * dir_vec
                sample_depth = src[ti.cast(P + sample_offset, int)]
                if sample_depth > pixel_depth:
                    d = (sample_depth - pixel_depth) * height
                    r = ti.sqrt(d * d + sample_offset.dot(sample_offset))
                    sample_ao = d / r * falloff(step_size * step_id, radius_px)
                    max_ao = ti.max(max_ao, sample_ao)
            accum_ao += max_ao
        dst[P] = 1.0 - accum_ao / DIR_NUM


@ti.func
def lerp(x: ti.f32, y: ti.f32, w: ti.f32):
    return x + (y - x) * w


@ti.kernel
def hbao_sd(radius: ti.f32, height_depth: ti.f32, src: ti.template(),
            dst: ti.template()):
    ti.static_assert(dst.shape == src.shape,
                     "needs src and dst fields to be same shape")
    ti.block_dim(1024)
    scale = height_depth * src.shape[0]
    for P in ti.grouped(src):
        # for each sample direction
        accum_ao = 0.0
        pixel_depth = src[P]
        for dir_id in ti.static(range(DIR_NUM)):
            dir_ang = dir_id * 2.0 * PI / DIR_NUM
            dir_vec = ti.Vector([ti.cos(dir_ang), ti.sin(dir_ang)])
            # for each step
            # substance designer use a power-exponent step size like cone tracing, rather than fixed step
            max_ao = 0.0
            max_d = 0.0
            offset = 1
            offset_inv = 1.0
            for mip_level in ti.static(range(1, STEP_NUM_SD + 1)):
                sample_offset = offset * dir_vec
                # actually it should sample mipmap here, but we don't have texture object so mimic with mip 0
                S = ti.cast(P + sample_offset, int)
                sample_depth = src[S[0] & (res - 1), S[1] & (res - 1)]
                # we upres offset for next level, deliberately upres before dividing depth because it's so in SD
                offset = offset << 1
                offset_inv = offset_inv * 0.5
                # we see it as depth contribution normalized by offset
                d = (sample_depth - pixel_depth) * offset_inv
                max_d = ti.max(max_d, d)
                # would be better if we have "saturate" operator
                # radius here basically decide how many miplevel will contribute I guess
                interp = min(
                    max(((1 << (STEP_NUM_SD - mip_level)) * radius - 1.0),
                        0.0), 1.0)
                max_ao = lerp(max_ao, max_d, interp)
            # not sure what "sizeMin" in SD graph means, here use a magic number 10.0
            # would be better if we have "pow2" operator
            max_ao *= (scale * 2)
            max_ao = max_ao / (ti.sqrt(1 + max_ao * max_ao))
            accum_ao += max_ao
        dst[P] = 1.0 - accum_ao / DIR_NUM


gui = ti.GUI('HBAO', res=(res, res))
init_noise()
# hbao(1, 0.1, in_tex, out_tex)
# hbao_sd(1, 0.1, in_tex, out_tex)

# for i in range(1000):
hbao_sd(1, 0.1, in_tex, out_tex)
ti.kernel_profiler_print()
while True:
    gui.set_image(out_tex.to_numpy())
    gui.show()
