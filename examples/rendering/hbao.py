from PIL import Image
import numpy as np
from numpy.core.fromnumeric import shape
import taichi_glsl as ts

import taichi as ti

ti.init(arch=ti.cpu, cpu_max_num_threads=1)
# ti.init(arch=ti.gpu, kernel_profiler=True)
res = 2048
DIR_NUM = 16
STEP_NUM = 8
STEP_NUM_SD = 11
PI = 3.1415926
INV_2PI = 1.0 / (2 * PI)

in_tex = ti.field(ti.f32, shape=(res, res))
out_tex = ti.field(ti.f32, shape=(res, res))
mipmap_tex = ti.field(ti.f32, shape=(res, res))

noise_tex = ti.Vector.field(3, ti.f32, shape=(16, 16))

source = Image.open('source.png')
source = source.rotate(-90)
source_np = np.array(source) / 255.0
source_np = np.float32(source_np)

mipmap = Image.open('mipmap.png')
mipmap = mipmap.rotate(-90)
mipmap_np = np.array(mipmap) / 255.0
mipmap_np = np.float32(mipmap_np)

in_tex.from_numpy(source_np)
mipmap_tex.from_numpy(mipmap_np)

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
def hbao_sd(radius: ti.f32, height_depth: ti.f32, src: ti.template(), mipmap: ti.template(),
            dst: ti.template()):
    ti.static_assert(dst.shape == src.shape,
                     "needs src and dst fields to be same shape")
    ti.block_dim(1024)
    sizemin = src.shape[0]
    scale = height_depth * sizemin
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
                # S = P + sample_offset
                S = ti.cast(P + sample_offset, int)
                S = S & (res - 1)
                mip_res = res >> mip_level
                mip_S = S / res * mip_res
                x = ti.cast(mip_S[0], int)
                y = ti.cast(mip_S[1], int)
                sample_depth = mipmap[x & (res - 1), (mip_res + y) & (res - 1)]
                # sample_depth = src[S[0] & 2047, S[1] & 2047]
                # we upres offset for next level, deliberately upres before dividing depth because it's so in SD
                offset = offset << 1
                offset_inv = offset_inv * 0.5
                # we see it as depth contribution normalized by offset
                d = (sample_depth - pixel_depth) * offset_inv * 0.5
                max_d = ti.max(max_d, d)
                # would be better if we have "saturate" operator
                # radius here basically decide how many miplevel will contribute I guess
                interp = min(
                    max(((1 << (STEP_NUM_SD - mip_level)) * radius - 1.0),
                        0.0), 1.0)
                max_ao = lerp(max_ao, max_d, interp)
            max_ao *= (scale * 2)
            max_ao = max_ao / (ti.sqrt(1 + max_ao * max_ao))
            accum_ao += max_ao
        dst[P] = 1.0 - accum_ao / DIR_NUM

@ti.func
def sample_mipmap_func(src:ti.template(), dst:ti.template(), x, y, mip_level):
    '''
    x: x in 0-1
    y: y in 0-1
    '''
    mip_size = 1024
    for i in range(1, mip_level):
        mip_size = mip_size >> 1
    pos = ti.Vector([x,y])
    i_pos = ti.cast(pos * 2048, ti.i32)
    mip_pos = ti.cast(pos * mip_size, ti.i32)
    dst[i_pos] = src[mip_pos.x, (mip_pos.y + mip_size)]
    '''
    1: 0
    2: 1024
    3: 1024 + 512
    '''

@ti.kernel
def sample_mipmap(src:ti.template(), dst:ti.template(), mip_level:ti.i32):
    for P in ti.grouped(src):
        normalize_p = P / res
        sample_mipmap_func(src, dst, normalize_p.x, normalize_p.y, mip_level)


def halve_image(image):
    rows, cols = image.shape
    image = image.astype('uint16')
    image = image.reshape(rows // 2, 2, cols // 2, 2)
    image = image.sum(axis=3).sum(axis=1)
    return ((image + 2) >> 2).astype('uint8')

def mipmap(image):
    img = image.copy()
    rows, cols = image.shape
    mipmap = np.zeros((rows, cols), dtype='uint8')
    row = 0
    while rows > 1:
        img = halve_image(img)
        rows = img.shape[0]
        mipmap[row:row + rows, :img.shape[1]] = img
        row += rows
    return mipmap

img = np.array(Image.open('source.png'))
Image.fromarray(mipmap(img)).save('mipmap.png')

gui = ti.GUI('HBAO', res=(res, res))
init_noise()
# hbao(1, 0.1, in_tex, out_tex)
# hbao_sd(1, 0.1, in_tex, out_tex)
height = gui.slider('height', 0, 1, step=0.1)
radius = gui.slider('radius', 0, 1, step=0.1)
height.value = 1.0
radius.value = 1.0
# for i in range(1000):
ti.kernel_profiler_print()
while True:
    # sample_mipmap(mipmap_tex, out_tex, 10)
    hbao_sd(height.value, radius.value, in_tex, mipmap_tex, out_tex)
    gui.set_image(out_tex.to_numpy())
    gui.show()
