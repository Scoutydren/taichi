# References: 
# https://www.iquilezles.org/www/articles/fbm/fbm.htm

import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

res = 512
fbm_octaves = 10
noise_freq = 2.0
warp_freq = 8.0
noise_mag = 2.0
tv_static = False
fbm_v = False
perlin_v = False
worley_v = True

noise_greyscale = ti.field(ti.f32, shape=(res, res))
noise_outlook = ti.Vector.field(3, float, shape=(res, res))

@ti.func
def fract(i):
    return i - ti.floor(i)

@ti.func
def lerp(l, r, frac):
    return l + frac * (r - l)

@ti.func
def quintic_interpolate(l, r, t):
    t = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    return lerp(l, r, t)

# give pseudo randomized float values
@ti.func
def randomf(i, j):
    vec2 = ti.Vector([i, j])
    randv1 = ti.Vector([127.1, 311.7])
    dot1 = vec2.dot(randv1)
    randf = ti.sin(dot1)
    return fract(randf * 2873.2341)

# give pseudo randomized vector values
@ti.func
def randomv2(i, j):
    vec2 = ti.Vector([i, j])
    randv1 = ti.Vector([127.1, 311.7])
    randv2 = ti.Vector([269.5, 183.3])
    dot1 = vec2.dot(randv1)
    dot2 = vec2.dot(randv2)
    randv3 = ti.Vector([dot1, dot2])
    randv4 = ti.sin(randv3)
    return fract(randv4 * 43758.5453)

@ti.func
def noise2d(i, j):
    ll = randomf(ti.floor(i), ti.floor(j))
    lr = randomf(ti.floor(i) + 1.0, ti.floor(j))
    ul = randomf(ti.floor(i), ti.floor(j) + 1.0)
    ur = randomf(ti.floor(i) + 1.0, ti.floor(j) + 1.0)
    lerpxl = quintic_interpolate(ll, lr, fract(i))
    lerpxu = quintic_interpolate(ul, ur, fract(i))
    return quintic_interpolate(lerpxl, lerpxu, fract(j))

#h: self-similarity
@ti.func
def fbm(x, y, h):
    gain = 2 ** (-h)
    freq = 8.0
    amp = 0.5
    t = 0.0
    for _ in range(fbm_octaves):
        t += amp * noise2d(freq * x, freq * y)
        freq *= 2.0
        amp *= gain
    return t

@ti.func
def worley(u, v):
    u *= 10.0
    v *= 10.0
    minDist = 1.0
    uv_int = ti.Vector([ti.floor(u), ti.floor(v)])
    uv_fract = ti.Vector([fract(u), fract(v)])
    for x in range(-1, 2): # loop through neighboring cells
        for y in range(-1, 2):
            loc = uv_int + ti.Vector([x, y])
            centerp = randomv2(loc.x, loc.y)
            diff = ti.Vector([x, y]) + centerp - uv_fract
            dist = diff.norm()
            minDist = ti.min(minDist, dist)
    return minDist


## TV static noise
@ti.kernel
def draw_tv_noise(f: ti.template()):
    for i, j in f:
        u = i / res
        v = j / res
        h = randomf(u, v)
        f[i, j] = h

#FBM noise
@ti.kernel
def draw_fbm(f: ti.template()):
    for i, j in f:
        u = i / res
        v = j / res
        h = fbm(u * noise_freq, v * noise_freq, 1.0)
        f[i, j] = h

#worley noise
@ti.kernel
def draw_worley(f: ti.template()):
    for i, j in f:
        u = i / res
        v = j / res
        h = worley(u, v)
        f[i, j] = h

#perlin noise
def draw_perlin(f: ti.template()):
    for i, j in f: 
        u = i / res
        v = j / res
        

@ti.kernel
def test(i: ti.f32, j: ti.f32) -> ti.f32:
    vec2 = ti.Vector([i, j])
    randv1 = ti.Vector([127.1, 311.7])
    randv2 = ti.Vector([269.5, 183.3])
    dot1 = vec2.dot(randv1)
    dot2 = vec2.dot(randv2)
    randv3 = ti.Vector([dot1, dot2])
    print(randv3)
    randv4 = ti.sin(randv3 * 43.5453)
    print(randv4 - ti.floor(randv4))

gui = ti.GUI('Noise', (res, res))

while gui.running:
    #draw tv static noise
    if tv_static:
        draw_tv_noise(noise_greyscale)
        gui.set_image(noise_greyscale)
    #draw fbm noise
    elif fbm_v:
        draw_fbm(noise_greyscale)
        gui.set_image(noise_greyscale)
    #draw worley noise
    elif worley_v:
        draw_worley(noise_greyscale)
        gui.set_image(noise_greyscale)
    gui.show()
