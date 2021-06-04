# References: 
# https://www.iquilezles.org/www/articles/fbm/fbm.htm

import taichi as ti
import numpy as np
import taichi_glsl as ts

ti.init(arch=ti.gpu, print_kernel_llvm_ir = True)

res = 512
fbm_octaves = 10
noise_freq = 2.0
warp_freq = 8.0
noise_mag = 2.0
tv_static = False
fbm_v = False
perlin_v = False
worley_v = True

radius = 3

noise_greyscale = ti.field(ti.f32, shape=(res, res))
noise_outlook = ti.Vector.field(3, float, shape=(res, res))
m = ti.Vector.field(3, shape=(4), dtype=float)
s = ti.Vector.field(3, shape=(4), dtype=float)
noise_afterprocess = ti.Vector.field(3, float, shape=(res, res))

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

@ti.func
def surflet(p, gridp):
    distx = ti.abs(p.x - gridp.x)
    disty = ti.abs(p.y - gridp.y)
    tx = 1.0 - 6.0 * (distx ** 5.0) + 15.0 * (distx ** 4.0) - 10.0 *  (distx ** 3.0)
    ty = 1.0 - 6.0 * (disty ** 5.0) + 15.0 * (disty ** 4.0) - 10.0 *  (disty ** 3.0)

    gradient = randomv2(gridp.x, gridp.y) * 2.0 - ti.Vector([1.0, 1.0])
    diff = p - gridp
    height = diff.dot(gradient)
    return height * tx * ty

@ti.func
def perlin(u, v):
    u *= 10.0
    v *= 10.0
    uv = ti.Vector([u, v])
    bl = ti.Vector([ti.floor(u), ti.floor(v)])
    br = bl + ti.Vector([1, 0])
    tr = bl + ti.Vector([1, 1])
    tl = bl + ti.Vector([0, 1])
    return surflet(uv, bl) + surflet(uv, br) + surflet(uv, tr) + surflet(uv, tl)

@ti.func
def sample(f, i, j):
    I = ti.Vector([int(i), int(j)])
    I = max(0, min(res - 1, I))
    return f[I]

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
        f[i, j] = ti.Vector([h, h, h])

#perlin noise
@ti.kernel
def draw_perlin(f: ti.template()):
    for i, j in f:
        u = i / res
        v = j / res
        h = perlin(u, v)
        f[i, j] = h

@ti.kernel
def apply_revert(f: ti.template(), new_f: ti.template()):
    for i, j in f:
        new_f[i, j] = ti.Vector([1 - f[i, j].x,
                                 1 - f[i, j].y,
                                 1 - f[i, j].z])

@ti.kernel
def apply_kuwahara(f: ti.template(), new_f: ti.template()):
    
    for i, j in f:
        n = ((radius + 1) ** 2)
        # for y in range(-radius, 1):
            # for x in range(-radius, 1):
                # c = sample(f, i + x, j + y)
        m[0] += 1.0
                # s[0] += s[0] + ti.Vector([c.x * c.x, c.y * c.y, c.z * c.z])
        
        # for y in range(-radius, 1):
        #     for x in range(0, radius + 1):
        #         c = sample(f, i + x, j + y)
        #         m[1] = m[1] + c
        #         s[1] = s[1] + ti.Vector([c.x * c.x, c.y * c.y, c.z * c.z])
        
        # for y in range(0, radius + 1):
        #     for x in range(0, radius + 1):
        #         c = sample(f, i + x, j + y)
        #         m[2] += c
        #         s[2] += c * c
        
        # for y in range(0, radius + 1):
        #     for x in range(-radius, 1):
        #         c = sample(f, i + x, j + y)
        #         m[3] += c
        #         s[3] += c * c

        # min_sigma2 = 1e+2
        # for k in range(0, 4):
        #     m[k] /= n
        #     s[k] = abs(s[k] / n - m[k] * m[k])

        #     sigma2 = s[k].x + s[k].y + s[k].z
        #     if sigma2 < min_sigma2:
        #         min_sigma2 = sigma2
        #         r = m[k].x
        #         g = m[k].y
        #         b = m[k].z
        #         new_f[i, j] = ti.Vector([r, g, b])

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
# ti.set_logging_level(ti.DEBUG)

while not gui.get_event(ti.GUI.ESCAPE):
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
        # draw_worley(noise_outlook)
        apply_kuwahara(noise_outlook, noise_afterprocess)
        gui.set_image(noise_afterprocess)
    else:
        draw_perlin(noise_greyscale)
        gui.set_image(noise_greyscale)
    gui.show()
