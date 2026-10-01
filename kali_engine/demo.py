import sys
import math
import time
import numpy as np
import pygame
import moderngl
from PIL import Image

VERTEX_SHADER = """
#version 330
in vec2 in_position;
out vec2 uv;
void main() {
    uv = in_position;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
out vec4 fragColor;
in vec2 uv;

uniform vec2 iResolution;
uniform vec3 iCamPos;
uniform vec3 iCamDir;
uniform vec3 iCamUp;
uniform vec3 iCamRight;
uniform float iTime;
uniform float iFOV;

uniform int iFractalType; // 0 ile 9 arası (10 Üst Seviye Fraktal)[cite: 7]
uniform float iBloomIntensity;
uniform bool iFlashlightOn;
uniform bool iDarknessMode;
uniform bool iDOFEnabled;
uniform bool iGravLensEnabled;
uniform float iLightIntensity;
uniform bool iForceSphereActive;   // 1 tuşu ile açılıp kapanan kuvvet küresi modu
uniform vec3 iForceSpherePos;      // Kürenin konumu

#define PI 3.14159265359

// Mikro Yüzey Detayı İçin Hash[cite: 7]
float hash(vec3 p) {
    p = fract(p * vec3(443.897, 441.423, 437.195));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

// --- 10 ADET ÜST SEVİYE MATEMATİKSEL FRAKTAL ---[cite: 7]
vec4 sdGiantMandelbox(vec3 pos) {
    vec3 p = mod(pos + 10.0, 20.0) - 10.0;
    vec3 offset = p;
    float dr = 1.0;
    float scale = 2.1;
    vec3 trap = vec3(1e10);

    for (int i = 0; i < 12; i++) {
        p = clamp(p, -1.8, 1.8) * 2.0 - p;
        float r2 = dot(p, p);
        if (r2 < 0.35) {
            p *= 3.2; dr *= 3.2;
        } else if (r2 < 1.6) {
            float temp = 1.6 / r2;
            p *= temp; dr *= temp;
        }
        p = p * scale + offset;
        dr = dr * abs(scale) + 1.0;
        trap = min(trap, abs(p));
    }
    return vec4(trap, length(p) / abs(dr));
}

vec4 sdGiantMengerSponge(vec3 pos) {
    vec3 p = mod(pos + 8.0, 16.0) - 8.0;
    float d = max(abs(p.x), max(abs(p.y), abs(p.z))) - 4.0;
    float s = 1.0;
    vec3 trap = vec3(1e10);

    for (int m = 0; m < 5; m++) {
        vec3 a = mod(p * s, 2.0) - 1.0;
        s *= 2.6;
        vec3 r = abs(1.0 - 3.0 * abs(a));
        float da = max(r.x, r.y);
        float db = max(r.y, r.z);
        float dc = max(r.z, r.x);
        float c = (min(da, min(db, dc)) - 1.0) / s;
        d = max(d, c);
        trap = min(trap, vec3(length(a.xy), length(a.yz), length(a.zx)));
    }
    return vec4(trap, d);
}

vec4 sdMandelbulb(vec3 pos) {
    vec3 w = mod(pos + 4.0, 8.0) - 4.0;
    float dr = 1.0;
    float r = 0.0;
    float power = 8.0;
    vec3 trap = vec3(1e10);

    for (int i = 0; i < 8; i++) {
        r = length(w);
        if (r > 4.0) break;
        float theta = acos(w.z / r);
        float phi = atan(w.y, w.x);
        dr = pow(r, power - 1.0) * power * dr + 1.0;
        float zr = pow(r, power);
        w = zr * vec3(sin(theta * power) * cos(phi * power), sin(phi * power) * sin(theta * power), cos(theta * power)) + pos * 0.2;
        trap = min(trap, abs(w));
    }
    return vec4(trap, 0.5 * log(r) * r / dr);
}

vec4 sdKleinianJulia(vec3 pos) {
    vec3 p = mod(pos + 6.0, 12.0) - 6.0;
    float scale = 1.0;
    vec3 trap = vec3(1e10);
    for (int i = 0; i < 9; i++) {
        p = -1.0 + 2.0 * fract(0.5 * p + 0.5);
        float r2 = dot(p, p);
        float k = max(1.15 / r2, 0.95);
        p *= k; scale *= k;
        trap = min(trap, abs(p));
    }
    return vec4(trap, (length(p.xz) - 0.35) / scale);
}

vec4 sdApollonian(vec3 pos) {
    vec3 p = mod(pos + 4.0, 8.0) - 4.0;
    float s = 1.0;
    vec3 trap = vec3(1e10);
    for (int i = 0; i < 8; i++) {
        p = -1.0 + 2.0 * fract(0.5 * p + 0.5);
        float r2 = dot(p, p);
        float k = 1.25 / r2;
        p *= k; s *= k;
        trap = min(trap, abs(p));
    }
    return vec4(trap, (length(p) - 0.9) / s);
}

vec4 sdHybrid(vec3 pos) {
    vec4 box = sdGiantMandelbox(pos);
    vec4 sponge = sdGiantMengerSponge(pos);
    return vec4(min(box.xyz, sponge.xyz), max(box.w, -sponge.w * 0.85));
}

vec4 sdQuaternionJulia(vec3 pos) {
    vec4 z = vec4(mod(pos + 3.0, 6.0) - 3.0, 0.0);
    vec4 c = vec4(-0.2, 0.6, 0.2, 0.8);
    float md2 = 1.0, mz2 = dot(z, z);
    vec3 trap = vec3(1e10);
    for (int i = 0; i < 9; i++) {
        md2 *= 4.0 * mz2;
        z = vec4(z.x*z.x - dot(z.yzw, z.yzw), 2.0*z.x*z.yzw) + c;
        mz2 = dot(z, z);
        trap = min(trap, abs(z.xyz));
        if (mz2 > 4.0) break;
    }
    return vec4(trap, 0.25 * sqrt(mz2 / md2) * log(mz2));
}

vec4 sdPseudoKleinian(vec3 pos) {
    vec3 p = mod(pos + 5.0, 10.0) - 5.0;
    float size = 1.0;
    vec3 trap = vec3(1e10);
    for (int i = 0; i < 7; i++) {
        p = 2.0 * clamp(p, -vec3(0.9, 1.2, 0.9), vec3(0.9, 1.2, 0.9)) - p;
        float r2 = dot(p, p);
        float k = max(1.0 / r2, 0.6);
        p *= k; size *= k;
        trap = min(trap, abs(p));
    }
    return vec4(trap, (length(p.xz) - 0.2) / size);
}

vec4 sdBoxBulbHybrid(vec3 pos) {
    return mix(sdGiantMandelbox(pos), sdMandelbulb(pos), 0.5);
}

vec4 sdSierpinskiTetrahedron(vec3 pos) {
    vec3 p = mod(pos + 4.0, 8.0) - 4.0;
    float scale = 1.0;
    vec3 trap = vec3(1e10);
    for (int i = 0; i < 7; i++) {
        if (p.x + p.y < 0.0) p.xy = -p.yx;
        if (p.x + p.z < 0.0) p.xz = -p.zx;
        if (p.y + p.z < 0.0) p.yz = -p.zy;
        p = p * 2.0 - vec3(1.0);
        scale *= 2.0;
        trap = min(trap, abs(p));
    }
    return vec4(trap, (length(p) - 1.0) / scale);
}

// Temel haritalama fonksiyonu
vec4 baseMap(vec3 p) {
    if (iFractalType == 0) return sdGiantMandelbox(p);
    if (iFractalType == 1) return sdGiantMengerSponge(p);
    if (iFractalType == 2) return sdMandelbulb(p);
    if (iFractalType == 3) return sdKleinianJulia(p);
    if (iFractalType == 4) return sdApollonian(p);
    if (iFractalType == 5) return sdHybrid(p);
    if (iFractalType == 6) return sdQuaternionJulia(p);
    if (iFractalType == 7) return sdPseudoKleinian(p);
    if (iFractalType == 8) return sdBoxBulbHybrid(p);
    return sdSierpinskiTetrahedron(p);
}

// 1 tuşu ile aktifleşen, fraktalları çekerek ve kopararak yok eden küre deformasyonlu map
vec4 map(vec3 p) {
    vec4 res = baseMap(p);
    
    if (iForceSphereActive) {
        float sphereDist = length(p - iForceSpherePos);
        float radius = 2.5; // Kuvvet küresinin etki yarıçapı
        
        if (sphereDist < radius) {
            // Yavaşça yerinden koparma, çekme ve delik açarak yok etme etkisi
            float pullFactor = smoothstep(radius, 0.0, sphereDist);
            vec3 dirToCenter = normalize(iForceSpherePos - p);
            
            // Uzayı küre merkezine doğru bük ve mesafeyi açarak o bölgeyi yok et (delik aç)
            vec3 deformedP = p + dirToCenter * pullFactor * 1.2;
            res = baseMap(deformedP);
            
            // Küre içindeki mesafeyi artırarak o kısımları erit/yok et
            res.w += pullFactor * 0.8; 
        }
    }
    
    return res;
}

vec3 getNormal(vec3 p) {
    float eps = 0.002;
    vec3 n = normalize(vec3(
        map(vec3(p.x + eps, p.y, p.z)).w - map(vec3(p.x - eps, p.y, p.z)).w,
        map(vec3(p.x, p.y + eps, p.z)).w - map(vec3(p.x, p.y - eps, p.z)).w,
        map(vec3(p.x, p.y, p.z + eps)).w - map(vec3(p.x, p.y, p.z - eps)).w
    ));
    float bump = hash(floor(p * 100.0)) * 0.06;
    return normalize(n + bump * vec3(0.1));
}

// Kaliteli Ambient Occlusion (Derinlik Gölgesi)[cite: 7]
float calculateAO(vec3 p, vec3 n) {
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < 5; i++) {
        float hr = 0.01 + 0.12 * float(i) / 4.0;
        vec3 aopos = n * hr + p;
        float dd = map(aopos).w;
        occ += (hr - dd) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
}

float getShadow(vec3 ro, vec3 rd) {
    if (iDarknessMode && !iFlashlightOn) return 0.0;
    float res = 1.0;
    float t = 0.08;
    for (int i = 0; i < 16; i++) {
        float h = map(ro + rd * t).w;
        res = min(res, 8.0 * h / t);
        t += clamp(h, 0.08, 1.5);
        if (h < 0.002 || t > 60.0) break;
    }
    return clamp(res, 0.1, 1.0);
}

vec3 ACESFilm(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec2 st = (gl_FragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;

    if (iGravLensEnabled && length(st) < 0.6) {
        st += st * (0.12 / (length(st) + 0.05));
    }

    float fovFactor = tan(radians(iFOV) * 0.5);
    vec3 rayDir = normalize(iCamDir + (st.x * iCamRight + st.y * iCamUp) * fovFactor);
    vec3 rayOrigin = iCamPos;

    if (iDOFEnabled) {
        float focusDist = 4.5;
        vec3 focusPoint = iCamPos + rayDir * focusDist;
        vec2 aperture = (vec2(hash(vec3(st, iTime)), hash(vec3(st.yx, iTime + 1.0))) - 0.5) * 0.07;
        rayOrigin += iCamRight * aperture.x + iCamUp * aperture.y;
        rayDir = normalize(focusPoint - rayOrigin);
    }

    float t = 0.0;
    float maxT = 70000.0;
    vec3 p = rayOrigin;
    vec4 res = vec4(0.0);
    bool hit = false;
    float steps = 0.0;

    for (int i = 0; i < 280; i++) { // Yüksek kalite ve akıcılık dengesi[cite: 7]
        p = rayOrigin + rayDir * t;
        res = map(p);
        steps += 1.0;

        if (res.w < 0.0015) { hit = true; break; }
        t += max(res.w * 0.75, 0.0015);
        if (t > maxT) break;
    }

    vec3 color = vec3(0.0);

    if (hit) {
        vec3 norm = getNormal(p);

        // Sinematik Renk Paleti (Bronz, Altın, Turkuaz)[cite: 7]
        vec3 matBronz = vec3(0.92, 0.58, 0.28);
        vec3 matAltin = vec3(0.85, 0.68, 0.35);
        vec3 matTurkuaz = vec3(0.20, 0.55, 0.65);

        float matFactor = clamp(res.x * 0.6, 0.0, 1.0);
        vec3 matColor = mix(matBronz, matAltin, matFactor);
        matColor = mix(matColor, matTurkuaz, clamp(res.y * 0.35, 0.0, 1.0));

        float ao = calculateAO(p, norm);
        vec3 totalDiffuse = vec3(0.0);

        // Sadece Fener ve Küre'nin kendi parıltı efekti (Ana ışık kaynağı kaldırıldı)
        if (iFlashlightOn) {
            vec3 flashLightDir = normalize(iCamPos - p);
            float flashDist = length(iCamPos - p);
            float attenuation = 1.0 / (1.0 + 0.03 * flashDist + 0.003 * flashDist * flashDist);
            
            float spotEffect = dot(-flashLightDir, iCamDir);
            if (spotEffect > 0.70) {
                float spotIntensity = pow(spotEffect, 8.0);
                float flashDiff = max(dot(norm, flashLightDir), 0.0);
                totalDiffuse += matColor * vec3(1.0, 0.98, 0.90) * flashDiff * attenuation * spotIntensity * 7.0 * iLightIntensity;
            }
        }

        // Kuvvet küresinin etrafındaki enerji halkası efekti
        if (iForceSphereActive) {
            float distToSphere = length(p - iForceSpherePos);
            if (distToSphere < 2.5) {
                float ring = sin(distToSphere * 10.0 - iTime * 5.0);
                totalDiffuse += vec3(0.2, 0.8, 1.0) * max(0.0, ring) * 3.0;
            }
        }

        vec3 ambient = iDarknessMode ? vec3(0.0) : matColor * 0.25 * iLightIntensity;
        color = (ambient + totalDiffuse) * ao;

        float glowStep = steps / 280.0;
        color += matColor * pow(glowStep, 2.2) * iBloomIntensity * 2.0 * iLightIntensity;

    } else {
        color = vec3(0.0);
    }

    color = ACESFilm(color);

    // Kromatik Sapma & Vignette (Sinematik Lens Efektleri)[cite: 7]
    float distUV = length(st);
    color.r += distUV * 0.025;
    color.b -= distUV * 0.025;
    color *= smoothstep(1.2, 0.4, distUV);

    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    color = mix(vec3(luminance), color, 1.45);

    color = pow(color, vec3(1.0 / 2.2));
    fragColor = vec4(color, 1.0);
}
"""

class HighQualityStudio:
    def __init__(self, width=1280, height=720):
        pygame.init()
        self.width, self.height = width, height
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

        self.screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("3D Master Studio (Force Sphere Destructive Mode)")

        self.ctx = moderngl.create_context()
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

        vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4')
        self.vao = self.ctx.simple_vertex_array(self.prog, self.ctx.buffer(vertices), 'in_position')

        self.cam_pos = np.array([0.0, 2.0, -6.0], dtype='f4')
        self.cam_vel = np.zeros(3, dtype='f4')
        self.yaw, self.pitch, self.fov = 90.0, 0.0, 70.0
        self.fractal_type = 4
        self.bloom_intensity = 0.35
        self.flashlight_on = True
        self.darkness_mode = False
        self.dof_enabled = False
        self.grav_lens_enabled = False
        self.light_intensity = 1.0
        
        # Kuvvet Küresi Değişkenleri
        self.force_sphere_active = False
        self.force_sphere_pos = np.array([0.0, 2.0, 0.0], dtype='f4')

        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()

    def get_cam_vectors(self):
        rad_yaw, rad_pitch = np.radians(self.yaw), np.radians(self.pitch)
        front = np.array([np.cos(rad_yaw) * np.cos(rad_pitch), np.sin(rad_pitch), np.sin(rad_yaw) * np.cos(rad_pitch)], dtype='f4')
        cam_dir = front / np.linalg.norm(front)
        cam_right = np.cross(cam_dir, np.array([0.0, 1.0, 0.0], dtype='f4'))
        cam_right /= np.linalg.norm(cam_right)
        cam_up = np.cross(cam_right, cam_dir)
        return cam_dir, cam_up / np.linalg.norm(cam_up), cam_right

    def capture_12k_screenshot(self):
        w, h = 11520, 6480
        print(f"\n[12K ÇEKİM] Yüksek Kaliteli Fotoğraf İşleniyor ({w}x{h})...")[cite: 7]
        fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture((w, h), 3)])
        fbo.use()
        cam_dir, cam_up, cam_right = self.get_cam_vectors()
        
        self.prog['iResolution'].value = (float(w), float(h))
        self.prog['iCamPos'].value = tuple(self.cam_pos)
        self.prog['iCamDir'].value = tuple(cam_dir)
        self.prog['iCamUp'].value = tuple(cam_up)
        self.prog['iCamRight'].value = tuple(cam_right)
        self.prog['iTime'].value = float((pygame.time.get_ticks() - self.start_time) / 1000.0)
        self.prog['iFOV'].value = float(self.fov)
        self.prog['iFractalType'].value = self.fractal_type
        self.prog['iBloomIntensity'].value = float(self.bloom_intensity)
        self.prog['iFlashlightOn'].value = self.flashlight_on
        self.prog['iDarknessMode'].value = self.darkness_mode
        self.prog['iDOFEnabled'].value = self.dof_enabled
        self.prog['iGravLensEnabled'].value = self.grav_lens_enabled
        self.prog['iLightIntensity'].value = float(self.light_intensity)
        self.prog['iForceSphereActive'].value = self.force_sphere_active
        self.prog['iForceSpherePos'].value = tuple(self.force_sphere_pos)

        self.ctx.clear(0.0, 0.0, 0.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)
        img = Image.frombytes('RGB', (w, h), fbo.read(components=3), 'raw', 'RGB', 0, -1)
        filename = f"fractal_12k_{int(time.time())}.png"
        img.save(filename)
        self.ctx.screen.use()
        print(f"[BAŞARILI] 12K Kaydedildi: {filename}\n")

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEMOTION:
                dx, dy = event.rel
                self.yaw += dx * 0.15
                self.pitch = max(-89.0, min(89.0, self.pitch - dy * 0.15))
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    self.light_intensity = min(3.0, self.light_intensity + 0.15)
                if event.button == 5:
                    self.light_intensity = max(0.0, self.light_intensity - 0.15)

            if event.type == pygame.KEYDOWN:
                # 1 TUŞU İLE KUVVET KÜRESİNİ AÇ / KAPA (Kürenin konumu kameranın hemen önü olarak ayarlanır)
                if event.key == pygame.K_1:
                    self.force_sphere_active = not self.force_sphere_active
                    cam_dir, _, _ = self.get_cam_vectors()
                    self.force_sphere_pos = self.cam_pos + cam_dir * 3.5
                    print(f"Kuvvet Küresi (Yıkım Etkisi): {'AKTİF' if self.force_sphere_active else 'KAPALI'}")

                if pygame.K_F2 <= event.key <= pygame.K_F10:
                    self.fractal_type = event.key - pygame.K_F1
                    print(f"Aktif Fraktal ID: {self.fractal_type + 1}")
                if event.key == pygame.K_f: self.flashlight_on = not self.flashlight_on
                if event.key == pygame.K_y: self.darkness_mode = not self.darkness_mode
                if event.key == pygame.K_g: self.grav_lens_enabled = not self.grav_lens_enabled
                if event.key == pygame.K_o: self.dof_enabled = not self.dof_enabled
                if event.key == pygame.K_p: self.capture_12k_screenshot()

        keys = pygame.key.get_pressed()
        cam_dir, _, cam_right = self.get_cam_vectors()
        accel = np.zeros(3, dtype='f4')
        speed = 0.05 if not keys[pygame.K_LSHIFT] else 0.25

        if keys[pygame.K_w]: accel += cam_dir
        if keys[pygame.K_s]: accel -= cam_dir
        if keys[pygame.K_a]: accel -= cam_right
        if keys[pygame.K_d]: accel += cam_right
        if keys[pygame.K_e]: accel[1] += 1.0
        if keys[pygame.K_q]: accel[1] -= 1.0

        if np.linalg.norm(accel) > 0:
            accel = accel / np.linalg.norm(accel) * speed
        self.cam_vel = self.cam_vel * 0.85 + accel
        self.cam_pos += self.cam_vel

        # Kuvvet küresi aktifse kamerayla birlikte yavaşça hareket ettirebilir veya sabit tutabilirsin
        if self.force_sphere_active:
            # Küre yavaşça zamanla salınım yapsın veya kamerayı takip etsin
            pass

    def run(self):
        while True:
            self.handle_input()
            t_sec = (pygame.time.get_ticks() - self.start_time) / 1000.0
            cam_dir, cam_up, cam_right = self.get_cam_vectors()

            self.prog['iResolution'].value = (float(self.width), float(self.height))
            self.prog['iCamPos'].value = tuple(self.cam_pos)
            self.prog['iCamDir'].value = tuple(cam_dir)
            self.prog['iCamUp'].value = tuple(cam_up)
            self.prog['iCamRight'].value = tuple(cam_right)
            self.prog['iTime'].value = float(t_sec)
            self.prog['iFOV'].value = float(self.fov)
            self.prog['iFractalType'].value = self.fractal_type
            self.prog['iBloomIntensity'].value = float(self.bloom_intensity)
            self.prog['iFlashlightOn'].value = self.flashlight_on
            self.prog['iDarknessMode'].value = self.darkness_mode
            self.prog['iDOFEnabled'].value = self.dof_enabled
            self.prog['iGravLensEnabled'].value = self.grav_lens_enabled
            self.prog['iLightIntensity'].value = float(self.light_intensity)
            self.prog['iForceSphereActive'].value = self.force_sphere_active
            self.prog['iForceSpherePos'].value = tuple(self.force_sphere_pos)

            self.ctx.clear(0.0, 0.0, 0.0)
            self.vao.render(moderngl.TRIANGLE_STRIP)
            pygame.display.flip()
            self.clock.tick(60)

if __name__ == "__main__":
    app = HighQualityStudio()
    app.run()