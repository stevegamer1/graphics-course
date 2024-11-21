#version 430
#extension GL_GOOGLE_include_directive : require


#include "cpp_glsl_compat.h"


layout(binding = 0) uniform sampler2D iChannel0;
layout(binding = 1) uniform sampler2D fileTexture;

layout(location = 0) out vec4 out_fragColor;

layout(push_constant) uniform params_t
{
  uvec2 resolution;
} params;


const float iTime = 10.0f;

const float PI = 3.14159265359;

const float shape_tolerance = 1.0; // (0; inf) how smooth will shape transition be
const float color_mix_sharpness = 10.; // (1; inf) how sharp will color transition
const float trefoil_size = 6.0; // how big the whole thing is
const float radius = 0.5;


struct sphere {
    vec3 center;
    float radius;
    vec3 color;
};


const int n_spheres = 24;
sphere spheres[n_spheres] = sphere[](
    sphere(vec3(0, 0, 0), radius, vec3(1.0, 0.0, 0.0)),
    sphere(vec3(0, 0, 0), radius, vec3(0.0, 1.0, 0.0)),
    sphere(vec3(0, 0, 0), radius, vec3(0.0, 0.0, 1.0)),
    sphere(vec3(0, 0, 0), radius, vec3(1.0, 0.0, 0.0)),
    sphere(vec3(0, 0, 0), radius, vec3(0.0, 1.0, 0.0)),
    sphere(vec3(0, 0, 0), radius, vec3(0.0, 0.0, 1.0)),
    sphere(vec3(0, 0, 0), radius, vec3(1.0, 0.0, 0.0)),
    sphere(vec3(0, 0, 0), radius, vec3(0.0, 1.0, 0.0)),
    sphere(vec3(0, 0, 0), radius, vec3(0.0, 0.0, 1.0)),
    sphere(vec3(0, 0, 0), radius, vec3(1.0, 0.0, 0.0)),
    sphere(vec3(0, 0, 0), radius, vec3(0.0, 1.0, 0.0)),
    sphere(vec3(0, 0, 0), radius, vec3(0.0, 0.0, 1.0)),
    sphere(vec3(0, 0, 0), radius, vec3(1.0, 0.0, 0.0)),
    sphere(vec3(0, 0, 0), radius, vec3(0.0, 1.0, 0.0)),
    sphere(vec3(0, 0, 0), radius, vec3(0.0, 0.0, 1.0)),
    sphere(vec3(0, 0, 0), radius, vec3(1.0, 0.0, 0.0)),
    sphere(vec3(0, 0, 0), radius, vec3(0.0, 1.0, 0.0)),
    sphere(vec3(0, 0, 0), radius, vec3(0.0, 0.0, 1.0)),
    sphere(vec3(0, 0, 0), radius, vec3(1.0, 0.0, 0.0)),
    sphere(vec3(0, 0, 0), radius, vec3(0.0, 1.0, 0.0)),
    sphere(vec3(0, 0, 0), radius, vec3(0.0, 0.0, 1.0)),
    sphere(vec3(0, 0, 0), radius, vec3(1.0, 0.0, 0.0)),
    sphere(vec3(0, 0, 0), radius, vec3(0.0, 1.0, 0.0)),
    sphere(vec3(0, 0, 0), radius, vec3(0.0, 0.0, 1.0))
);

const bool metal_reflection = false;


struct light{
    vec3 position;
    float strength;
    vec3 color;
};

const int n_lights = 1;
light lights[n_lights] = light[](
    light(vec3(0, 0, -100.0), 10000.0, vec3(1, 1, 1))
);

vec3 grad(in vec3 point, in float delta);

float sphere_sdf(in vec3 point, in vec3 center, in float radius) {
    return length(point - center) - radius;
}

// Color weight field - determines how much of a shape's color should be used.
float sphere_cwf(in vec3 point, in vec3 center, in float radius) {
    return pow(color_mix_sharpness, radius - (length(point - center)));
}

float sdf(in vec3 point) {
    float sum = 0.0;
    for (int i = 0; i < n_spheres; ++i) {
        float dist = sphere_sdf(point, spheres[i].center, spheres[i].radius);
        sum += exp2(-dist / shape_tolerance);
    }
    return -shape_tolerance * log2(sum);
}

vec3 get_sphere_texture_color(in sphere sph, in vec3 point) {
    point -= sph.center + vec3(2.0);
    point /= 20.0;
    vec3 normal = normalize(abs(point * 20.0 + vec3(2.0)));
    vec3 result_procedural = normal.x * texture(iChannel0, point.yz).rgb +
    normal.y * texture(iChannel0, point.xz).rgb +
    normal.z * texture(iChannel0, point.xy).rgb;
    vec3 result_file = normal.x * texture(fileTexture, point.yz).rgb +
    normal.y * texture(fileTexture, point.xz).rgb +
    normal.z * texture(fileTexture, point.xy).rgb;

    float mix_factor = 0.5f;
    vec3 result = mix_factor * result_procedural + (1.0f - mix_factor) * result_file;
    return result * 1.5;
}

vec3 base_color(in vec3 point) {
    vec3 color_num = vec3(0, 0, 0);
    vec3 color_denom = vec3(0, 0, 0);
    for (int i = 0; i < n_spheres; ++i) {
        float cwf = sphere_cwf(point, spheres[i].center, spheres[i].radius);
        color_num += get_sphere_texture_color(spheres[i], point) * cwf;
        color_denom += cwf;
    }
    return color_num / color_denom;
}

bool traceRay(in vec3 ray, in vec3 pos, out vec3 impact_point) {
    while (true) {
        float dist = abs(sdf(pos));
    
        if (dist < 0.01) {
            impact_point = pos;
            return true;
        }
        
        if (dist > 100.0) {
            return false;
        }
        
        pos += ray * dist;
    }
}

vec3 grad(in vec3 point, in float delta) {
    vec3 dx = vec3(delta, 0, 0);
    vec3 dy = vec3(0, delta, 0);
    vec3 dz = vec3(0, 0, delta);
    float y0 = sdf(point);
    return (vec3(sdf(point+dx), sdf(point+dy), sdf(point+dz)) - vec3(y0, y0, y0)) / delta;
}


vec3 diffuse(in vec3 point, in vec3 base_color) {
    vec3 normal = normalize(grad(point, 0.0001));
    vec3 result = vec3(0, 0, 0);
    
    for (int i = 0; i < n_lights; ++i) {
        vec3 lightray = lights[i].position - point;
        float intensity = lights[i].strength / pow(length(lightray), 2.0);
        result += max(dot(normal, normalize(lightray)), 0.0) * intensity * lights[i].color;
    }
    result *= base_color;
    return result;
}

vec3 specular(in vec3 eye, in vec3 point, in vec3 base_color) {
    vec3 normal = normalize(grad(point, 0.0001));
    vec3 eye_ray = normalize(eye - point);
    vec3 result = vec3(0, 0, 0);
    
    for (int i = 0; i < n_lights; ++i) {
        vec3 lightray_refl = -reflect(lights[i].position - point, normal);
        float intensity = lights[i].strength / pow(length(lightray_refl), 2.0);
        float powered = pow(max(dot(eye_ray, normalize(lightray_refl)), 0.0), 20.0);
        result += powered * intensity * lights[i].color;
    }
    
    if (metal_reflection) {
        result *= base_color;
    }
    return result;
}

void positionSpheres() {
    for (int i = 0; i < n_spheres; ++i) {
        //float t = iTime + PI * float(i) / float(n_spheres);
        float t = iTime + PI * float(i) / float(n_spheres);
        spheres[i].center = vec3(trefoil_size * cos(3.0 * t) * sin(t) + sin(2.0 * t) * 0.2,
                                 trefoil_size * cos(3.0 * t) * cos(t) + sin(t) * 0.2,
                                 36.0 + 30.0 * sin(t * 0.5 + PI / 4.));
    }
}

vec3 skybox_color(in vec3 direction) {
    // return texture(iChannel1, direction).rgb;
    return vec3(0.3, 0.05, 0.05);
}


void main() {
  positionSpheres();

  vec2 uv = (gl_FragCoord.xy - params.resolution.xy / 2.0) / min(params.resolution.x, params.resolution.y);

  vec3 camera = vec3(0, 0, -1);
  vec3 ray = normalize(vec3(uv, 0) - camera);
  vec3 surface_point;
  vec3 color;
  if (traceRay(ray, camera, surface_point)) {
      vec3 base = base_color(surface_point);
      vec3 normal = normalize(grad(surface_point, 0.0001));
      vec3 skyray = reflect(ray, normal);
      color = specular(camera, surface_point, base) + diffuse(surface_point, base);
  } else {
      color = skybox_color(ray);
  }
  
  out_fragColor = vec4(color, 1.0f);
}
