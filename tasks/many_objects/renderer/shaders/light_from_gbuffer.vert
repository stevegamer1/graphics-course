#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "unpack_attributes.glsl"


layout(location = 0) in vec4 vPos;

layout(push_constant) uniform params_t
{
  mat4 mProjView;
  vec4 wCamPos;
  uvec2 resolution;
} params;

const uint LIGHT_TYPE_POINT = 0;
const uint LIGHT_TYPE_DIRECTIONAL = 1;
const uint LIGHT_TYPE_AMBIENT = 2;

struct Light {
  vec3 center;
  float intensity;
  vec3 color;
  uint type;
};

layout(binding = 0, std140) readonly buffer Lights
{
  Light values[];
} lights;

layout (location = 0 ) out VS_OUT
{
  flat uint instanceIndex;
} vOut;

out gl_PerVertex { vec4 gl_Position; };


const float LIGHT_INVISIBLE_THRESHOLD = 1.0f / 256.0f;  // Highest invisible light intensity.

float pointLightRadius(float intensity) {
  // We know: newIntensity(distance) = intensity / (distance * distance);
  // Light is invisible as soon as newIntensity(distance) < LIGHT_INVISIBLE_THRESHOLD,
  // Equivalent to intensity / (distance * distance) < LIGHT_INVISIBLE_THRESHOLD,
  // Equivalent to distance > sqrt(intensity / LIGHT_INVISIBLE_THRESHOLD);
  return sqrt(intensity / LIGHT_INVISIBLE_THRESHOLD);
}

void main(void)
{
  const uint lightType = lights.values[gl_InstanceIndex].type;
  vec3 center;
  float radius;
  if (lightType == LIGHT_TYPE_POINT) {
    center = lights.values[gl_InstanceIndex].center;
    radius = pointLightRadius(lights.values[gl_InstanceIndex].intensity);
  } else if (lightType == LIGHT_TYPE_DIRECTIONAL || lightType == LIGHT_TYPE_AMBIENT) {
    center = params.wCamPos.xyz;
    radius = 99.0f;
  }

  vOut.instanceIndex = gl_InstanceIndex;

  gl_Position = params.mProjView * vec4(vPos.xyz * radius + center, 1.0f);
}
