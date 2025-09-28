#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require


layout(location = 0) out vec4 out_albedo;
layout(location = 1) out vec2 out_normal;

layout(location = 0) in VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
  vec3 wTangent;
  vec2 texCoord;
} surf;

const uint FLAG_ZSIGN = (uint(1) << 0);

float flagsToAlbedoAlpha(uint flags) {
  return float(flags) / 255.0f;
}

void main()
{
  const vec3 surfaceColor = vec3(1.0f, 1.0f, 1.0f);
  const vec3 wNormal = normalize(surf.wNorm);
  uint flags = 0;
  flags |= wNormal.z > 0.0f ? FLAG_ZSIGN : 0;

  out_albedo.rgb = surfaceColor;
  out_albedo.a = flagsToAlbedoAlpha(flags);

  out_normal.rg = wNormal.xy;
}
