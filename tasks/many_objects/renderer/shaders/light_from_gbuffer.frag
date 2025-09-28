#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require


layout(location = 0) out vec4 out_color;

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

layout (location = 0 ) in VS_OUT
{
  flat uint instanceIndex;
} vOut;

layout(binding = 1) uniform sampler2D gBufferAlbedo;
layout(binding = 2) uniform sampler2D gBufferNormal;
layout(binding = 3) uniform sampler2D gBufferDepth;

void main()
{
  vec2 uv = gl_FragCoord.xy / params.resolution;

  const vec3 albedo = texture(gBufferAlbedo, uv).rgb;
  const vec3 wNormal = texture(gBufferNormal, uv).rgb;
  const float depth = texture(gBufferDepth, uv).r;

  mat4 mInvProjView = inverse(params.mProjView);
  vec4 lookVectorStart = mInvProjView * vec4(uv.xy * 2.0f - vec2(1.0f), 0.0f, 1.0f);
  vec4 lookVectorEnd = mInvProjView * vec4(uv.xy * 2.0f - vec2(1.0f), depth, 1.0f);
  lookVectorStart /= lookVectorStart.w;
  lookVectorEnd /= lookVectorEnd.w;
  vec3 lookVector = lookVectorEnd.xyz - lookVectorStart.xyz;
  vec3 wCamPos = params.wCamPos.xyz;

  

  const vec3 lightColor = lights.values[vOut.instanceIndex].color;
  const uint lightType = lights.values[vOut.instanceIndex].type;
  const vec3 wFragPos = wCamPos + lookVector;

  const vec3 wLightPos = lights.values[vOut.instanceIndex].center;
  const float lightIntensity = lights.values[vOut.instanceIndex].intensity;

  vec3 incidentLight;
  if (lightType == LIGHT_TYPE_POINT) {
    const vec3 lightVector = wLightPos - wFragPos;
    const float lightDistance = length(lightVector);
    const float fadeFactor = 1.0f / (lightDistance * lightDistance);
    incidentLight = max(dot(wNormal, normalize(lightVector)), 0.0f) * lightColor * lightIntensity * fadeFactor;
  } else if (lightType == LIGHT_TYPE_DIRECTIONAL) {
    const vec3 lightVector = wLightPos;
    incidentLight = max(dot(wNormal, normalize(lightVector)), 0.0f) * lightColor * lightIntensity;
  } else if (lightType == LIGHT_TYPE_AMBIENT) {
    incidentLight = lightIntensity * lightColor;
  }

  out_color.rgb = incidentLight * albedo;
  out_color.a = 1.0f;
}
