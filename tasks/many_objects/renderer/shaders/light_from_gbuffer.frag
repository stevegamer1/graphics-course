#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require


layout(location = 0) out vec4 out_color;

layout(push_constant) uniform params_t
{
  mat4 mProjView;
  vec4 wCamPos;
  uvec2 resolution;
  float camNear;
} params;

const uint LIGHT_TYPE_POINT = 0;
const uint LIGHT_TYPE_DIRECTIONAL = 1;
const uint LIGHT_TYPE_AMBIENT = 2;

struct Light {
  vec3 center;
  float intensity;
  vec3 color;
  uint padding23_castsShadows1_type8;
};

const uint LIGHT_CASTS_SHADOW_MASK = 0x00000100;
const uint LIGHT_TYPE_MASK         = 0x000000FF;

layout(binding = 0, std140) readonly buffer Lights
{
  Light values[];
} lights;

layout (location = 0 ) in VS_OUT
{
  flat uint instanceIndex;
} vOut;

layout(binding = 1) uniform sampler2D gBufferAlbedo;
layout(binding = 2) uniform sampler2D gBufferMetallicRoughness;
layout(binding = 3) uniform sampler2D gBufferNormal;
layout(binding = 4) uniform sampler2D gBufferDepth;
layout(binding = 5) readonly buffer MLightViewProj {
  mat4 mats[];
} mLightViewProj;

// In the future, I imagine each light having uint firstShadowMap and shadowMapCount, and they are allocated in this bindless array of textures.
// Currently it's actually just one cascaded shadow map.
// Actually, currently it's actually not cascaded yet.
layout(set = 1, binding = 0) uniform sampler2D shadowMap[];

const uint FLAG_ZSIGN = (uint(1) << 0);

uint getFlags(float albedoAlpha) {
  return uint(albedoAlpha * 255.0f + 0.5f);
}

float getZSign(uint flags) {
  return ((flags & FLAG_ZSIGN) != 0) ? 1.0f : -1.0f;
}

vec3 getNormal(vec2 wNormalXY, float zSign) {
  vec3 result = vec3(0.0f);
  result.xy = wNormalXY;
  float zSqr = 1.0f - result.x * result.x - result.y * result.y;
  result.z = sqrt(max(zSqr, 0.0f)) * zSign;
  return normalize(result);
}

const float pi = 3.14159265359f;
const float gamma = 2.2f;

vec3 fresnel ( in vec3 f0, in float product )
{
    product = clamp ( product, 0.0, 1.0 );      // saturate
    
    return mix ( f0, vec3 (1.0), pow(1.0 - product, 5.0) );
}

float D_GGX ( in float roughness, in float NdH )
{
    float m  = roughness * roughness;
    float m2 = m * m;
    float NdH2 = NdH * NdH;
    float d  = (m2 - 1.0) * NdH2 + 1.0;
    
    return m2 / (pi * d * d);
}

float G_neumann ( in float nl, in float nv )
{
    return nl * nv / max ( nl, nv );
}

vec3 cookTorrance ( in float nl, in float nv, in float nh, in float vh, in vec3 f0, in float roughness )
{
    float D = D_GGX     ( roughness, nh );
    float G = G_neumann ( nl, nv );

    return f0 * D * G;
}

vec3 getReflectedLightPBR(
  vec3 lightVector,
  vec3 normal,
  vec3 lookVector,
  vec3 lightColorWithIntensity,
  float metalness,
  float roughness,
  vec3 baseColor
) {
  vec3 v = normalize(lookVector);
  vec3 l = normalize(lightVector);
  vec3 n = normalize(normal);
  vec3 h = normalize(v + l);

  float nl = max(dot(n, l), 0.0f);
  float nv = max(dot(n, v), 0.0f);
  float nh = max(dot(n, h), 0.0f);
  float vh = max(dot(v, h), 0.0f);

  const float FDiel = 0.04;

  vec3 F0          = mix ( vec3(FDiel), baseColor, metalness );
  vec3 specFresnel = fresnel ( F0, nv );
  vec3 spec        = cookTorrance ( nl, nv, nh, vh, specFresnel, roughness ) * nl / max ( 0.001, 4.0 * nl * nv );
  vec3 diff        = (vec3(1.0) - specFresnel) * nl / pi;

  return pow ( ( diff * mix ( baseColor, vec3(0.0), metalness) + spec ) * lightColorWithIntensity, vec3 ( 1.0 / gamma ) );
}

uint getShadowMapCascadeIndex(float wCamFragDist, float camNear) {
  // offsetForThisParticularLight + 
  // return min(uint(floor(log2(wCamFragDist / camNear))), MAX_IMAGE_IN_CASCADE);
  return 0;
}

bool isFragmentInShadow(vec3 wFragPos, float wCamFragDist, float camNear) {
  uint shadowMapIndex = getShadowMapCascadeIndex(wCamFragDist, camNear);

  vec3 lFragPos = (mLightViewProj.mats[shadowMapIndex] * vec4(wFragPos, 1.0f)).xyz;
  vec2 lFragUV = lFragPos.xy * 0.5f + vec2(0.5f);
  return texture(shadowMap[shadowMapIndex], lFragUV).r < lFragPos.z;
}

void main()
{
  vec2 uv = gl_FragCoord.xy / params.resolution;

  const uint flags = getFlags(texture(gBufferAlbedo, uv).a);
  const vec3 albedo = texture(gBufferAlbedo, uv).rgb;
  const vec3 wNormal = getNormal(texture(gBufferNormal, uv).xy, getZSign(flags));
  const float depth = texture(gBufferDepth, uv).r;

  mat4 mInvProjView = inverse(params.mProjView);
  vec4 lookVectorStart = mInvProjView * vec4(uv.xy * 2.0f - vec2(1.0f), 0.0f, 1.0f);
  vec4 lookVectorEnd = mInvProjView * vec4(uv.xy * 2.0f - vec2(1.0f), depth, 1.0f);
  lookVectorStart /= lookVectorStart.w;
  lookVectorEnd /= lookVectorEnd.w;
  vec3 lookVector = lookVectorEnd.xyz - lookVectorStart.xyz;
  vec3 wCamPos = params.wCamPos.xyz;

  const vec3 lightColor = lights.values[vOut.instanceIndex].color;
  const uint lightType = lights.values[vOut.instanceIndex].padding23_castsShadows1_type8 & LIGHT_TYPE_MASK;
  const bool castsShadows = (lights.values[vOut.instanceIndex].padding23_castsShadows1_type8 & LIGHT_CASTS_SHADOW_MASK) != 0;
  const vec3 wFragPos = wCamPos + lookVector;

  const vec3 wLightPos = lights.values[vOut.instanceIndex].center;
  const float lightIntensity = lights.values[vOut.instanceIndex].intensity;

  vec3 reflectedLight;

  if (castsShadows && isFragmentInShadow(wFragPos, length(lookVector), params.camNear)) {
    reflectedLight = vec3(0);
  } else if (lightType == LIGHT_TYPE_POINT) {
    const vec3 lightVector = wLightPos - wFragPos;
    const float lightDistance = length(lightVector);
    const float fadeFactor = 1.0f / (lightDistance * lightDistance);

    vec3 lightColorWithIntensity = lightColor * lightIntensity * fadeFactor;
    float metalness = texture(gBufferMetallicRoughness, uv).g;
    float roughness = texture(gBufferMetallicRoughness, uv).b;
    reflectedLight = getReflectedLightPBR(
      lightVector,
      wNormal,
      -lookVector,
      lightColorWithIntensity,
      metalness,
      roughness,
      albedo
    );
  } else if (lightType == LIGHT_TYPE_DIRECTIONAL) {
    const vec3 lightVector = wLightPos;

    vec3 lightColorWithIntensity = lightColor * lightIntensity;
    float roughness = texture(gBufferMetallicRoughness, uv).g;
    float metalness = texture(gBufferMetallicRoughness, uv).b;
    reflectedLight = getReflectedLightPBR(
      lightVector,
      wNormal,
      -lookVector,
      lightColorWithIntensity,
      metalness,
      roughness,
      albedo
    );
  } else if (lightType == LIGHT_TYPE_AMBIENT) {
    reflectedLight = albedo * lightIntensity * lightColor;
  }

  out_color.rgb = reflectedLight;
  out_color.a = 1.0f;
}
