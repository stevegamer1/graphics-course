#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require


layout(location = 0) out vec4 out_albedo;
layout(location = 1) out vec3 out_metallicRoughness;
layout(location = 2) out vec2 out_normal;

layout(location = 0) in VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
  vec4 wTangent;
  vec2 texCoord;
  flat uint instanceIndex;
} surf;

struct PBRMaterial {
  uint albedoIndex;
  uint metallicRoughnessIndex;
  uint normalsIndex;
  vec4 albedoMultiplier;
  vec4 metallicRoughnessMultiplier;
  vec4 normalsMultiplier;
};

layout(set = 0, binding = 2) readonly buffer Materials {
  PBRMaterial mats[];
} materials;

layout(set = 0, binding = 3) readonly buffer InstanceToMaterialMap {
  uint indices[];
} instanceToMaterialMap;

layout(set = 1, binding = 0) uniform sampler2D pbrTextures[];

const uint FLAG_ZSIGN = (uint(1) << 0);

float flagsToAlbedoAlpha(uint flags) {
  return float(flags) / 255.0f;
}

void main()
{
  PBRMaterial mat = materials.mats[instanceToMaterialMap.indices[surf.instanceIndex]];

  vec3 surfaceColor = texture(pbrTextures[nonuniformEXT(mat.albedoIndex)], surf.texCoord).rgb * mat.albedoMultiplier.rgb;

  const vec3 wNormal = normalize(surf.wNorm);
  // const vec3 wTangent = normalize(surf.wTangent.xyz);  // This is the correct way after using MikkTSpace on CPU.
  // const float bitangentSign = surf.wTangent.w;
  //
  // But I am retarded and lazy, so I'll just slap good ol' derivatives here.
  float det = (dFdx(surf.texCoord.x) * dFdy(surf.texCoord.y) - dFdx(surf.texCoord.y) * dFdy(surf.texCoord.x));
  vec3 wTangent = (dFdx(surf.wPos) * dFdy(surf.texCoord.y) - dFdy(surf.wPos) * dFdx(surf.texCoord.y)) / det;
  wTangent = normalize(wTangent - wNormal * dot(wNormal, wTangent));  // It should work without this. But this makes the tangent sharper.
  const float bitangentSign = 1.0f;
  const vec3 wBitangent = cross(wNormal, wTangent) * bitangentSign;  // From GLTF specification.

  uint flags = 0;
  flags |= wNormal.z > 0.0f ? FLAG_ZSIGN : 0;

  out_albedo.rgb = surfaceColor;
  out_albedo.a = flagsToAlbedoAlpha(flags);

  vec3 sampledNormal = normalize(texture(pbrTextures[nonuniformEXT(mat.normalsIndex)], surf.texCoord).xyz * 2.0f - vec3(1.0f));
  sampledNormal.z = max(sampledNormal.z, 0.0f);
  out_normal.rg = (normalize(sampledNormal.x * wTangent + sampledNormal.y * wBitangent + sampledNormal.z * wNormal)).xy;

  out_metallicRoughness = texture(pbrTextures[nonuniformEXT(mat.metallicRoughnessIndex)], surf.texCoord).rgb * mat.metallicRoughnessMultiplier.rgb;
}
