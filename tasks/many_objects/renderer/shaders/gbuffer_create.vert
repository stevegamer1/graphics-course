#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "unpack_attributes.glsl"


layout(location = 0) in vec4 vPosNorm;
layout(location = 1) in vec4 vTexCoordAndTang;

layout(push_constant) uniform params_t
{
  mat4 mProjView;
} params;

layout(binding = 0) readonly buffer DrawParams
{
  mat4 mModel[];
} drawParams;

layout(binding = 1) readonly buffer DrawParamsIndices
{
  uint indices[];
} drawParamsIndices;

layout (location = 0 ) out VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
  vec4 wTangent;
  vec2 texCoord;
} vOut;

out gl_PerVertex { vec4 gl_Position; };

void main(void)
{
  const vec4 wNorm = vec4(decode_normal(floatBitsToInt(vPosNorm.w)),     0.0f);
  const vec4 wTang = vec4(decode_normal(floatBitsToInt(vTexCoordAndTang.z)), 0.0f);

  const mat4 mModel = drawParams.mModel[drawParamsIndices.indices[gl_InstanceIndex]];

  vOut.wPos   = (mModel * vec4(vPosNorm.xyz, 1.0f)).xyz;
  vOut.wNorm  = normalize(mat3(transpose(inverse(mModel))) * wNorm.xyz);
  vOut.wTangent = vec4(normalize(mat3(transpose(inverse(mModel))) * wTang.xyz), wTang.w);
  vOut.texCoord = vTexCoordAndTang.xy;

  gl_Position   = params.mProjView * vec4(vOut.wPos, 1.0);
}
