#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(location = 0) in vec4 vPos;

layout(push_constant) uniform params_t
{
  mat4 mProjView;
} params;

layout(binding = 0) readonly buffer DrawParams
{
  mat4 mModel[];
} drawParams;

// Will need when I add indirection to save space.
// layout(binding = 1) readonly buffer DrawParamsIndices
// {
//   uint indices[];
// } drawParamsIndices;


out gl_PerVertex { vec4 gl_Position; };

void main(void)
{
//   const mat4 mModel = drawParams.mModel[drawParamsIndices.indices[gl_InstanceIndex]];
  const mat4 mModel = drawParams.mModel[gl_InstanceIndex];  // TODO: take less space and hence add indirection
  gl_Position = params.mProjView * mModel * vec4(vPos.xyz, 1.0f);
}
