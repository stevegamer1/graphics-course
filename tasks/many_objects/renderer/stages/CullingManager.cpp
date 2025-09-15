#include "CullingManager.hpp"
#include "SynchronizedBuffer.hpp"
#include <cmath>
#include <glm/ext/scalar_constants.hpp>
#include <glm/fwd.hpp>
#include <vulkan/vulkan_enums.hpp>
#include "etna/Etna.hpp"
#include "etna/PipelineManager.hpp"


void CullingManager::loadShader() {
    etna::create_program(
        PROGRAM_NAME, 
        {MANY_OBJECTS_RENDERER_SHADERS_ROOT "cull.comp.spv"}
    );
}

void CullingManager::createPipeline() {
    auto& pipelineManager = etna::get_context().getPipelineManager();
    pipeline = pipelineManager.createComputePipeline(PROGRAM_NAME, {});
}

void CullingManager::zeroOutInstanceCountsInCommands(
  vk::CommandBuffer cmd_buf, SynchronizedBuffer& indirect_commands) {
  indirect_commands.syncBeforeUsage(BufferSyncUsage{
    .stageFlags = vk::PipelineStageFlagBits2::eHost,
    .accessFlags = vk::AccessFlagBits2::eHostWrite
  }, cmd_buf);

  indirect_commands.buffer.map();
  auto commands = reinterpret_cast<vk::DrawIndexedIndirectCommand*>(indirect_commands.buffer.data());
  size_t count = indirect_commands.current_size / sizeof(vk::DrawIndexedIndirectCommand);
  for (size_t i = 0; i < count; ++i) {
    commands[i].instanceCount = 0;
  }
  indirect_commands.buffer.unmap();
}

CullingManager::Frustum CullingManager::getFrustum(float fov, float near, float far, float aspect) {
    fov = fov / 180.0f * glm::pi<float>();

    Frustum result;
    float vfov = atan(tan(fov) / aspect);  // vertical fov

    result.near = glm::vec4(0, 0, 1, near);
    result.far = glm::vec4(0, 0, -1, -far);
    result.left = glm::vec4(-cos(fov), 0, sin(fov), 0);
    result.right = glm::vec4(cos(fov), 0, sin(fov), 0);
    result.top = glm::vec4(0, -cos(vfov), sin(vfov), 0);
    result.bottom = glm::vec4(0, cos(vfov), sin(vfov), 0);

    return result;
}

void CullingManager::run(vk::CommandBuffer cmd_buf, SynchronizedBuffer& draw_params, SynchronizedBuffer& aabbs,
             SynchronizedBuffer& indirect_commands, SynchronizedBuffer& draw_params_indices,
             SynchronizedBuffer& command_indices, uint32_t instance_count, const Frustum& camera_frustum) {    
    zeroOutInstanceCountsInCommands(cmd_buf, indirect_commands);

    draw_params.syncBeforeUsage(BufferSyncUsage{
        .stageFlags = vk::PipelineStageFlagBits2::eComputeShader,
        .accessFlags = vk::AccessFlagBits2::eShaderRead
    }, cmd_buf);

    aabbs.syncBeforeUsage(BufferSyncUsage{
        .stageFlags = vk::PipelineStageFlagBits2::eComputeShader,
        .accessFlags = vk::AccessFlagBits2::eShaderRead
    }, cmd_buf);

    indirect_commands.syncBeforeUsage(BufferSyncUsage{
        .stageFlags = vk::PipelineStageFlagBits2::eComputeShader,
        .accessFlags = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite
    }, cmd_buf);

    draw_params_indices.syncBeforeUsage(BufferSyncUsage{
        .stageFlags = vk::PipelineStageFlagBits2::eComputeShader,
        .accessFlags = vk::AccessFlagBits2::eShaderWrite
    }, cmd_buf);

    command_indices.syncBeforeUsage(BufferSyncUsage{
        .stageFlags = vk::PipelineStageFlagBits2::eComputeShader,
        .accessFlags = vk::AccessFlagBits2::eShaderRead
    }, cmd_buf);



    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getVkPipeline());

    auto programInfo = etna::get_shader_program(PROGRAM_NAME);

    auto set = etna::create_descriptor_set(
        programInfo.getDescriptorLayoutId(0),
        cmd_buf,
        {
            etna::Binding{0, draw_params.buffer.genBinding()},
            etna::Binding{1, aabbs.buffer.genBinding()},
            etna::Binding{2, indirect_commands.buffer.genBinding()},
            etna::Binding{3, draw_params_indices.buffer.genBinding()},
            etna::Binding{4, command_indices.buffer.genBinding()},
        }
    );

    cmd_buf.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        pipeline.getVkPipelineLayout(),
        0,
        {set.getVkSet()},
        {}
    );

    PushConstants pushConsts{
        .camera_frustum = camera_frustum,
        .instances_to_cull_count = instance_count
    };
    cmd_buf.pushConstants<PushConstants>(
        pipeline.getVkPipelineLayout(),
        vk::ShaderStageFlagBits::eCompute,
        0,
        {pushConsts}
    );

    etna::flush_barriers(cmd_buf);

    size_t groupSizeX = 16;
    uint32_t groupCountX = uint32_t((instance_count + groupSizeX - 1) / groupSizeX);  // Division with rounding up.
    cmd_buf.dispatch(groupCountX, 1, 1);
}
