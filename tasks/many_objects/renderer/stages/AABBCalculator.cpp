#include "AABBCalculator.hpp"
#include "SynchronizedBuffer.hpp"
#include <vulkan/vulkan_enums.hpp>
#include "etna/DescriptorSet.hpp"
#include "etna/Etna.hpp"
#include "etna/PipelineManager.hpp"
#include "scene/SceneManager.hpp"


void AABBCalculator::loadShader() {
    etna::create_program(
        PROGRAM_NAME, 
        {MANY_OBJECTS_RENDERER_SHADERS_ROOT "aabb_calculator.comp.spv"}
    );
}

void AABBCalculator::createPipeline() {
    auto& pipelineManager = etna::get_context().getPipelineManager();
    pipeline = pipelineManager.createComputePipeline(PROGRAM_NAME, {});
}

void AABBCalculator::run(vk::CommandBuffer cmd_buf,

    SynchronizedBuffer& aabbs,
    SynchronizedBuffer& indirect_commands,

    etna::Buffer& vertex_buffer,
    etna::Buffer& index_buffer,

    uint32_t relem_count) {
    aabbs.syncBeforeUsage(BufferSyncUsage{
        .stageFlags = vk::PipelineStageFlagBits2::eComputeShader,
        .accessFlags = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite
    }, cmd_buf);

    indirect_commands.syncBeforeUsage(BufferSyncUsage{
        .stageFlags = vk::PipelineStageFlagBits2::eComputeShader,
        .accessFlags = vk::AccessFlagBits2::eShaderRead
    }, cmd_buf);

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getVkPipeline());

    auto programInfo = etna::get_shader_program(PROGRAM_NAME);

    auto set = etna::create_descriptor_set(
        programInfo.getDescriptorLayoutId(0),
        cmd_buf,
        {
            etna::Binding{0, aabbs.buffer.genBinding()},
            etna::Binding{1, indirect_commands.buffer.genBinding()},
            etna::Binding{2, vertex_buffer.genBinding()},
            etna::Binding{3, index_buffer.genBinding()}
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
        .abbs_count = relem_count,
        .vertex_size_in_bytes = sizeof(SceneManager::Vertex)
    };
    cmd_buf.pushConstants<PushConstants>(
        pipeline.getVkPipelineLayout(),
        vk::ShaderStageFlagBits::eCompute,
        0,
        {pushConsts}
    );

    etna::flush_barriers(cmd_buf);

    size_t groupSizeX = 16;
    uint32_t groupCountX = uint32_t((relem_count + groupSizeX - 1) / groupSizeX);  // Division with rounding up.
    cmd_buf.dispatch(groupCountX, 1, 1);
}
