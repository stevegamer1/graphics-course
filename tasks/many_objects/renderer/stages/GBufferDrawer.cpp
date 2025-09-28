#include "GBufferDrawer.hpp"
#include "SynchronizedBuffer.hpp"
#include <vulkan/vulkan_enums.hpp>
#include "etna/DescriptorSet.hpp"
#include "etna/Etna.hpp"
#include "etna/PipelineManager.hpp"
#include "etna/RenderTargetStates.hpp"


void GBufferDrawer::loadShader()
{
  etna::create_program(
    PROGRAM_NAME,
    {MANY_OBJECTS_RENDERER_SHADERS_ROOT "gbuffer_create.frag.spv",
     MANY_OBJECTS_RENDERER_SHADERS_ROOT "gbuffer_create.vert.spv"});
}

void GBufferDrawer::createPipeline(
  vk::Format albedo_format, vk::Format normal_format, vk::Format depth_format,
  etna::VertexByteStreamFormatDescription vertex_format_description)
{
  etna::VertexShaderInputDescription sceneVertexInputDesc{
    .bindings = {etna::VertexShaderInputDescription::Binding{
      .byteStreamDescription = vertex_format_description,
    }},
  };

  auto& pipelineManager = etna::get_context().getPipelineManager();

  pipeline = {};
  pipeline = pipelineManager.createGraphicsPipeline(
    PROGRAM_NAME,
    etna::GraphicsPipeline::CreateInfo{
      .vertexShaderInput = sceneVertexInputDesc,
      .rasterizationConfig =
        vk::PipelineRasterizationStateCreateInfo{
          .polygonMode = vk::PolygonMode::eFill,
          .cullMode = vk::CullModeFlagBits::eBack,
          .frontFace = vk::FrontFace::eCounterClockwise,
          .lineWidth = 1.f,
        },
      .blendingConfig = {
        .attachments ={
          vk::PipelineColorBlendAttachmentState{
            .blendEnable = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
          },
          vk::PipelineColorBlendAttachmentState{
            .blendEnable = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG,
          }
        },
        .logicOp = {}
      },
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {albedo_format, normal_format},
          .depthAttachmentFormat = depth_format,
        },
    });
}

void GBufferDrawer::run(
  vk::CommandBuffer cmd_buf,

    SynchronizedBuffer& draw_params,
    SynchronizedBuffer& indirect_commands,
    SynchronizedBuffer& draw_params_indices,
    SynchronizedBuffer& commands_count,

    vk::Image albedo_image,
    vk::ImageView albedo_image_view,
    vk::Image normals_image,
    vk::ImageView normals_image_view,
    vk::Image depth_image,
    vk::ImageView depth_image_view,
    vk::Buffer vertex_buffer,
    vk::Buffer index_buffer,

    glm::uvec2 resolution,
    uint32_t relems_count,
    glm::mat4 proj_view)
{
  BufferSyncUsage graphicsUsage{
    .stageFlags =
      vk::PipelineStageFlagBits2::eVertexShader |
      vk::PipelineStageFlagBits2::eFragmentShader |
      vk::PipelineStageFlagBits2::eDrawIndirect,
    .accessFlags =
      vk::AccessFlagBits2::eUniformRead |
      vk::AccessFlagBits2::eShaderStorageRead |
      vk::AccessFlagBits2::eIndirectCommandRead,
  };
  draw_params.syncBeforeUsage(graphicsUsage, cmd_buf);
  draw_params_indices.syncBeforeUsage(graphicsUsage, cmd_buf);
  indirect_commands.syncBeforeUsage(graphicsUsage, cmd_buf);
  commands_count.syncBeforeUsage(graphicsUsage, cmd_buf);

  {
    etna::RenderTargetState renderTargets(
      cmd_buf,
      {{0, 0}, {resolution.x, resolution.y}},
      {
        {.image = albedo_image, .view = albedo_image_view},
        {.image = normals_image, .view = normals_image_view}
      },
      {.image = depth_image, .view = depth_image_view});

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.getVkPipeline());

    cmd_buf.bindVertexBuffers(0, {vertex_buffer}, {0});
    cmd_buf.bindIndexBuffer(index_buffer, 0, vk::IndexType::eUint32);

    {
      auto programInfo = etna::get_shader_program(PROGRAM_NAME);

      auto set = etna::create_descriptor_set(
        programInfo.getDescriptorLayoutId(0),
        cmd_buf,
        {etna::Binding{0, draw_params.buffer.genBinding()},
         etna::Binding(1, draw_params_indices.buffer.genBinding())});

      cmd_buf.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, pipeline.getVkPipelineLayout(), 0, {set.getVkSet()}, {});
    }

    PushConstants pushConst{.projView = proj_view};
    cmd_buf.pushConstants<PushConstants>(
      pipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eVertex, 0, {pushConst});

    etna::flush_barriers(cmd_buf);
    cmd_buf.drawIndexedIndirectCount(
      indirect_commands.buffer.get(),
      vk::DeviceSize(0),
      commands_count.buffer.get(),
      vk::DeviceSize(0),
      uint32_t(relems_count),
      uint32_t(sizeof(vk::DrawIndexedIndirectCommand)));
  }
}
