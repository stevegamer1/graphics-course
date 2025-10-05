#include "IndirectDrawManager.hpp"
#include <vulkan/vulkan_enums.hpp>
#include "etna/Buffer.hpp"
#include "etna/DescriptorSet.hpp"
#include "etna/Etna.hpp"
#include "etna/PipelineManager.hpp"
#include "etna/RenderTargetStates.hpp"


void IndirectDrawManager::loadShader()
{
  etna::create_program(
    PROGRAM_NAME,
    {MANY_OBJECTS_RENDERER_SHADERS_ROOT "static_mesh.frag.spv",
     MANY_OBJECTS_RENDERER_SHADERS_ROOT "static_mesh.vert.spv"});
}

void IndirectDrawManager::createPipeline(
  vk::Format swapchain_format, etna::VertexByteStreamFormatDescription vertex_format_description)
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
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {swapchain_format},
          .depthAttachmentFormat = DEPTH_ATTACHMENT_FORMAT,
        },
    });
}

void IndirectDrawManager::createDepthImage(glm::uvec2 resolution)
{
  mainViewDepth = etna::get_context().createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "main_view_depth",
    .format = DEPTH_ATTACHMENT_FORMAT,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
  });
}

void IndirectDrawManager::run(
  vk::CommandBuffer cmd_buf,
  etna::Buffer& draw_params,
  etna::Buffer& indirect_commands,
  etna::Buffer& draw_params_indices,
  etna::Buffer& commands_count,
  vk::Image target_image,
  vk::ImageView target_image_view,
  vk::Buffer vertex_buffer,
  vk::Buffer index_buffer,
  glm::uvec2 resolution,
  uint32_t relems_count,
  glm::mat4 proj_view)
{
  vk::PipelineStageFlags2 pipelineStage =
      vk::PipelineStageFlagBits2::eVertexShader |
      vk::PipelineStageFlagBits2::eFragmentShader |
      vk::PipelineStageFlagBits2::eDrawIndirect;
  vk::AccessFlags2 access =
      vk::AccessFlagBits2::eUniformRead |
      vk::AccessFlagBits2::eShaderStorageRead |
      vk::AccessFlagBits2::eIndirectCommandRead;

  etna::set_state(cmd_buf, draw_params.get(), pipelineStage, access);
  etna::set_state(cmd_buf, draw_params_indices.get(), pipelineStage, access);
  etna::set_state(cmd_buf, indirect_commands.get(), pipelineStage, access);
  etna::set_state(cmd_buf, commands_count.get(), pipelineStage, access);

  etna::flush_barriers(cmd_buf);

  {
    etna::RenderTargetState renderTargets(
      cmd_buf,
      {{0, 0}, {resolution.x, resolution.y}},
      {{.image = target_image, .view = target_image_view}},
      {.image = mainViewDepth.get(), .view = mainViewDepth.getView({})});

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.getVkPipeline());

    cmd_buf.bindVertexBuffers(0, {vertex_buffer}, {0});
    cmd_buf.bindIndexBuffer(index_buffer, 0, vk::IndexType::eUint32);

    {
      auto programInfo = etna::get_shader_program(PROGRAM_NAME);

      auto set = etna::create_descriptor_set(
        programInfo.getDescriptorLayoutId(0),
        cmd_buf,
        {etna::Binding{0, draw_params.genBinding()},
         etna::Binding(1, draw_params_indices.genBinding())});

      cmd_buf.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, pipeline.getVkPipelineLayout(), 0, {set.getVkSet()}, {});
    }

    PushConstants pushConst{.projView = proj_view};
    cmd_buf.pushConstants<PushConstants>(
      pipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eVertex, 0, {pushConst});

    etna::flush_barriers(cmd_buf);
    cmd_buf.drawIndexedIndirectCount(
      indirect_commands.get(),
      vk::DeviceSize(0),
      commands_count.get(),
      vk::DeviceSize(0),
      uint32_t(relems_count),
      uint32_t(sizeof(vk::DrawIndexedIndirectCommand)));
  }
}
