#include "GBufferDrawer.hpp"
#include <vector>
#include <vulkan/vulkan_enums.hpp>
#include "etna/Buffer.hpp"
#include "etna/DescriptorSet.hpp"
#include "etna/Etna.hpp"
#include "etna/GlobalContext.hpp"
#include "etna/Image.hpp"
#include "etna/PipelineManager.hpp"
#include "etna/RenderTargetStates.hpp"


GBufferDrawer::GBufferDrawer()
{
  defaultSampler = etna::Sampler(etna::Sampler::CreateInfo{.name = "default_sampler"});
}

void GBufferDrawer::loadShader()
{
  etna::create_program(
    PROGRAM_NAME,
    {MANY_OBJECTS_RENDERER_SHADERS_ROOT "gbuffer_create.frag.spv",
     MANY_OBJECTS_RENDERER_SHADERS_ROOT "gbuffer_create.vert.spv"});
}

void GBufferDrawer::createPipeline(
  vk::Format albedo_format, vk::Format metal_rough_format, vk::Format normal_format, vk::Format depth_format,
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
          albedoAttachmentState,
          metallicRoughnessAttachmentState,
          normalsAttachmentState
        },
        .logicOp = {}
      },
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {albedo_format, metal_rough_format, normal_format},
          .depthAttachmentFormat = depth_format,
        },
    });
}

void GBufferDrawer::updateTexturesDescriptorSet(std::span<const etna::Image> textures) {
  auto programInfo = etna::get_shader_program(PROGRAM_NAME);
  std::vector<etna::Binding> bindings;

  for (uint32_t i = 0; i < textures.size(); ++i) {
    bindings.emplace_back(0, textures[i].genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal), i);
  }

  texturesDescriptorSet = etna::create_persistent_descriptor_set(
    programInfo.getDescriptorLayoutId(1),
    bindings,
    true
  );
}

void GBufferDrawer::run(
  vk::CommandBuffer cmd_buf,

    const etna::Buffer& draw_params,
    const etna::Buffer& indirect_commands,
    const etna::Buffer& draw_params_indices,
    const etna::Buffer& instances_to_commands_map,

    vk::Image albedo_image,
    vk::ImageView albedo_image_view,
    vk::Image metallic_roughness_image,
    vk::ImageView metallic_roughness_image_view,
    vk::Image normals_image,
    vk::ImageView normals_image_view,
    vk::Image depth_image,
    vk::ImageView depth_image_view,
    vk::Buffer vertex_buffer,
    vk::Buffer index_buffer,

    std::span<const etna::Image> textures,
    const etna::Buffer& materials,

    glm::uvec2 resolution,
    uint32_t first_relem,
    uint32_t relems_count,
    glm::mat4 proj_view)
{
  vk::PipelineStageFlags2 bufferStage =
      vk::PipelineStageFlagBits2::eVertexShader |
      vk::PipelineStageFlagBits2::eFragmentShader |
      vk::PipelineStageFlagBits2::eDrawIndirect;
  vk::PipelineStageFlags2 colorAttachmentStage =
      vk::PipelineStageFlagBits2::eColorAttachmentOutput |
      vk::PipelineStageFlagBits2::eFragmentShader |
      vk::PipelineStageFlagBits2::eDrawIndirect;
  vk::PipelineStageFlags2 depthAttachmentStage =
      vk::PipelineStageFlagBits2::eEarlyFragmentTests |
      vk::PipelineStageFlagBits2::eFragmentShader |
      vk::PipelineStageFlagBits2::eDrawIndirect;
  vk::PipelineStageFlags2 sampledTexturesStage =
      vk::PipelineStageFlagBits2::eFragmentShader;

  vk::AccessFlags2 bufferAccess =
      vk::AccessFlagBits2::eUniformRead |
      vk::AccessFlagBits2::eShaderStorageRead |
      vk::AccessFlagBits2::eIndirectCommandRead;
  vk::AccessFlags2 colorAttachmentAccess =
      vk::AccessFlagBits2::eColorAttachmentRead |
      vk::AccessFlagBits2::eColorAttachmentWrite;
  vk::AccessFlags2 depthAttachmentAccess =
      vk::AccessFlagBits2::eDepthStencilAttachmentRead |
      vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
  vk::AccessFlags2 sampledTexturesAccess =
      vk::AccessFlagBits2::eShaderSampledRead;
  
  etna::set_state(cmd_buf, albedo_image, colorAttachmentStage, colorAttachmentAccess, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageAspectFlagBits::eColor);
  etna::set_state(cmd_buf, metallic_roughness_image, colorAttachmentStage, colorAttachmentAccess, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageAspectFlagBits::eColor);
  etna::set_state(cmd_buf, normals_image, colorAttachmentStage, colorAttachmentAccess, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageAspectFlagBits::eColor);
  etna::set_state(cmd_buf, depth_image, depthAttachmentStage, depthAttachmentAccess, vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageAspectFlagBits::eDepth);

  etna::set_state(cmd_buf, draw_params.get(), bufferStage, bufferAccess);
  etna::set_state(cmd_buf, draw_params_indices.get(), bufferStage, bufferAccess);
  etna::set_state(cmd_buf, indirect_commands.get(), bufferStage, bufferAccess);
  etna::set_state(cmd_buf, instances_to_commands_map.get(), bufferStage, bufferAccess);

  for (const etna::Image& texture : textures) {
    etna::set_state(cmd_buf, texture.get(), sampledTexturesStage, sampledTexturesAccess, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageAspectFlagBits::eColor);
  }

  etna::set_state(cmd_buf, materials.get(), bufferStage, bufferAccess);

  etna::flush_barriers(cmd_buf);

  {
    etna::RenderTargetState renderTargets(
      cmd_buf,
      {{0, 0}, {resolution.x, resolution.y}},
      {
        {.image = albedo_image, .view = albedo_image_view, .loadOp=vk::AttachmentLoadOp::eLoad},
        {.image = metallic_roughness_image, .view = metallic_roughness_image_view, .loadOp=vk::AttachmentLoadOp::eLoad},
        {.image = normals_image, .view = normals_image_view, .loadOp=vk::AttachmentLoadOp::eLoad}
      },
      {.image = depth_image, .view = depth_image_view, .loadOp=vk::AttachmentLoadOp::eLoad});

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.getVkPipeline());

    cmd_buf.bindVertexBuffers(0, {vertex_buffer}, {0});
    cmd_buf.bindIndexBuffer(index_buffer, 0, vk::IndexType::eUint32);

    {
      auto programInfo = etna::get_shader_program(PROGRAM_NAME);

      std::vector<etna::Binding> bindings;

      bindings.emplace_back(0, draw_params.genBinding());
      bindings.emplace_back(1, draw_params_indices.genBinding());
      bindings.emplace_back(2, materials.genBinding());
      bindings.emplace_back(3, instances_to_commands_map.genBinding());

      auto set = etna::create_descriptor_set(
        programInfo.getDescriptorLayoutId(0),
        cmd_buf,
        bindings
      );

      assert(set.isValid());
      assert(texturesDescriptorSet.isValid());

      cmd_buf.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, pipeline.getVkPipelineLayout(), 0, {set.getVkSet(), texturesDescriptorSet.getVkSet()}, {});
    }

    PushConstants pushConst{.projView = proj_view};
    cmd_buf.pushConstants<PushConstants>(
      pipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eVertex, 0, {pushConst});
    
    cmd_buf.drawIndexedIndirect(
      indirect_commands.get(),
      vk::DeviceSize(sizeof(vk::DrawIndexedIndirectCommand) * first_relem),
      relems_count,
      uint32_t(sizeof(vk::DrawIndexedIndirectCommand)));
  }
}
