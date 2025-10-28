#include "ShadowMapRenderer.hpp"
#include "GBufferLightResolver.hpp"
#include "etna/Assert.hpp"
#include "etna/DescriptorSet.hpp"
#include "etna/Etna.hpp"
#include "etna/GlobalContext.hpp"
#include "etna/Image.hpp"
#include "etna/PipelineManager.hpp"
#include "etna/RenderTargetStates.hpp"
#include "etna/Sampler.hpp"
#include "etna/ShaderProgram.hpp"
#include "scene/SceneManager.hpp"
#include <glm/common.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/fwd.hpp>
#include <glm/matrix.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_structs.hpp>


ShadowMapRenderer::ShadowMapRenderer() {
  defaultSampler = etna::Sampler(etna::Sampler::CreateInfo{.name = "default_sampler"});
}

void ShadowMapRenderer::createPipeline() {
  etna::VertexShaderInputDescription sceneVertexInputDesc{
    .bindings = {etna::VertexShaderInputDescription::Binding{
      .byteStreamDescription = etna::VertexByteStreamFormatDescription{
        .stride = sizeof(SceneManager::Vertex),
        .attributes={{
          .format=vk::Format::eR32G32B32A32Sfloat,
          .offset=0
        }}
      },
    }},
  };

  pipeline = etna::get_context().getPipelineManager().createGraphicsPipeline(
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
        .attachments ={},
        .logicOp = {}
      },
      .fragmentShaderOutput =
        {
          .depthAttachmentFormat = DEPTH_FORMAT
        },
    });
}

void ShadowMapRenderer::loadShader() {
  etna::create_program(
    PROGRAM_NAME,
    {
      MANY_OBJECTS_RENDERER_SHADERS_ROOT "shadowmap.frag.spv",
      MANY_OBJECTS_RENDERER_SHADERS_ROOT "shadowmap.vert.spv"
    }
  );
}

glm::mat4x4 ShadowMapRenderer::createViewProjMatrix(
  const GBufferLightResolver::Light& light
) {
  ETNA_VERIFYF(light.getType() == GBufferLightResolver::Light::LightType::Directional, "Sorry, no shadows for non-directional lights");
  glm::mat4x4 viewMatrix = glm::lookAtLH(glm::vec3(light.posAndIntensity), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
  glm::mat4x4 projMatrix = glm::orthoLH_ZO(-3.0f, 3.0f, -3.0f, 3.0f, -10.0f, 10.0f);

  return projMatrix * viewMatrix;
}

std::vector<etna::Image> ShadowMapRenderer::createCascades(uint32_t lights_count) {
  std::vector<etna::Image> result;
  result.reserve(lights_count * IMAGES_IN_CASCADE);
  for (uint32_t l = 0; l < lights_count; ++l) {
    for (uint32_t i = 0; i < IMAGES_IN_CASCADE; ++i) {
      result.push_back(etna::get_context().createImage(etna::Image::CreateInfo{
        .extent = vk::Extent3D{RESOLUTION.x, RESOLUTION.y, 1},
        .name = std::format("Image number {} of a shadowmap cascade", i),
        .format = ShadowMapRenderer::DEPTH_FORMAT,
        .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eDepthStencilAttachment
      }));
    }
  }
  return result;
}

void ShadowMapRenderer::run(
  vk::CommandBuffer cmd_buf,

  const GBufferLightResolver::Light& light,

  std::span<etna::Image, IMAGES_IN_CASCADE> shadowmaps_target,

  vk::Buffer vertex_buffer,
  vk::Buffer index_buffer,
  const etna::Buffer& transform_matrices,
  vk::Buffer scene_indirect_draw_commands,
  uint32_t relem_count
)
{
  for (uint32_t i = 0; i < shadowmaps_target.size(); ++i) {
    etna::set_state(
      cmd_buf,
      shadowmaps_target[i].get(),
      vk::PipelineStageFlagBits2::eFragmentShader,
      vk::AccessFlagBits2::eShaderWrite,
      vk::ImageLayout::eDepthAttachmentOptimal,
      vk::ImageAspectFlagBits::eDepth
    );
  }

  etna::flush_barriers(cmd_buf);

  for (uint32_t i = 0; i < shadowmaps_target.size(); ++i) {
    etna::RenderTargetState renderTargets(
      cmd_buf,
      {{0, 0}, {RESOLUTION.x, RESOLUTION.y}},
      {},
      {.image = shadowmaps_target[i].get(), .view = shadowmaps_target[i].getView(etna::Image::ViewParams{})});

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.getVkPipeline());

    cmd_buf.bindVertexBuffers(0, {vertex_buffer}, {0});
    cmd_buf.bindIndexBuffer(index_buffer, 0, vk::IndexType::eUint32);

    {
      etna::ShaderProgramInfo programInfo = etna::get_shader_program(PROGRAM_NAME);

      etna::DescriptorSet set = etna::create_descriptor_set(
        programInfo.getDescriptorLayoutId(0),
        cmd_buf,
        {
          etna::Binding{0, transform_matrices.genBinding()}
        }
      );

      cmd_buf.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, pipeline.getVkPipelineLayout(), 0, {set.getVkSet()}, {});
    }

    PushConstants pushConst{
      .projView = createViewProjMatrix(light)
    };
    cmd_buf.pushConstants<PushConstants>(
      pipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eVertex, 0, {pushConst});

    cmd_buf.drawIndexedIndirect(scene_indirect_draw_commands, vk::DeviceSize{0}, relem_count, sizeof(vk::DrawIndexedIndirectCommand));
  }

}
