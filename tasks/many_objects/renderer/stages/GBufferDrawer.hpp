#pragma once
#include "etna/Buffer.hpp"
#include "etna/DescriptorSet.hpp"
#include "etna/GraphicsPipeline.hpp"
#include "etna/Sampler.hpp"
#include <etna/Vulkan.hpp>
#include <glm/fwd.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>
#include <glm/ext/matrix_float4x4.hpp>


class GBufferDrawer
{
public:
  explicit GBufferDrawer();

  GBufferDrawer(const GBufferDrawer&) = delete;
  GBufferDrawer(GBufferDrawer&&) = delete;
  GBufferDrawer& operator=(const GBufferDrawer&) = delete;
  GBufferDrawer& operator=(GBufferDrawer&&) = delete;

  void loadShader();

  void createPipeline(
    vk::Format albedo_format, vk::Format metal_rough_format, vk::Format normal_format, vk::Format depth_format,
    etna::VertexByteStreamFormatDescription vertex_format_description);

  // Call every time textures change.
  void updateTexturesDescriptorSet(std::span<const etna::Image> textures);

  void run(
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
    glm::mat4 proj_view);

private:
  struct PushConstants
  {
    glm::mat4x4 projView;
  };

  const vk::PipelineColorBlendAttachmentState albedoAttachmentState = vk::PipelineColorBlendAttachmentState{
    .blendEnable = vk::False,
    .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
  };
  const vk::PipelineColorBlendAttachmentState metallicRoughnessAttachmentState = vk::PipelineColorBlendAttachmentState{
    .blendEnable = vk::False,
    .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB,
  };
  const vk::PipelineColorBlendAttachmentState normalsAttachmentState = vk::PipelineColorBlendAttachmentState{
    .blendEnable = vk::False,
    .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG,
  };

  etna::GraphicsPipeline pipeline;
  etna::Sampler defaultSampler;
  etna::PersistentDescriptorSet texturesDescriptorSet;
  const char* PROGRAM_NAME = "gbuffer_generate_program";
};
