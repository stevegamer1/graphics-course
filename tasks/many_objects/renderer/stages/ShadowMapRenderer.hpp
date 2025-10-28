#pragma once
#include "GBufferLightResolver.hpp"
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>


class ShadowMapRenderer
{
public:
  ShadowMapRenderer();

  ShadowMapRenderer(const ShadowMapRenderer&) = delete;
  ShadowMapRenderer(ShadowMapRenderer&&) = delete;
  ShadowMapRenderer& operator=(const ShadowMapRenderer&) = delete;
  ShadowMapRenderer& operator=(ShadowMapRenderer&&) = delete;

  void loadShader();

  void createPipeline();

  static constexpr glm::uvec2 RESOLUTION{1024, 1024};  // approximately full hd screen
  // static constexpr uint32_t IMAGES_IN_CASCADE = 4;  // idk feels ok
  static constexpr uint32_t IMAGES_IN_CASCADE = 1;  // no cascade because i can't do it all in one take
  static constexpr vk::Format DEPTH_FORMAT = vk::Format::eD32Sfloat;

  static glm::mat4x4 createViewProjMatrix(
    const GBufferLightResolver::Light& light
  );

  std::vector<etna::Image> createCascades(uint32_t lights_count);
  
  void run(
    vk::CommandBuffer cmd_buf,

    const GBufferLightResolver::Light& light,

    std::span<etna::Image, IMAGES_IN_CASCADE> shadowmaps_target,

    vk::Buffer vertex_buffer,
    vk::Buffer index_buffer,
    const etna::Buffer& transform_matrices,
    vk::Buffer scene_indirect_draw_commands,
    uint32_t relem_count
  );

private:
  etna::Sampler defaultSampler;

  struct PushConstants
  {
    glm::mat4x4 projView;
  };

  etna::GraphicsPipeline pipeline;
  const char* PROGRAM_NAME = "shadow_map_program";
};
