#pragma once
#include "BufferWithSize.hpp"
#include "etna/BlockingTransferHelper.hpp"
#include "etna/DescriptorSet.hpp"
#include "etna/GraphicsPipeline.hpp"
#include <cstdint>
#include <etna/Sampler.hpp>
#include "etna/OneShotCmdMgr.hpp"
#include <etna/Vulkan.hpp>
#include <glm/fwd.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>
#include <glm/ext/matrix_float4x4.hpp>


class GBufferLightResolver
{
public:
  GBufferLightResolver();

  GBufferLightResolver(const GBufferLightResolver&) = delete;
  GBufferLightResolver(GBufferLightResolver&&) = delete;
  GBufferLightResolver& operator=(const GBufferLightResolver&) = delete;
  GBufferLightResolver& operator=(GBufferLightResolver&&) = delete;

  struct Light {
    enum class LightType : uint8_t {
      Point = 0,
      Directional = 1,
      Ambient = 2
    };

    glm::vec4 posAndIntensity;
    glm::vec3 color;
    uint32_t padding23_castsShadows1_lightType8 = 0;
    static constexpr uint32_t CASTS_SHADOWS_MASK = 0x00000100;
    static constexpr uint32_t LIGHT_TYPE_MASK = 0x000000FF;

    Light(glm::vec3 pos, float intensity, glm::vec3 color, bool casts_shadows, LightType type);

    bool castsShadows() const;

    LightType getType() const;
  };

  void loadShader();

  void createPipeline(
    vk::Format output_color_format);

  void createShadowmapsDescriptorSet(std::span<etna::Image> lights_shadowmaps);

  void allocateAndFillResources();

  void run(
    vk::CommandBuffer cmd_buf,

    etna::Buffer& lights,
    const etna::Buffer& shadow_lights_viewproj_matrices,
    uint32_t lights_count,
    std::span<etna::Image> lights_shadowmaps,

    etna::Image& albedo_image,
    etna::Image& metallic_roughness_image,
    etna::Image& normals_image,
    etna::Image& depth_image,
    vk::Image color_image,
    vk::ImageView color_image_view,

    glm::uvec2 resolution,
    glm::mat4 proj_view,
    glm::vec3 cam_pos,
    float cam_near
  );

private:
  std::unique_ptr<etna::OneShotCmdMgr> oneShotCommands;
  etna::BlockingTransferHelper transferHelper;
  etna::Sampler defaultSampler;
  etna::PersistentDescriptorSet shadowmapSet;

  static constexpr int SPHERE_ROWS = 10;
  static constexpr int SPHERE_COLUMNS = 18;

  struct PushConstants
  {
    glm::mat4x4 projView;
    glm::vec4 wCamPos;
    glm::uvec2 resolution;
    float camNear;
  };

  using Vertex = glm::vec4;

  std::vector<Vertex> sphereVertices;
  std::vector<uint32_t> sphereIndices;

  BufferWithSize sphereVertexBuffer;
  BufferWithSize sphereIndexBuffer;

  etna::GraphicsPipeline pipeline;
  const char* PROGRAM_NAME = "resolve_gbuffer_with_light_program";
};
