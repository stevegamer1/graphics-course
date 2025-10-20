#pragma once
#include "BufferWithSize.hpp"
#include "etna/BlockingTransferHelper.hpp"
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
    enum class LightType : uint32_t {
      Point = 0,
      Directional = 1,
      Ambient = 2
    };

    glm::vec4 posAndIntensity;
    glm::vec3 color;
    LightType lightType;
  };

  void loadShader();

  void createPipeline(
    vk::Format output_color_format);

  void allocateAndFillResources();

  // Synchronizes buffers by itself.
  void run(
    vk::CommandBuffer cmd_buf,

    etna::Buffer& lights,
    uint32_t lights_count,

    etna::Image& albedo_image,
    etna::Image& metallic_roughness_image,
    etna::Image& normals_image,
    etna::Image& depth_image,
    vk::Image color_image,
    vk::ImageView color_image_view,

    glm::uvec2 resolution,
    glm::mat4 proj_view,
    glm::vec3 cam_pos
  );

private:
  std::unique_ptr<etna::OneShotCmdMgr> oneShotCommands;
  etna::BlockingTransferHelper transferHelper;
  etna::Sampler defaultSampler;

  static constexpr int SPHERE_ROWS = 10;
  static constexpr int SPHERE_COLUMNS = 18;

  struct PushConstants
  {
    glm::mat4x4 projView;
    glm::vec4 wCamPos;
    glm::uvec2 resolution;
  };

  using Vertex = glm::vec4;

  std::vector<Vertex> sphereVertices;
  std::vector<uint32_t> sphereIndices;

  BufferWithSize sphereVertexBuffer;
  BufferWithSize sphereIndexBuffer;

  etna::GraphicsPipeline pipeline;
  const char* PROGRAM_NAME = "resolve_gbuffer_with_light_program";
};
