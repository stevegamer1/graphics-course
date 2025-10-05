#pragma once

#include <cstdint>
#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/Buffer.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "etna/BlockingTransferHelper.hpp"
#include "etna/GpuSharedResource.hpp"
#include "etna/OneShotCmdMgr.hpp"
#include "scene/SceneManager.hpp"
#include "stages/AABBCalculator.hpp"
#include "stages/CullingManager.hpp"
#include "stages/GBufferDrawer.hpp"
#include "stages/GBufferLightResolver.hpp"
#include "stages/BufferWithSize.hpp"
#include "wsi/Keyboard.hpp"

#include "FramePacket.hpp"


class WorldRenderer
{
public:
  explicit WorldRenderer(const etna::GpuWorkCount& work_count);

  void loadScene(std::filesystem::path path);

  void loadShaders();
  void allocateResources(glm::uvec2 swapchain_resolution);
  void setupPipelines(vk::Format swapchain_format);

  void debugInput(const Keyboard& kb);
  void update(const FramePacket& packet);
  void drawGui();
  void renderWorld(
    vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_image_view);

  void markSceneDirty() { sceneDirty = true; }
  void markAABBsDirty() { aabbsDirty = true; }

private:
  void generateGBuffer(
    vk::CommandBuffer cmd_buf, const glm::mat4x4& glob_tm);

  void cullMeshes(
    vk::CommandBuffer cmd_buf, const Camera& camera);

  void resolveGBufferWithLights(
    vk::CommandBuffer cmd_buf, const glm::mat4x4& glob_tm, vk::Image target_image, vk::ImageView target_image_view);

  void recreateAndUploadBuffersIfNecessary(vk::CommandBuffer cmd_buf);

  struct SingleRelemDrawParams {
    glm::mat4x4 model;
  };

  struct AABB {
    glm::vec4 min;
    glm::vec4 max;
  };

  struct BuffersForDrawIndexedIndirectCount {
    std::vector<SingleRelemDrawParams> draw_params;
    std::vector<vk::DrawIndexedIndirectCommand> commands;
  };

  struct BuffersForCulling {
    std::vector<uint32_t> instanceToIndirectCommandMap;
  };

  using RelemID = uint32_t;
  // todo: collect by material, not relem.
  std::map<RelemID, std::vector<SingleRelemDrawParams>> collectRelemsDrawParamsForIndirect();
  BuffersForDrawIndexedIndirectCount prepareDrawParamsBuffersOnCPU();
  BuffersForCulling prepareCullingBuffersOnCPU(const std::vector<vk::DrawIndexedIndirectCommand>& commands);

  void recreateDrawParamsBuffers(uint32_t count);
  void recreateIndirectCommandsBuffer(uint32_t count);
  void createIndirectCommandCountBuffer();
  void recreateAABBBuffer(uint32_t count);
  void recreateInstancesToCommandsMapBuffer(uint32_t count);
  void recreateLights(uint32_t count);

  void recalculateAABBs(vk::CommandBuffer cmd_buf);

private:
  std::unique_ptr<etna::OneShotCmdMgr> oneShotCommands;
  etna::BlockingTransferHelper transferHelper;
  std::unique_ptr<SceneManager> sceneMgr;

  etna::GpuSharedResource<BufferWithSize> lights;
  etna::GpuSharedResource<BufferWithSize> drawParams;
  etna::GpuSharedResource<BufferWithSize> drawParamsCulledIndicesBuffer;
  etna::GpuSharedResource<BufferWithSize> instanceMeshToIndirectCommandMap;
  etna::GpuSharedResource<BufferWithSize> aabbBuffer;
  etna::GpuSharedResource<BufferWithSize> indirectCommandsBuffer;
  etna::GpuSharedResource<BufferWithSize> indirectCommandsCountBuffer;

  etna::GpuSharedResource<etna::Image> albedoImage;
  etna::GpuSharedResource<glm::uvec2> albedoImageResolution;
  const vk::Format ALBEDO_FORMAT = vk::Format::eR8G8B8A8Srgb;
  etna::GpuSharedResource<etna::Image> normalsImage;
  etna::GpuSharedResource<glm::uvec2> normalsImageResolution;
  const vk::Format NORMAL_FORMAT = vk::Format::eR8G8Snorm;
  etna::GpuSharedResource<etna::Image> depthImage;
  etna::GpuSharedResource<glm::uvec2> depthImageResolution;
  const vk::Format DEPTH_FORMAT = vk::Format::eD32Sfloat;

  glm::mat4x4 worldViewProj;
  Camera cameraCopy;
  glm::mat4x4 lightMatrix;

  CullingManager culler;
  GBufferDrawer gbufferDrawer;
  GBufferLightResolver lightGBufferResolver;
  AABBCalculator aabbCalculator;

  glm::uvec2 resolution;

  bool aabbsDirty = true;
  bool sceneDirty = true;

  std::vector<GBufferLightResolver::Light> lightsVector = {
    {
      .posAndIntensity = glm::vec4(1.0f, 1.0f, -1.0f, 1.0f),
      .color = glm::vec3(1.0f, 0.0f, 0.0f),
      .lightType = GBufferLightResolver::Light::LightType::Point
    },
    {
      .posAndIntensity = glm::vec4(1.0f, 1.0f, 1.0f, 0.1f),
      .color = glm::vec3(0.0f, 1.0f, 0.0f),
      .lightType = GBufferLightResolver::Light::LightType::Directional
    },
    {
      .posAndIntensity = glm::vec4(0.0f, 0.0f, 0.0f, 0.1f),
      .color = glm::vec3(0.0f, 0.0f, 1.0f),
      .lightType = GBufferLightResolver::Light::LightType::Ambient
    },
  };
};
