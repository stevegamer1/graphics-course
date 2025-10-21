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
#include "stages/SceneUploader.hpp"
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

  void markAABBsDirty() { aabbsDirty = true; }

private:
  void clearAttachments(
    vk::CommandBuffer cmd_buf);

  void generateGBuffer(
    vk::CommandBuffer cmd_buf, uint32_t first_command, uint32_t command_count, const glm::mat4x4& glob_tm);

  void cullMeshes(
    vk::CommandBuffer cmd_buf, const Camera& camera);

  void resolveGBufferWithLights(
    vk::CommandBuffer cmd_buf, const glm::mat4x4& glob_tm, vk::Image target_image, vk::ImageView target_image_view);

  void recreateAndUploadBuffersIfNecessary();

  struct AABB {
    glm::vec4 min;
    glm::vec4 max;
  };

  void recreateDrawParamsBuffers(uint32_t count);
  void recreateIndirectCommandsBuffer(uint32_t count);
  void recreateAABBBuffer(uint32_t count);
  void recreateLights(uint32_t count);

  void recalculateAABBs(vk::CommandBuffer cmd_buf);

private:
  std::unique_ptr<etna::OneShotCmdMgr> oneShotCommands;
  etna::BlockingTransferHelper transferHelper;
  std::unique_ptr<SceneManager> sceneMgr;
  bool loaded = false;

  BufferWithSize lights;
  etna::GpuSharedResource<BufferWithSize> drawParamsCulledIndicesBuffer;
  BufferWithSize aabbBuffer;

  etna::GpuSharedResource<etna::Image> albedoImage;
  glm::uvec2 albedoImageResolution {};
  static const vk::Format ALBEDO_FORMAT = vk::Format::eR8G8B8A8Srgb;
  etna::GpuSharedResource<etna::Image> metallicRoughnessImage;
  glm::uvec2 metallicRoughnessImageResolution {};
  static const vk::Format METAL_ROUGH_FORMAT = vk::Format::eR8G8B8A8Srgb;
  etna::GpuSharedResource<etna::Image> normalsImage;
  glm::uvec2 normalsImageResolution {};
  static const vk::Format NORMAL_FORMAT = vk::Format::eR8G8Snorm;
  etna::GpuSharedResource<etna::Image> depthImage;
  glm::uvec2 depthImageResolution {};
  static const vk::Format DEPTH_FORMAT = vk::Format::eD32Sfloat;

  glm::mat4x4 worldViewProj;
  Camera cameraCopy;
  glm::mat4x4 lightMatrix;

  SceneUploader sceneUploader;
  CullingManager culler;
  GBufferDrawer gbufferDrawer;
  GBufferLightResolver lightGBufferResolver;
  AABBCalculator aabbCalculator;

  glm::uvec2 resolution;

  bool aabbsDirty = true;

  std::vector<GBufferLightResolver::Light> lightsVector = {
    {
      .posAndIntensity = glm::vec4(1.0f, 1.0f, -1.0f, 1.0f),
      .color = glm::vec3(1.0f, 0.0f, 0.0f),
      .lightType = GBufferLightResolver::Light::LightType::Point
    },
    {
      .posAndIntensity = glm::vec4(1.0f, 1.0f, 1.0f, 5.0f),
      .color = glm::vec3(1.0f, 1.0f, 1.0f),
      .lightType = GBufferLightResolver::Light::LightType::Directional
    },
    {
      .posAndIntensity = glm::vec4(0.5f, 0.5f, 1.0f, 0.1f),
      .color = glm::vec3(1.0f, 1.0f, 1.0f),
      .lightType = GBufferLightResolver::Light::LightType::Ambient
    },
  };
};
