#pragma once

#include <cstdint>
#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/Buffer.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "etna/GpuSharedResource.hpp"
#include "scene/SceneManager.hpp"
#include "stages/CullingManager.hpp"
#include "stages/IndirectDrawManager.hpp"
#include "stages/SynchronizedBuffer.hpp"
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

private:
  void renderScene(
    vk::CommandBuffer cmd_buf, const glm::mat4x4& glob_tm, vk::Image target_image, vk::ImageView target_image_view);

  void cullMeshes(
    vk::CommandBuffer cmd_buf, const glm::mat4x4& glob_tm);

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

  // void simpleBufferSynchronizeAfterUpload(vk::CommandBuffer cmd_buf, vk::Buffer buffer);
  // void simpleBufferSynchronizeAfterCompute(vk::CommandBuffer cmd_buf, vk::Buffer buffer);

  void recreateDrawParamsBuffers(uint32_t count);
  void recreateIndirectCommandsBuffer(uint32_t count);
  void createIndirectCommandCountBuffer();
  void recreateAABBBuffer(uint32_t count);
  void recreateInstancesToCommandsMapBuffer(uint32_t count);

  template <typename Element>
  void uploadBuffer(const std::vector<Element>& source, SynchronizedBuffer& destination, vk::CommandBuffer cmd_buf);

  // void uploadDrawParamsBuffer(const std::vector<SingleRelemDrawParams>& source, vk::CommandBuffer cmd_buf);
  // void uploadIndirectDrawCommands(const std::vector<vk::DrawIndexedIndirectCommand>& source, vk::CommandBuffer cmd_buf);
  // void uploadIndirectCommandCount(uint32_t count, vk::CommandBuffer cmd_buf);
  // void uploadAABBs(const std::vector<AABB>& source, vk::CommandBuffer cmd_buf);
  // void uploadInstancesToCommandsMapBuffer(const std::vector<uint32_t>& source, vk::CommandBuffer cmd_buf);

  AABB calculateAABB(const RenderElement& relem);
  void recalculateAABBsCPU(vk::CommandBuffer cmd_buf);
  void recalculateAABBs(vk::CommandBuffer cmd_buf);

private:
  std::unique_ptr<SceneManager> sceneMgr;

  etna::GpuSharedResource<SynchronizedBuffer> drawParams;
  etna::GpuSharedResource<SynchronizedBuffer> drawParamsCulledIndicesBuffer;
  etna::GpuSharedResource<SynchronizedBuffer> instanceMeshToIndirectCommandMap;
  etna::GpuSharedResource<SynchronizedBuffer> aabbBuffer;
  etna::GpuSharedResource<SynchronizedBuffer> indirectCommandsBuffer;
  etna::GpuSharedResource<SynchronizedBuffer> indirectCommandsCountBuffer;

  // struct StaticMeshPipelinePushConstants
  // {
  //   glm::mat4x4 projView;
  // };

  // struct CullPipelinePushConstants
  // {
  //   glm::mat4x4 projView;
  // };

  glm::mat4x4 worldViewProj;
  glm::mat4x4 lightMatrix;

  // etna::GraphicsPipeline staticMeshPipeline{};
  // const char* STATIC_MESH_PROGRAM_NAME = "static_mesh_program";

  // etna::ComputePipeline cullPipeline{};
  CullingManager culler;
  IndirectDrawManager drawer;


  glm::uvec2 resolution;
};
