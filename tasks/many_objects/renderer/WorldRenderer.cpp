#include "WorldRenderer.hpp"
#include "etna/Buffer.hpp"
#include "etna/Image.hpp"
#include "stages/SynchronizedBuffer.hpp"

#include <cstdint>
#include <cstring>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/Profiling.hpp>
#include <glm/ext.hpp>
#include <utility>
#include <vector>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>


WorldRenderer::WorldRenderer(const etna::GpuWorkCount& work_count)
  : sceneMgr{std::make_unique<SceneManager>()}
  , drawParams(work_count, std::in_place_t{})
  , drawParamsCulledIndicesBuffer(work_count, std::in_place_t{})
  , instanceMeshToIndirectCommandMap(work_count, std::in_place_t{})
  , aabbBuffer(work_count, std::in_place_t{})
  , indirectCommandsBuffer(work_count, std::in_place_t{})
  , indirectCommandsCountBuffer(work_count, std::in_place_t{})
{
}

void WorldRenderer::allocateResources(glm::uvec2 swapchain_resolution)
{
  resolution = swapchain_resolution;

  drawer.createDepthImage(resolution);
}

void WorldRenderer::loadScene(std::filesystem::path path)
{
  sceneMgr->selectScene(path);
}

void WorldRenderer::loadShaders()
{
  culler.loadShader();
  drawer.loadShader();
}

void WorldRenderer::setupPipelines(vk::Format swapchain_format)
{
  drawer.createPipeline(swapchain_format, sceneMgr->getVertexFormatDescription());
  culler.createPipeline();
}

void WorldRenderer::debugInput(const Keyboard&) {}

void WorldRenderer::update(const FramePacket& packet)
{
  ZoneScoped;

  // calc camera matrix
  {
    const float aspect = float(resolution.x) / float(resolution.y);
    worldViewProj = packet.mainCam.projTm(aspect) * packet.mainCam.viewTm();
  }
}

void WorldRenderer::recreateDrawParamsBuffers(uint32_t count) {
  size_t drawParamsSize = sizeof(SingleRelemDrawParams) * count;
  size_t culledIndicesSize = sizeof(uint32_t) * count;

  drawParams.get().buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = sizeof(SingleRelemDrawParams) * count,
    .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU,
    .name = "drawParams",
  });
  drawParams.get().current_size = drawParamsSize;

  drawParamsCulledIndicesBuffer.get().buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = culledIndicesSize,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "drawParamsCulledIndicesBuffer",
  });
  drawParamsCulledIndicesBuffer.get().current_size = culledIndicesSize;
}

void WorldRenderer::recreateIndirectCommandsBuffer(uint32_t count) {
  size_t size = sizeof(SingleRelemDrawParams) * count;
  indirectCommandsBuffer.get().buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = size,
    .bufferUsage = vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU,
    .name = "indirectCommands",
  });
  indirectCommandsBuffer.get().current_size = size;
}

void WorldRenderer::createIndirectCommandCountBuffer() {
  size_t size = sizeof(uint32_t);
  indirectCommandsCountBuffer.get().buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = size,
    .bufferUsage = vk::BufferUsageFlagBits::eIndirectBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU,
    .name = "indirectCommandCountBuffer"
  });
  indirectCommandsCountBuffer.get().current_size = size;
}

void WorldRenderer::recreateAABBBuffer(uint32_t count) {
  size_t size = sizeof(AABB) * count;
  aabbBuffer.get().buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = size,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU,
    .name = "AABBBuffer"
  });
  aabbBuffer.get().current_size = size;
}

void WorldRenderer::recreateInstancesToCommandsMapBuffer(uint32_t count) {
  size_t size = sizeof(uint32_t) * count;
  instanceMeshToIndirectCommandMap.get().buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = size,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU,
    .name = "instanceMeshToIndirectCommandMap"
  });
  instanceMeshToIndirectCommandMap.get().current_size = size;
}

template <typename Element>
void WorldRenderer::uploadBuffer(const std::vector<Element>& source, SynchronizedBuffer& destination, vk::CommandBuffer cmd_buf) {
  destination.syncBeforeUsage(BufferSyncUsage{
    .stageFlags = vk::PipelineStageFlagBits2::eHost,
    .accessFlags = vk::AccessFlagBits2::eHostWrite
  }, cmd_buf);

  destination.buffer.map();
  memcpy(destination.buffer.data(), source.data(), source.size() * sizeof(Element));
  destination.buffer.unmap();
}

WorldRenderer::AABB WorldRenderer::calculateAABB(const RenderElement&) {
  AABB result;

  result.min = glm::vec4(-100);
  result.max = glm::vec4(+100);

  return result;
}

void WorldRenderer::recalculateAABBsCPU(vk::CommandBuffer cmd_buf) {
  auto relems = sceneMgr->getRenderElements();
  std::vector<AABB> aabbs(relems.size());

  for (size_t i = 0; i < relems.size(); ++i) {
    aabbs[i] = calculateAABB(relems[i]);
  }

  uploadBuffer(aabbs, aabbBuffer.get(), cmd_buf);
}

std::map<WorldRenderer::RelemID, std::vector<WorldRenderer::SingleRelemDrawParams>> WorldRenderer::collectRelemsDrawParamsForIndirect() {
  std::map<RelemID, std::vector<SingleRelemDrawParams>> result;

  auto instanceMatrices = sceneMgr->getInstanceMatrices();
  auto instanceMeshes = sceneMgr->getInstanceMeshes();
  auto meshes = sceneMgr->getMeshes();
  for (std::size_t instIdx = 0; instIdx < instanceMeshes.size(); ++instIdx)
  {
    const auto meshIdx = instanceMeshes[instIdx];

    for (std::size_t j = 0; j < meshes[meshIdx].relemCount; ++j)
    {
      const auto relemIdx = meshes[meshIdx].firstRelem + j;
      result[uint32_t(relemIdx)].push_back({
        instanceMatrices[instIdx]
      });
    }
  }

  return result;
}

WorldRenderer::BuffersForDrawIndexedIndirectCount WorldRenderer::prepareDrawParamsBuffersOnCPU() {
  BuffersForDrawIndexedIndirectCount result;

  auto relems = sceneMgr->getRenderElements();

  std::map<RelemID, std::vector<SingleRelemDrawParams>> relemsDrawParams = collectRelemsDrawParamsForIndirect();
  for (auto& kvPair : relemsDrawParams) {
    auto& [relemId, paramsVector] = kvPair;
 
    result.commands.push_back({
      .indexCount = uint32_t(relems[relemId].indexCount),
      .instanceCount = uint32_t(paramsVector.size()),
      .firstIndex = relems[relemId].indexOffset,
      .vertexOffset = int32_t(relems[relemId].vertexOffset),
      .firstInstance = uint32_t(result.draw_params.size())
    });

    result.draw_params.resize(result.draw_params.size() + paramsVector.size());
    std::copy(paramsVector.begin(), paramsVector.end(), result.draw_params.end() - paramsVector.size());
  }

  return result;
}

WorldRenderer::BuffersForCulling WorldRenderer::prepareCullingBuffersOnCPU(const std::vector<vk::DrawIndexedIndirectCommand>& commands) {
  BuffersForCulling result;

  for (uint32_t c = 0; c < commands.size(); ++c) {
    for (uint32_t i = 0; i < commands[c].instanceCount; ++i) {
      result.instanceToIndirectCommandMap.push_back(c);
    }
  }

  return result;
}

void WorldRenderer::recreateAndUploadBuffersIfNecessary(vk::CommandBuffer cmd_buf) {
  BuffersForDrawIndexedIndirectCount cpuBuffersForIndirectDraw = prepareDrawParamsBuffersOnCPU();
  BuffersForCulling cullingBuffers = prepareCullingBuffersOnCPU(cpuBuffersForIndirectDraw.commands);

  {
    uint32_t desiredDrawParamsCount = uint32_t(cpuBuffersForIndirectDraw.draw_params.size());
    uint32_t currentCount = uint32_t(drawParams.get().current_size / sizeof(SingleRelemDrawParams));
    if (currentCount < desiredDrawParamsCount) {
      recreateDrawParamsBuffers(desiredDrawParamsCount);
    }
    uploadBuffer(cpuBuffersForIndirectDraw.draw_params, drawParams.get(), cmd_buf);
  }

  {
    uint32_t desiredInstancesToCommandsMapEntryCount = uint32_t(cpuBuffersForIndirectDraw.draw_params.size());
    uint32_t currentCount = uint32_t(instanceMeshToIndirectCommandMap.get().current_size / sizeof(uint32_t));
    if (currentCount < desiredInstancesToCommandsMapEntryCount) {
      recreateInstancesToCommandsMapBuffer(desiredInstancesToCommandsMapEntryCount);
    }
    uploadBuffer(cullingBuffers.instanceToIndirectCommandMap, instanceMeshToIndirectCommandMap.get(), cmd_buf);
  }

  {
    uint32_t desiredCommandsCount = uint32_t(cpuBuffersForIndirectDraw.commands.size());
    uint32_t currentCount = uint32_t(indirectCommandsBuffer.get().current_size / sizeof(vk::DrawIndexedIndirectCommand));
    if (currentCount < desiredCommandsCount) {
      recreateIndirectCommandsBuffer(desiredCommandsCount);
    }
    uploadBuffer(cpuBuffersForIndirectDraw.commands, indirectCommandsBuffer.get(), cmd_buf);
  }

  {
    if (indirectCommandsCountBuffer.get().current_size == 0) {
      createIndirectCommandCountBuffer();
    }
    std::vector<uint32_t> countVector{uint32_t(cpuBuffersForIndirectDraw.commands.size())};
    uploadBuffer(countVector, indirectCommandsCountBuffer.get(), cmd_buf);
  }

  {
    uint32_t desiredAABBCount = uint32_t(cpuBuffersForIndirectDraw.draw_params.size());
    uint32_t currentCount = uint32_t(aabbBuffer.get().current_size / sizeof(uint32_t));
    if (currentCount < desiredAABBCount) {
      recreateAABBBuffer(desiredAABBCount);
    }
    recalculateAABBsCPU(cmd_buf);
  }
}

void WorldRenderer::cullMeshes(vk::CommandBuffer cmd_buf, const glm::mat4x4& glob_tm) {
  culler.run(cmd_buf,
    drawParams.get(), 
    aabbBuffer.get(), 
    indirectCommandsBuffer.get(), 
    drawParamsCulledIndicesBuffer.get(), 
    instanceMeshToIndirectCommandMap.get(),
    uint32_t(sceneMgr->getInstanceMeshes().size()), glob_tm);
}

void WorldRenderer::renderScene(
  vk::CommandBuffer cmd_buf, const glm::mat4x4& glob_tm, vk::Image target_image, vk::ImageView target_image_view)
{
  if (!sceneMgr->getVertexBuffer())
    return;

  drawer.run(cmd_buf,
    drawParams.get(),
    aabbBuffer.get(),
    indirectCommandsBuffer.get(),
    drawParamsCulledIndicesBuffer.get(),
    instanceMeshToIndirectCommandMap.get(),
    indirectCommandsCountBuffer.get(),
    target_image,
    target_image_view,
    sceneMgr->getVertexBuffer(),
    sceneMgr->getIndexBuffer(),
    resolution,
    uint32_t(sceneMgr->getRenderElements().size()),
    glob_tm);
}

void WorldRenderer::renderWorld(
  vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_image_view)
{
  ETNA_PROFILE_GPU(cmd_buf, renderWorld);

  // draw final scene to screen
  {
    ETNA_PROFILE_GPU(cmd_buf, renderForward);

    recreateAndUploadBuffersIfNecessary(cmd_buf);

    cullMeshes(cmd_buf, worldViewProj);

    renderScene(cmd_buf, worldViewProj, target_image, target_image_view);

    drawParams.get().resetAccumulatedUsage();
    drawParamsCulledIndicesBuffer.get().resetAccumulatedUsage();
    instanceMeshToIndirectCommandMap.get().resetAccumulatedUsage();
    aabbBuffer.get().resetAccumulatedUsage();
    indirectCommandsBuffer.get().resetAccumulatedUsage();
    indirectCommandsCountBuffer.get().resetAccumulatedUsage();
  }
}
