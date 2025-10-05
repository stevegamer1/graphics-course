#include "WorldRenderer.hpp"
#include "etna/BlockingTransferHelper.hpp"
#include "etna/Buffer.hpp"
#include "etna/Image.hpp"
#include "stages/CullingManager.hpp"
#include "stages/GBufferLightResolver.hpp"
#include "stages/BufferWithSize.hpp"

#include <cstdint>
#include <cstring>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/Profiling.hpp>
#include <glm/ext.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/common.hpp>
#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/fwd.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/matrix.hpp>
#include <utility>
#include <vector>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>


WorldRenderer::WorldRenderer(const etna::GpuWorkCount& work_count)
  : oneShotCommands{etna::get_context().createOneShotCmdMgr()}
  , transferHelper(etna::BlockingTransferHelper::CreateInfo{ .stagingSize = 1024 })
  , sceneMgr{std::make_unique<SceneManager>()}
  , lights(work_count, std::in_place_t{})
  , drawParams(work_count, std::in_place_t{})
  , drawParamsCulledIndicesBuffer(work_count, std::in_place_t{})
  , instanceMeshToIndirectCommandMap(work_count, std::in_place_t{})
  , aabbBuffer(work_count, std::in_place_t{})
  , indirectCommandsBuffer(work_count, std::in_place_t{})
  , indirectCommandsCountBuffer(work_count, std::in_place_t{})
  , albedoImage(work_count, std::in_place_t{})
  , albedoImageResolution(work_count, std::in_place_t{})
  , normalsImage(work_count, std::in_place_t{})
  , normalsImageResolution(work_count, std::in_place_t{})
  , depthImage(work_count, std::in_place_t{})
  , depthImageResolution(work_count, std::in_place_t{})
{
}

void WorldRenderer::allocateResources(glm::uvec2 swapchain_resolution)
{
  resolution = swapchain_resolution;
  lightGBufferResolver.allocateAndFillResources();
}

void WorldRenderer::loadScene(std::filesystem::path path)
{
  sceneMgr->selectScene(path);
}

void WorldRenderer::loadShaders()
{
  aabbCalculator.loadShader();
  culler.loadShader();
  gbufferDrawer.loadShader();
  lightGBufferResolver.loadShader();
}

void WorldRenderer::setupPipelines(vk::Format swapchain_format)
{
  aabbCalculator.createPipeline();
  culler.createPipeline();
  gbufferDrawer.createPipeline(
    ALBEDO_FORMAT, NORMAL_FORMAT, DEPTH_FORMAT,
    sceneMgr->getVertexFormatDescription());
  lightGBufferResolver.createPipeline(
    swapchain_format);
}

void WorldRenderer::debugInput(const Keyboard&) {}

void WorldRenderer::update(const FramePacket& packet)
{
  ZoneScoped;

  // calc camera matrix
  {
    const float aspect = float(resolution.x) / float(resolution.y);
    worldViewProj = packet.mainCam.projTm(aspect) * packet.mainCam.viewTm();
    cameraCopy = packet.mainCam;
  }
}

void WorldRenderer::recreateDrawParamsBuffers(uint32_t count) {
  size_t drawParamsSize = sizeof(SingleRelemDrawParams) * count;
  size_t culledIndicesSize = sizeof(uint32_t) * count;

  drawParams.get().buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = drawParamsSize,
    .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "drawParams",
  });
  drawParams.get().size = drawParamsSize;

  drawParamsCulledIndicesBuffer.get().buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = culledIndicesSize,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "drawParamsCulledIndicesBuffer",
  });
  drawParamsCulledIndicesBuffer.get().size = culledIndicesSize;
}

void WorldRenderer::recreateIndirectCommandsBuffer(uint32_t count) {
  size_t size = sizeof(vk::DrawIndexedIndirectCommand) * count;
  indirectCommandsBuffer.get().buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = size,
    .bufferUsage = vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "indirectCommands",
  });
  indirectCommandsBuffer.get().size = size;
}

void WorldRenderer::createIndirectCommandCountBuffer() {
  size_t size = sizeof(uint32_t);
  indirectCommandsCountBuffer.get().buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = size,
    .bufferUsage = vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "indirectCommandCountBuffer"
  });
  indirectCommandsCountBuffer.get().size = size;
}

void WorldRenderer::recreateAABBBuffer(uint32_t count) {
  size_t size = sizeof(AABB) * count;
  aabbBuffer.get().buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = size,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "AABBBuffer"
  });
  aabbBuffer.get().size = size;
}

void WorldRenderer::recreateInstancesToCommandsMapBuffer(uint32_t count) {
  size_t size = sizeof(uint32_t) * count;
  instanceMeshToIndirectCommandMap.get().buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = size,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "instanceMeshToIndirectCommandMap"
  });
  instanceMeshToIndirectCommandMap.get().size = size;
}

void WorldRenderer::recreateLights(uint32_t count) {
  size_t size = sizeof(GBufferLightResolver::Light) * count;
  lights.get().buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = size,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "lights",
  });
  lights.get().size = size;
}

void WorldRenderer::recalculateAABBs(vk::CommandBuffer cmd_buf) {
  aabbCalculator.run(cmd_buf,
    aabbBuffer.get().buffer,
    indirectCommandsBuffer.get().buffer,
    sceneMgr->getVertexBufferEtna(),
    sceneMgr->getIndexBufferEtna(),
    uint32_t(sceneMgr->getRenderElements().size()));
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
  uint32_t instancesCount = uint32_t(sceneMgr->getInstanceMeshes().size());
  uint32_t relemsCount = uint32_t(sceneMgr->getRenderElements().size());

  {
    uint32_t desiredDrawParamsCount = instancesCount;
    uint32_t currentCount = uint32_t(drawParams.get().size / sizeof(SingleRelemDrawParams));
    if (currentCount < desiredDrawParamsCount) {
      recreateDrawParamsBuffers(desiredDrawParamsCount);
      markSceneDirty();
    }
  }

  {
    uint32_t desiredInstancesToCommandsMapEntryCount = instancesCount;
    uint32_t currentCount = uint32_t(instanceMeshToIndirectCommandMap.get().size / sizeof(uint32_t));
    if (currentCount < desiredInstancesToCommandsMapEntryCount) {
      recreateInstancesToCommandsMapBuffer(desiredInstancesToCommandsMapEntryCount);
      markSceneDirty();
    }
  }

  {
    uint32_t desiredCommandsCount = relemsCount;
    uint32_t currentCount = uint32_t(indirectCommandsBuffer.get().size / sizeof(vk::DrawIndexedIndirectCommand));
    if (currentCount < desiredCommandsCount) {
      recreateIndirectCommandsBuffer(desiredCommandsCount);
      markSceneDirty();
    }
  }

  {
    if (indirectCommandsCountBuffer.get().size == 0) {
      createIndirectCommandCountBuffer();
      markSceneDirty();
    }
  }

  {
    uint32_t desiredAABBCount = relemsCount;
    uint32_t currentCount = uint32_t(aabbBuffer.get().size / sizeof(AABB));
    if (currentCount < desiredAABBCount) {
      recreateAABBBuffer(desiredAABBCount);
      markAABBsDirty();
    } 
  }

  {
    uint32_t desiredLightsCount = uint32_t(lightsVector.size());
    uint32_t currentCount = uint32_t(lights.get().size / sizeof(GBufferLightResolver::Light));
    if (currentCount < desiredLightsCount) {
      recreateLights(desiredLightsCount);
      markSceneDirty();
    } 
  }

  if (albedoImageResolution.get() != resolution) {
    albedoImage.get() = etna::get_context().createImage({
      .extent = vk::Extent3D{resolution.x, resolution.y, 1},
      .name = "G-Buffer albedo image",
      .format = ALBEDO_FORMAT,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
    });
    albedoImageResolution.get() = resolution;
  }

  if (normalsImageResolution.get() != resolution) {
    normalsImage.get() = etna::get_context().createImage({
      .extent = vk::Extent3D{resolution.x, resolution.y, 1},
      .name = "G-Buffer normals image",
      .format = NORMAL_FORMAT,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
    });
    normalsImageResolution.get() = resolution;
  }

  if (depthImageResolution.get() != resolution) {
    depthImage.get() = etna::get_context().createImage({
      .extent = vk::Extent3D{resolution.x, resolution.y, 1},
      .name = "G-Buffer depth image",
      .format = DEPTH_FORMAT,
      .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled
    });
    depthImageResolution.get() = resolution;
  }

  if (sceneDirty) {
    BuffersForDrawIndexedIndirectCount cpuBuffersForIndirectDraw = prepareDrawParamsBuffersOnCPU();
    BuffersForCulling cullingBuffers = prepareCullingBuffersOnCPU(cpuBuffersForIndirectDraw.commands);

    // Need SynchronizedBuffer::syncForUsage here?
    transferHelper.uploadBuffer(
      *oneShotCommands, 
      drawParams.get().buffer, 0, 
      std::span<const SingleRelemDrawParams>(cpuBuffersForIndirectDraw.draw_params));
    
    transferHelper.uploadBuffer(
      *oneShotCommands, 
      instanceMeshToIndirectCommandMap.get().buffer, 0, 
      std::span<const uint32_t>(cullingBuffers.instanceToIndirectCommandMap));

    transferHelper.uploadBuffer(
      *oneShotCommands, 
      indirectCommandsBuffer.get().buffer, 0, 
      std::span<const vk::DrawIndexedIndirectCommand>(cpuBuffersForIndirectDraw.commands));
    
    std::vector<uint32_t> countVector{uint32_t(cpuBuffersForIndirectDraw.commands.size())};
    transferHelper.uploadBuffer(
      *oneShotCommands, 
      indirectCommandsCountBuffer.get().buffer, 0, 
      std::span<const uint32_t>(countVector));
    
    using Light = GBufferLightResolver::Light;
    transferHelper.uploadBuffer(
      *oneShotCommands, 
      lights.get().buffer, 0, 
      std::span<const Light>(lightsVector));

    sceneDirty = false;
  }

  if (aabbsDirty) {
    recalculateAABBs(cmd_buf);
    aabbsDirty = false;
  }
}

// matrix is forward transformation matrix.
glm::vec4 transform_plane(glm::vec4 plane, glm::mat4 matrix) {
  glm::vec4 resultXYZ4 = glm::normalize(matrix * glm::vec4(plane.x, plane.y, plane.z, 0.0f));

  glm::vec4 pointOnPlane(glm::vec3(plane) * plane.w, 1.0f);
  float resultW = glm::dot(matrix * pointOnPlane, resultXYZ4);

  return glm::vec4(resultXYZ4.x, resultXYZ4.y, resultXYZ4.z, resultW);
}


void WorldRenderer::cullMeshes(vk::CommandBuffer cmd_buf, const Camera& camera) {
  CullingManager::Frustum frustum = CullingManager::getFrustum(camera.fov, camera.zNear, camera.zFar, float(resolution.x) / resolution.y);

  glm::mat4x4 invView = cameraCopy.viewItm();
  frustum.near = transform_plane(frustum.near, invView);
  frustum.far = transform_plane(frustum.far, invView);
  frustum.left = transform_plane(frustum.left, invView);
  frustum.right = transform_plane(frustum.right, invView);
  frustum.top = transform_plane(frustum.top, invView);
  frustum.bottom = transform_plane(frustum.bottom, invView);

  culler.run(cmd_buf,
    drawParams.get().buffer, 
    aabbBuffer.get().buffer, 
    indirectCommandsBuffer.get().buffer, 
    drawParamsCulledIndicesBuffer.get().buffer, 
    instanceMeshToIndirectCommandMap.get().buffer,
    uint32_t(sceneMgr->getInstanceMeshes().size()), frustum);
}

void WorldRenderer::generateGBuffer(
  vk::CommandBuffer cmd_buf, const glm::mat4x4& glob_tm)
{
  if (!sceneMgr->getVertexBuffer())
    return;

  gbufferDrawer.run(cmd_buf,
    drawParams.get().buffer,
    indirectCommandsBuffer.get().buffer,
    drawParamsCulledIndicesBuffer.get().buffer,
    indirectCommandsCountBuffer.get().buffer,
    albedoImage.get().get(),
    albedoImage.get().getView({}),
    normalsImage.get().get(),
    normalsImage.get().getView({}),
    depthImage.get().get(),
    depthImage.get().getView({}),
    sceneMgr->getVertexBuffer(),
    sceneMgr->getIndexBuffer(),
    resolution,
    uint32_t(sceneMgr->getRenderElements().size()),
    glob_tm);
}

void WorldRenderer::resolveGBufferWithLights(
  vk::CommandBuffer cmd_buf, const glm::mat4x4& glob_tm, vk::Image target_image, vk::ImageView target_image_view)
{
  if (!sceneMgr->getVertexBuffer())
    return;

  lightGBufferResolver.run(cmd_buf,
    lights.get().buffer,
    uint32_t(lightsVector.size()),
    albedoImage.get(),
    normalsImage.get(),
    depthImage.get(),
    target_image,
    target_image_view,

    resolution,
    glob_tm,
    cameraCopy.position);
}

void WorldRenderer::renderWorld(
  vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_image_view)
{
  ETNA_PROFILE_GPU(cmd_buf, renderWorld);

  // draw final scene to screen
  {
    ETNA_PROFILE_GPU(cmd_buf, renderForward);

    recreateAndUploadBuffersIfNecessary(cmd_buf);

    cullMeshes(cmd_buf, cameraCopy);

    generateGBuffer(cmd_buf, worldViewProj);

    resolveGBufferWithLights(cmd_buf, worldViewProj, target_image, target_image_view);
  }
}
