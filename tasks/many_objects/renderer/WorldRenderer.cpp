#include "WorldRenderer.hpp"
#include "etna/BlockingTransferHelper.hpp"
#include "etna/Buffer.hpp"
#include "etna/Etna.hpp"
#include "etna/Image.hpp"
#include "scene/SceneManager.hpp"
#include "stages/CullingManager.hpp"
#include "stages/GBufferDrawer.hpp"
#include "stages/GBufferLightResolver.hpp"
#include "stages/BufferWithSize.hpp"
#include "stages/SceneUploader.hpp"
#include "stages/ShadowMapRenderer.hpp"

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
  , drawParamsCulledIndicesBuffer(work_count, std::in_place_t{})
  , albedoImage(work_count, std::in_place_t{})
  , metallicRoughnessImage(work_count, std::in_place_t{})
  , normalsImage(work_count, std::in_place_t{})
  , depthImage(work_count, std::in_place_t{})
  , shadowmaps(work_count, std::in_place_t{})
  , lightsProjViewMatrices(work_count, std::in_place_t{})
  , gbufferDrawer()
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
  sceneUploader.updateScene(*sceneMgr);
  recreateAndUploadBuffersIfNecessary();
  gbufferDrawer.updateTexturesDescriptorSet(sceneMgr->getImages());

  uint32_t shadowLightsCount = 0;
  for (uint32_t l = 0; l < lightsVector.size(); ++l) {
    if (lightsVector[l].castsShadows()) {
      ++shadowLightsCount;
    }
  }
  shadowmaps.iterate([this, shadowLightsCount](std::vector<etna::Image>& cascades) {
    cascades = shadowMapRenderer.createCascades(shadowLightsCount);
  });
  lightGBufferResolver.createShadowmapsDescriptorSet(shadowmaps.get());

  lightsProjViewMatrices.iterate([shadowLightsCount](etna::Buffer& matrices){
    matrices = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
      .size = std::max(sizeof(glm::mat4x4) * shadowLightsCount, size_t{1}),
      .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
      .name = "Lights view-projection matrices"
    });
  });

  aabbsDirty = true;
  loaded = true;
}

void WorldRenderer::loadShaders()
{
  aabbCalculator.loadShader();
  culler.loadShader();
  gbufferDrawer.loadShader();
  lightGBufferResolver.loadShader();
  shadowMapRenderer.loadShader();
}

void WorldRenderer::setupPipelines(vk::Format swapchain_format)
{
  aabbCalculator.createPipeline();
  culler.createPipeline();
  gbufferDrawer.createPipeline(
    ALBEDO_FORMAT, METAL_ROUGH_FORMAT, NORMAL_FORMAT, DEPTH_FORMAT,
    sceneMgr->getVertexFormatDescription());
  lightGBufferResolver.createPipeline(
    swapchain_format);
  shadowMapRenderer.createPipeline();
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
  size_t culledIndicesSize = sizeof(uint32_t) * count;

  drawParamsCulledIndicesBuffer.iterate([culledIndicesSize](BufferWithSize& buffer){
    buffer.buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
      .size = culledIndicesSize,
      .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
      .name = "drawParamsCulledIndicesBuffer",
    });
    buffer.size = culledIndicesSize;
  });
}

void WorldRenderer::recreateAABBBuffer(uint32_t count) {
  size_t size = sizeof(AABB) * count;
  aabbBuffer.buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = size,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "AABBBuffer"
  });
  aabbBuffer.size = size;
}

void WorldRenderer::recreateLights(uint32_t count) {
  size_t size = sizeof(GBufferLightResolver::Light) * count;
  lights.buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = size,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "lights",
  });
  lights.size = size;
}

void WorldRenderer::recalculateAABBs(vk::CommandBuffer cmd_buf) {
  aabbCalculator.run(cmd_buf,
    aabbBuffer.buffer,
    sceneUploader.getIndirectCommandsBuffer().buffer,
    sceneMgr->getVertexBufferEtna(),
    sceneMgr->getIndexBufferEtna(),
    uint32_t(sceneMgr->getRenderElements().size()));
}

void WorldRenderer::renderShadowmaps(vk::CommandBuffer cmd_buf) {
  std::vector<glm::mat4x4> lightsViewProjMatricesVector;
  for (uint32_t l = 0; l < lightsVector.size(); ++l) {
    if (lightsVector[l].castsShadows()) {
      lightsViewProjMatricesVector.push_back(shadowMapRenderer.createViewProjMatrix(lightsVector[l]));
    }
  }

  // TODO: this, as i understand it, kills frames in flight.
  transferHelper.uploadBuffer(*oneShotCommands, lightsProjViewMatrices.get(), 0, std::span<const glm::mat4x4>(lightsViewProjMatricesVector));

  uint32_t shadowmapIndex = 0;
  for (uint32_t l = 0; l < lightsVector.size(); ++l) {
    if (lightsVector[l].castsShadows()) {
      auto beginShadowmap = shadowmaps.get().begin() + shadowmapIndex * ShadowMapRenderer::IMAGES_IN_CASCADE;
      auto endShadowmap = beginShadowmap + ShadowMapRenderer::IMAGES_IN_CASCADE;

      shadowMapRenderer.run(
        cmd_buf,
        lightsVector[l],
        std::span<etna::Image, ShadowMapRenderer::IMAGES_IN_CASCADE>(beginShadowmap, endShadowmap),
        sceneMgr->getVertexBuffer(),
        sceneMgr->getIndexBuffer(),
        sceneUploader.getMatricesBuffer().buffer,
        sceneUploader.getIndirectCommandsBuffer().buffer.get(),
        sceneUploader.getPipelines().at(PipelineType::PBR).commandCount
      );
      ++shadowmapIndex;
    }
  }
}

void WorldRenderer::recreateAndUploadBuffersIfNecessary() {
  uint32_t instancesCount = uint32_t(sceneMgr->getInstanceMeshes().size());
  uint32_t relemsCount = uint32_t(sceneMgr->getRenderElements().size());

  {
    uint32_t desiredDrawParamsCount = instancesCount;
    uint32_t currentCount = uint32_t(drawParamsCulledIndicesBuffer.get().size / sizeof(uint32_t));
    if (currentCount < desiredDrawParamsCount) {
      recreateDrawParamsBuffers(desiredDrawParamsCount);
    }
  }

  {
    uint32_t desiredAABBCount = relemsCount;
    uint32_t currentCount = uint32_t(aabbBuffer.size / sizeof(AABB));
    if (currentCount < desiredAABBCount) {
      recreateAABBBuffer(desiredAABBCount);
      aabbsDirty = true;
    } 
  }

  {
    uint32_t desiredLightsCount = uint32_t(lightsVector.size());
    uint32_t currentCount = uint32_t(lights.size / sizeof(GBufferLightResolver::Light));
    if (currentCount < desiredLightsCount) {
      recreateLights(desiredLightsCount);
    } 
  }

  if (albedoImageResolution != resolution) {
    const glm::uvec2 resCapture = resolution;
    albedoImage.iterate([resCapture](etna::Image& image){
      image = etna::get_context().createImage({
        .extent = vk::Extent3D{resCapture.x, resCapture.y, 1},
        .name = "G-Buffer albedo image",
        .format = ALBEDO_FORMAT,
        .imageUsage =
          vk::ImageUsageFlagBits::eColorAttachment |
          vk::ImageUsageFlagBits::eSampled |
          vk::ImageUsageFlagBits::eTransferDst
      });
    });
    albedoImageResolution = resolution;
  }

  if (metallicRoughnessImageResolution != resolution) {
    const glm::uvec2 resCapture = resolution;
    metallicRoughnessImage.iterate([resCapture](etna::Image& image){
      image = etna::get_context().createImage({
        .extent = vk::Extent3D{resCapture.x, resCapture.y, 1},
        .name = "G-Buffer metallic & roughness image",
        .format = METAL_ROUGH_FORMAT,
        .imageUsage =
          vk::ImageUsageFlagBits::eColorAttachment |
          vk::ImageUsageFlagBits::eSampled |
          vk::ImageUsageFlagBits::eTransferDst
      });
    });
    metallicRoughnessImageResolution = resolution;
  }

  if (normalsImageResolution != resolution) {
    const glm::uvec2 resCapture = resolution;
    normalsImage.iterate([resCapture](etna::Image& image){
      image = etna::get_context().createImage({
        .extent = vk::Extent3D{resCapture.x, resCapture.y, 1},
        .name = "G-Buffer normals image",
        .format = NORMAL_FORMAT,
        .imageUsage =
          vk::ImageUsageFlagBits::eColorAttachment |
          vk::ImageUsageFlagBits::eSampled |
          vk::ImageUsageFlagBits::eTransferDst
      });
    });
    normalsImageResolution = resolution;
  }

  if (depthImageResolution != resolution) {
    const glm::uvec2 resCapture = resolution;
    depthImage.iterate([resCapture](etna::Image& image){
      image = etna::get_context().createImage({
        .extent = vk::Extent3D{resCapture.x, resCapture.y, 1},
        .name = "G-Buffer depth image",
        .format = DEPTH_FORMAT,
        .imageUsage =
          vk::ImageUsageFlagBits::eDepthStencilAttachment |
          vk::ImageUsageFlagBits::eSampled |
          vk::ImageUsageFlagBits::eTransferDst
      });
    });
    depthImageResolution = resolution;
  }

  using Light = GBufferLightResolver::Light;
  transferHelper.uploadBuffer(
    *oneShotCommands,
    lights.buffer, 0,
    std::span<const Light>(lightsVector));
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
    sceneUploader.getMatricesBuffer().buffer, 
    aabbBuffer.buffer, 
    sceneUploader.getIndirectCommandsBuffer().buffer, 
    drawParamsCulledIndicesBuffer.get().buffer, 
    sceneUploader.getInstancesToCommandsBuffer().buffer,
    uint32_t(sceneMgr->getInstanceMeshes().size()), frustum);
}

void WorldRenderer::clearAttachments(vk::CommandBuffer cmd_buf)
{
  vk::ImageLayout layout = vk::ImageLayout::eTransferDstOptimal;

  etna::set_state(
    cmd_buf, 
    albedoImage.get().get(), 
    vk::PipelineStageFlagBits2::eClear, 
    vk::AccessFlagBits2::eTransferWrite, 
    layout, 
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmd_buf, 
    metallicRoughnessImage.get().get(), 
    vk::PipelineStageFlagBits2::eClear, 
    vk::AccessFlagBits2::eTransferWrite,
    layout, 
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmd_buf, 
    normalsImage.get().get(), 
    vk::PipelineStageFlagBits2::eClear, 
    vk::AccessFlagBits2::eTransferWrite,
    layout, 
    vk::ImageAspectFlagBits::eColor);

  etna::set_state(
    cmd_buf, 
    depthImage.get().get(), 
    vk::PipelineStageFlagBits2::eClear, 
    vk::AccessFlagBits2::eTransferWrite, 
    layout, 
    vk::ImageAspectFlagBits::eDepth);
  
  etna::flush_barriers(cmd_buf);
  
  cmd_buf.clearColorImage(
    albedoImage.get().get(), 
    layout, 
    vk::ClearColorValue{0, 0, 0, 1},
  {
    vk::ImageSubresourceRange {
      .aspectMask = vk::ImageAspectFlagBits::eColor,
      .levelCount = 1,
      .layerCount = 1
    }
  });

  cmd_buf.clearColorImage(
    metallicRoughnessImage.get().get(),
    layout,
    vk::ClearColorValue{0, 0, 0, 1},
  {
    vk::ImageSubresourceRange {
      .aspectMask = vk::ImageAspectFlagBits::eColor,
      .levelCount = 1,
      .layerCount = 1
    }
  });

  cmd_buf.clearColorImage(
    normalsImage.get().get(), 
    layout, 
    vk::ClearColorValue{0, 0, 0, 1},
  {
    vk::ImageSubresourceRange {
      .aspectMask = vk::ImageAspectFlagBits::eColor,
      .levelCount = 1,
      .layerCount = 1
    }
  });

  cmd_buf.clearDepthStencilImage(
    depthImage.get().get(), 
    layout, 
    vk::ClearDepthStencilValue(1),
  {
    vk::ImageSubresourceRange {
      .aspectMask = vk::ImageAspectFlagBits::eDepth,
      .levelCount = 1,
      .layerCount = 1
    }
  });
}

void WorldRenderer::generateGBuffer(
  vk::CommandBuffer cmd_buf, uint32_t first_command, uint32_t command_count, const glm::mat4x4& glob_tm)
{
  if (!sceneMgr->getVertexBuffer())
    return;

  vk::ImageView albedoView = albedoImage.get().getView({});
  vk::ImageView metallicRoughnessView = metallicRoughnessImage.get().getView({});
  vk::ImageView normalsView = normalsImage.get().getView({});
  vk::ImageView depthView = depthImage.get().getView({});


  gbufferDrawer.run(cmd_buf,
    sceneUploader.getMatricesBuffer().buffer,
    sceneUploader.getIndirectCommandsBuffer().buffer,
    drawParamsCulledIndicesBuffer.get().buffer,
    sceneUploader.getInstancesToCommandsBuffer().buffer,
    albedoImage.get().get(),
    albedoView,
    metallicRoughnessImage.get().get(),
    metallicRoughnessView,
    normalsImage.get().get(),
    normalsView,
    depthImage.get().get(),
    depthView,
    sceneMgr->getVertexBuffer(),
    sceneMgr->getIndexBuffer(),

    sceneMgr->getImages(),
    sceneUploader.getMaterialsBuffer().buffer,

    resolution,
    first_command,
    command_count,
    glob_tm);
}

void WorldRenderer::resolveGBufferWithLights(
  vk::CommandBuffer cmd_buf, const glm::mat4x4& glob_tm, vk::Image target_image, vk::ImageView target_image_view)
{
  if (!sceneMgr->getVertexBuffer())
    return;

  

  lightGBufferResolver.run(cmd_buf,
    lights.buffer,
    lightsProjViewMatrices.get(),
    uint32_t(lightsVector.size()),
    shadowmaps.get(),
    albedoImage.get(),
    metallicRoughnessImage.get(),
    normalsImage.get(),
    depthImage.get(),
    target_image,
    target_image_view,

    resolution,
    glob_tm,
    cameraCopy.position,
    cameraCopy.zNear
  );
}

void WorldRenderer::renderWorld(
  vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_image_view)
{
  ETNA_PROFILE_GPU(cmd_buf, renderWorld);

  // draw final scene to screen
  {
    ETNA_PROFILE_GPU(cmd_buf, renderForward);

    if (!loaded) {
      return;
    }

    if (aabbsDirty) {
      recalculateAABBs(cmd_buf);
      aabbsDirty = false;
    }

    renderShadowmaps(cmd_buf);

    cullMeshes(cmd_buf, cameraCopy);

    clearAttachments(cmd_buf);

    if (sceneUploader.getPipelines().contains(PipelineType::PBR)) {
      SceneUploader::PipelineInfo pbrInfo = sceneUploader.getPipelines().find(PipelineType::PBR)->second;
      generateGBuffer(cmd_buf, pbrInfo.firstIndirectCommand, pbrInfo.commandCount, worldViewProj);
    }

    resolveGBufferWithLights(cmd_buf, worldViewProj, target_image, target_image_view);
  }
}
