#include "WorldRenderer.hpp"
#include "etna/BlockingTransferHelper.hpp"
#include "etna/Buffer.hpp"
#include "etna/Etna.hpp"
#include "etna/Image.hpp"
#include "scene/SceneManager.hpp"
#include "scene_transformations/SceneInvertion.hpp"
#include "stages/CullingManager.hpp"
#include "stages/GBufferDrawer.hpp"
#include "stages/GBufferLightResolver.hpp"
#include "stages/BufferWithSize.hpp"
#include "stages/SceneUploader.hpp"

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
  // , drawParams(work_count, std::in_place_t{})
  , drawParamsCulledIndicesBuffer(work_count, std::in_place_t{})
  // , instanceMeshToIndirectCommandMap(work_count, std::in_place_t{})
  // , indirectCommandsBuffer(work_count, std::in_place_t{})
  // , indirectCommandsCountBuffer(work_count, std::in_place_t{})
  , albedoImage(work_count, std::in_place_t{})
  , metallicRoughnessImage(work_count, std::in_place_t{})
  , normalsImage(work_count, std::in_place_t{})
  , depthImage(work_count, std::in_place_t{})
  , gbufferDrawer(work_count)
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
  // invertedSceneView = invert_scene(*sceneMgr);
  sceneUploader.updateScene(*sceneMgr);
  recreateAndUploadBuffersIfNecessary();
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
    ALBEDO_FORMAT, METAL_ROUGH_FORMAT, NORMAL_FORMAT, DEPTH_FORMAT,
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

// TODO: use this instead of copypasta
void WorldRenderer::recreateBufferIfNecessary(
    etna::GpuSharedResource<BufferWithSize>& buffer,
    size_t desired_size,
    vk::BufferUsageFlags buf_usage,
    VmaMemoryUsage mem_usage,
    std::string_view name
  ) {
  buffer.iterate([desired_size, buf_usage, mem_usage, name](BufferWithSize& buffer){
    if (buffer.size == desired_size) {
      return;
    }

    buffer.buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
      .size = desired_size,
      .bufferUsage = buf_usage,
      .memoryUsage = mem_usage,
      .name = name,
    });
  });
}

void WorldRenderer::recreateDrawParamsBuffers(uint32_t count) {
  // size_t drawParamsSize = sizeof(SingleRelemDrawParams) * count;
  size_t culledIndicesSize = sizeof(uint32_t) * count;

  // drawParams.iterate([drawParamsSize](BufferWithSize& buffer) {
  //   buffer.buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
  //     .size = drawParamsSize,
  //     .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
  //     .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
  //     .name = "drawParams",
  //   });
  //   buffer.size = drawParamsSize;
  // });

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

// void WorldRenderer::recreateIndirectCommandsBuffer(uint32_t count) {
//   size_t size = sizeof(vk::DrawIndexedIndirectCommand) * count;
//   indirectCommandsBuffer.iterate([size](BufferWithSize& buffer){
//     buffer.buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
//       .size = size,
//       .bufferUsage = vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
//       .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
//       .name = "indirectCommands",
//     });
//     buffer.size = size;
//   });
// }

// void WorldRenderer::createIndirectCommandCountBuffer() {
//   size_t size = sizeof(uint32_t);
//   indirectCommandsCountBuffer.iterate([size](BufferWithSize& buffer) {
//     buffer.buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
//       .size = size,
//       .bufferUsage = vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferDst,
//       .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
//       .name = "indirectCommandCountBuffer"
//     });
//     buffer.size = size;
//   });
// }

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

// void WorldRenderer::recreateInstancesToCommandsMapBuffer(uint32_t count) {
//   size_t size = sizeof(uint32_t) * count;
//   instanceMeshToIndirectCommandMap.iterate([size](BufferWithSize& buffer) {
//     buffer.buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
//       .size = size,
//       .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
//       .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
//       .name = "instanceMeshToIndirectCommandMap"
//     });
//     buffer.size = size;
//   });
// }

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

// std::map<RelemID, std::vector<SingleRelemDrawParams>> WorldRenderer::collectRelemsDrawParamsForIndirect() {
//   std::map<RelemID, std::vector<SingleRelemDrawParams>> result;

//   auto instanceMatrices = sceneMgr->getInstanceMatrices();
//   auto instanceMeshes = sceneMgr->getInstanceMeshes();
//   auto meshes = sceneMgr->getMeshes();
//   for (std::size_t instIdx = 0; instIdx < instanceMeshes.size(); ++instIdx)
//   {
//     const auto meshIdx = instanceMeshes[instIdx];

//     for (std::size_t j = 0; j < meshes[meshIdx].relemCount; ++j)
//     {
//       const auto relemIdx = meshes[meshIdx].firstRelem + j;
//       result[static_cast<RelemID>(relemIdx)].push_back({
//         instanceMatrices[instIdx]
//       });
//     }
//   }

//   return result;
// }

// WorldRenderer::BuffersForDrawIndexedIndirectCount WorldRenderer::prepareDrawParamsBuffersOnCPU() {
//   BuffersForDrawIndexedIndirectCount result;

//   // auto relems = sceneMgr->getRenderElements();

//   // const size_t materialsCount = invertedSceneView->materials.size();
//   // for (uint32_t m = 0; m < materialsCount; ++m) {
//   //   const InvertedMaterial& material = invertedSceneView->materials[m];
//   //   const size_t relemsCount = material.relemCount;

//   //   for (uint32_t r = 0; r < relemsCount; ++r) {
//   //     uint32_t relemIndex = static_cast<uint32_t>(material.renderElements[r]);
//   //     const InvertedRelem& relem = invertedSceneView->relems[relemIndex];

//   //     result.commands.push_back(vk::DrawIndexedIndirectCommand{
//   //       .indexCount = relems[relemIndex].indexCount,
//   //       .instanceCount = static_cast<uint32_t>(relem.instances.size()),
//   //       .firstIndex = relems[relemIndex].indexOffset,
//   //       .vertexOffset = static_cast<int32_t>(relems[relemIndex].vertexOffset),
//   //       .firstInstance = static_cast<uint32_t>(relem.instances[0])
//   //     });
//   //   }
//   // }

//   result.commands.reserve(invertedSceneView->relems.size());
//   for (uint32_t r = 0; r < invertedSceneView->relems.size(); ++r) {
//     result.commands.push_back(vk::DrawIndexedIndirectCommand{
//       .indexCount = invertedSceneView->relems[r].indexCount,
//       .instanceCount = invertedSceneView->relems[r].instanceCount,
//       .firstIndex = invertedSceneView->relems[r].firstIndex,
//       .vertexOffset = invertedSceneView->relems[r].vertexOffset,
//       .firstInstance = static_cast<uint32_t>(invertedSceneView->relems[r].firstInstance)
//     });
//   }

//   return result;
// }

// WorldRenderer::BuffersForCulling WorldRenderer::prepareCullingBuffersOnCPU(const std::vector<vk::DrawIndexedIndirectCommand>& commands) {
//   BuffersForCulling result;

//   const size_t commandsCount = commands.size();
//   for (uint32_t c = 0; c < commandsCount; ++c) {

//     uint32_t instanceCount = commands[c].instanceCount;
//     for (uint32_t i = 0; i < instanceCount; ++i) {
      
//       result.instanceToIndirectCommandMap.push_back(c);
//     }
//   }

//   return result;
// }

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

  // {
  //   uint32_t desiredInstancesToCommandsMapEntryCount = instancesCount;
  //   uint32_t currentCount = uint32_t(instanceMeshToIndirectCommandMap.get().size / sizeof(uint32_t));
  //   if (currentCount < desiredInstancesToCommandsMapEntryCount) {
  //     recreateInstancesToCommandsMapBuffer(desiredInstancesToCommandsMapEntryCount);
  //   }
  // }

  // {
  //   uint32_t desiredCommandsCount = relemsCount;
  //   uint32_t currentCount = uint32_t(indirectCommandsBuffer.get().size / sizeof(vk::DrawIndexedIndirectCommand));
  //   if (currentCount < desiredCommandsCount) {
  //     recreateIndirectCommandsBuffer(desiredCommandsCount);
  //   }
  // }

  // {
  //   if (indirectCommandsCountBuffer.get().size == 0) {
  //     createIndirectCommandCountBuffer();
  //   }
  // }

  {
    uint32_t desiredAABBCount = relemsCount;
    uint32_t currentCount = uint32_t(aabbBuffer.size / sizeof(AABB));
    if (currentCount < desiredAABBCount) {
      recreateAABBBuffer(desiredAABBCount);
      markAABBsDirty();
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
          vk::ImageUsageFlagBits::eTransferDst  // Need to remove eTransferDst bit when clearing will be done by LoadOp of G-Buffer generator, but that's after implementing bindless.
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
          vk::ImageUsageFlagBits::eTransferDst  // Need to remove eTransferDst bit when clearing will be done by LoadOp of G-Buffer generator, but that's after implementing bindless.
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
          vk::ImageUsageFlagBits::eTransferDst  // Need to remove eTransferDst bit when clearing will be done by LoadOp of G-Buffer generator, but that's after implementing bindless.
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
          vk::ImageUsageFlagBits::eTransferDst  // Need to remove eTransferDst bit when clearing will be done by LoadOp of G-Buffer generator, but that's after implementing bindless.
      });
    });
    depthImageResolution = resolution;
  }

  
  // BuffersForDrawIndexedIndirectCount cpuBuffersForIndirectDraw = prepareDrawParamsBuffersOnCPU();
  // BuffersForCulling cullingBuffers = prepareCullingBuffersOnCPU(cpuBuffersForIndirectDraw.commands);

  // drawParams.iterate([this](BufferWithSize& buffer){
  //   transferHelper.uploadBuffer(
  //     *oneShotCommands, 
  //     buffer.buffer, 0, 
  //     std::span<const SingleRelemDrawParams>(invertedSceneView->instances));
  // });
  
  // instanceMeshToIndirectCommandMap.iterate([this, cullingBuffers](BufferWithSize& buffer){
  //   transferHelper.uploadBuffer(
  //     *oneShotCommands, 
  //     buffer.buffer, 0,
  //     std::span<const uint32_t>(cullingBuffers.instanceToIndirectCommandMap));
  // });

  // indirectCommandsBuffer.iterate([this, cpuBuffersForIndirectDraw](BufferWithSize& buffer) {
  //   transferHelper.uploadBuffer(
  //     *oneShotCommands, 
  //     buffer.buffer, 0, 
  //     std::span<const vk::DrawIndexedIndirectCommand>(cpuBuffersForIndirectDraw.commands));
  // });
  
  // std::vector<uint32_t> countVector{uint32_t(cpuBuffersForIndirectDraw.commands.size())};
  // indirectCommandsCountBuffer.iterate([this, countVector](BufferWithSize& buffer) {
  //   transferHelper.uploadBuffer(
  //     *oneShotCommands, 
  //     buffer.buffer, 0, 
  //     std::span<const uint32_t>(countVector));
  // });
  
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
    sceneUploader.debugInstancesToCommandsMapOnGPU.buffer,
    uint32_t(sceneMgr->getInstanceMeshes().size()), frustum);
}

// Will remove when I add bindless.
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

  // When I add bindless, I will not pass these.
  TextureID pbrBaseColorId = sceneUploader.debugMaterialsOnCPU[first_command + 3].textures[0];
  TextureID pbrMetallicRoughnessId = sceneUploader.debugMaterialsOnCPU[first_command + 3].textures[1];
  TextureID pbrNormalsId = sceneUploader.debugMaterialsOnCPU[first_command + 3].textures[2];

  gbufferDrawer.run(cmd_buf,
    sceneUploader.getMatricesBuffer().buffer,
    sceneUploader.getIndirectCommandsBuffer().buffer,
    drawParamsCulledIndicesBuffer.get().buffer,
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

    sceneMgr->getImages()[static_cast<uint32_t>(pbrBaseColorId)],
    sceneMgr->getImages()[static_cast<uint32_t>(pbrMetallicRoughnessId)],
    sceneMgr->getImages()[static_cast<uint32_t>(pbrNormalsId)],

    sceneUploader.debugMaterialsOnCPU[first_command].textures_factors[0],
    sceneUploader.debugMaterialsOnCPU[first_command].textures_factors[1],

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
    uint32_t(lightsVector.size()),
    albedoImage.get(),
    metallicRoughnessImage.get(),
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

    if (aabbsDirty) {
      recalculateAABBs(cmd_buf);
      aabbsDirty = false;
    }

    cullMeshes(cmd_buf, cameraCopy);

    // generateGBuffer(cmd_buf, worldViewProj);

    // resolveGBufferWithLights(cmd_buf, worldViewProj, target_image, target_image_view);

    clearAttachments(cmd_buf);

    // for (uint32_t p = 0; p < invertedSceneView->pipelines.size(); ++p) {
    //   if (invertedSceneView->pipelines[p].type != PipelineType::PBR) {
    //     continue;  // TODO: store objects by type, not type inside object
    //   }

    //   for (uint32_t m = 0; m < invertedSceneView->pipelines[p].materials.size(); ++m) {
    //     MaterialID materialId = invertedSceneView->pipelines[p].materials[m];
    //     const InvertedMaterial& material = invertedSceneView->materials[static_cast<uint32_t>(materialId)];

    //     generateGBuffer(cmd_buf, material, worldViewProj);
    //   }
    // }

    if (sceneUploader.getPipelines().contains(PipelineType::PBR)) {
      SceneUploader::PipelineInfo pbrInfo = sceneUploader.getPipelines().find(PipelineType::PBR)->second;
      generateGBuffer(cmd_buf, pbrInfo.firstIndirectCommand, pbrInfo.commandCount, worldViewProj);
    }

    resolveGBufferWithLights(cmd_buf, worldViewProj, target_image, target_image_view);
  }
}
