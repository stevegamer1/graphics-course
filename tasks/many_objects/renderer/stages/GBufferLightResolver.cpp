#include "GBufferLightResolver.hpp"
#include <cstdint>
#include <glm/ext/scalar_constants.hpp>
#include <glm/fwd.hpp>
#include <vulkan/vulkan_enums.hpp>
#include "etna/Buffer.hpp"
#include "etna/DescriptorSet.hpp"
#include "etna/Etna.hpp"
#include "etna/GlobalContext.hpp"
#include "etna/Image.hpp"
#include "etna/PipelineManager.hpp"
#include "etna/RenderTargetStates.hpp"


GBufferLightResolver::GBufferLightResolver()
  : oneShotCommands{etna::get_context().createOneShotCmdMgr()}
  , transferHelper(etna::BlockingTransferHelper::CreateInfo{ .stagingSize = SPHERE_ROWS * SPHERE_COLUMNS * sizeof(glm::vec4) }) {
  sphereVertices.resize(SPHERE_ROWS * SPHERE_COLUMNS);

  float pi = glm::pi<float>();
  for (int row = 0; row < SPHERE_ROWS; ++row) {
    float verticalAngle = (float(row) / (SPHERE_ROWS - 1) - 0.5f) * pi;

    float y = std::sin(verticalAngle);
    float xAndZFactor = std::cos(verticalAngle);

    for (int column = 0; column < SPHERE_COLUMNS; ++column) {
      float angleAroundVerticalAxis = float(column) / (SPHERE_COLUMNS - 1) * 2.0f * pi;

      float x = std::sin(angleAroundVerticalAxis) * xAndZFactor;
      float z = std::cos(angleAroundVerticalAxis) * xAndZFactor;

      sphereVertices[row * SPHERE_COLUMNS + column] = glm::vec4(x, y, z, 1.0f);
    }
  }

  const int quads = (SPHERE_ROWS - 1) * (SPHERE_COLUMNS - 1);
  const int triangles = 2 * quads;
  const int indicesCount = triangles * 3;
  sphereIndices.resize(indicesCount);

  for (int row = 0; row < SPHERE_ROWS - 1; ++row) {
    for (int column = 0; column < SPHERE_COLUMNS - 1; ++column) {
      int bottomLeft = row * SPHERE_COLUMNS + column;
      int bottomRight = bottomLeft + 1;
      int topLeft = bottomLeft + SPHERE_COLUMNS;
      int topRight = bottomRight + SPHERE_COLUMNS;

      int quadIndex = row * (SPHERE_COLUMNS - 1) + column;
      int triangleIndex = quadIndex * 2;
      int firstIndexIndex = triangleIndex * 3;
      // Lower triangle.
      sphereIndices[firstIndexIndex + 0] = topLeft;
      sphereIndices[firstIndexIndex + 1] = bottomLeft;
      sphereIndices[firstIndexIndex + 2] = bottomRight;

      // Upper triangle.
      sphereIndices[firstIndexIndex + 3] = topLeft;
      sphereIndices[firstIndexIndex + 4] = bottomRight;
      sphereIndices[firstIndexIndex + 5] = topRight;
    }
  }

  

  defaultSampler = etna::Sampler(etna::Sampler::CreateInfo{.name = "default_sampler"});
}

GBufferLightResolver::Light::Light(glm::vec3 pos, float intensity, glm::vec3 color, bool casts_shadows, LightType type)
: posAndIntensity(pos, intensity)
, color(color)
, padding23_castsShadows1_lightType8(
    0 |
    (casts_shadows ? CASTS_SHADOWS_MASK : uint32_t{0}) |
    uint32_t{static_cast<uint8_t>(type)}
  )
{}

bool GBufferLightResolver::Light::castsShadows() const {
  return (padding23_castsShadows1_lightType8 & CASTS_SHADOWS_MASK) != uint32_t{0};
}

GBufferLightResolver::Light::LightType GBufferLightResolver::Light::getType() const {
  return LightType{static_cast<uint8_t>(padding23_castsShadows1_lightType8 & LIGHT_TYPE_MASK)};
}

void GBufferLightResolver::allocateAndFillResources() {
  size_t vertexBufferSize = sizeof(Vertex) * sphereVertices.size();
  sphereVertexBuffer.buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = vertexBufferSize,
    .bufferUsage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "light sphere vertices buffer",
  });
  sphereVertexBuffer.size = vertexBufferSize;

  size_t indexBufferSize = sizeof(uint32_t) * sphereIndices.size();
  sphereIndexBuffer.buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = indexBufferSize,
    .bufferUsage = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "light sphere indices buffer",
  });
  sphereIndexBuffer.size = indexBufferSize;

  transferHelper.uploadBuffer(*oneShotCommands, sphereVertexBuffer.buffer, 0, std::span<const Vertex>(sphereVertices));
  transferHelper.uploadBuffer(*oneShotCommands, sphereIndexBuffer.buffer, 0, std::span<const uint32_t>(sphereIndices));
}

void GBufferLightResolver::loadShader()
{
  etna::create_program(
    PROGRAM_NAME,
    {MANY_OBJECTS_RENDERER_SHADERS_ROOT "light_from_gbuffer.frag.spv",
     MANY_OBJECTS_RENDERER_SHADERS_ROOT "light_from_gbuffer.vert.spv"});
}

void GBufferLightResolver::createPipeline(
  vk::Format output_color_format)
{
  etna::VertexShaderInputDescription sceneVertexInputDesc{
    .bindings = {etna::VertexShaderInputDescription::Binding{
      .byteStreamDescription = etna::VertexByteStreamFormatDescription{
        .stride = sizeof(Vertex),
        .attributes={{
          .format=vk::Format::eR32G32B32A32Sfloat,
          .offset=0
        }}
      },
    }},
  };

  auto& pipelineManager = etna::get_context().getPipelineManager();

  pipeline = {};
  pipeline = pipelineManager.createGraphicsPipeline(
    PROGRAM_NAME,
    etna::GraphicsPipeline::CreateInfo{
      .vertexShaderInput = sceneVertexInputDesc,
      .rasterizationConfig =
        vk::PipelineRasterizationStateCreateInfo{
          .polygonMode = vk::PolygonMode::eFill,
          .cullMode = vk::CullModeFlagBits::eFront,  // We want to draw the inside of the sphere.
          .frontFace = vk::FrontFace::eCounterClockwise,
          .lineWidth = 1.f,
        },
      .blendingConfig = {
        .attachments ={
          vk::PipelineColorBlendAttachmentState{
            .blendEnable = vk::True,
            .srcColorBlendFactor = vk::BlendFactor::eOne,
            .dstColorBlendFactor = vk::BlendFactor::eOne,
            .colorBlendOp = vk::BlendOp::eAdd,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
              vk::ColorComponentFlagBits::eB
          }
        },
        .logicOp = {}
      },
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {output_color_format}
        },
    });
}

void GBufferLightResolver::createShadowmapsDescriptorSet(std::span<etna::Image> lights_shadowmaps) {
  auto programInfo = etna::get_shader_program(PROGRAM_NAME);

  std::vector<etna::Binding> shadowMapBindings;
  for (uint32_t i = 0; i < lights_shadowmaps.size(); ++i) {
    shadowMapBindings.push_back(etna::Binding{0, lights_shadowmaps[i].genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal), i});
  }

  shadowmapSet = etna::create_persistent_descriptor_set(
    programInfo.getDescriptorLayoutId(1),
    shadowMapBindings
  );
}

void GBufferLightResolver::run(
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
  )
{
  vk::PipelineStageFlags2 graphicsPipelineStage =
      vk::PipelineStageFlagBits2::eVertexShader |
      vk::PipelineStageFlagBits2::eFragmentShader |
      vk::PipelineStageFlagBits2::eDrawIndirect;
  vk::AccessFlags2 graphicsAccess =
      vk::AccessFlagBits2::eUniformRead |
      vk::AccessFlagBits2::eShaderStorageRead |
      vk::AccessFlagBits2::eIndirectCommandRead;
  etna::set_state(cmd_buf, lights.get(), graphicsPipelineStage, graphicsAccess);
  etna::set_state(cmd_buf, shadow_lights_viewproj_matrices.get(), graphicsPipelineStage, graphicsAccess);

  etna::set_state(
    cmd_buf,
    albedo_image.get(),
    vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor
  );

  etna::set_state(
    cmd_buf,
    metallic_roughness_image.get(),
    vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor
  );

  etna::set_state(
    cmd_buf,
    normals_image.get(),
    vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor
  );

  etna::set_state(
    cmd_buf,
    depth_image.get(),
    vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eDepth
  );

  for (uint32_t i = 0; i < lights_shadowmaps.size(); ++i) {
    etna::set_state(
      cmd_buf,
      lights_shadowmaps[i].get(),
      vk::PipelineStageFlagBits2::eFragmentShader,
      vk::AccessFlagBits2::eShaderSampledRead,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eDepth
    );
  }

  etna::flush_barriers(cmd_buf);

  {
    etna::RenderTargetState renderTargets(
      cmd_buf,
      {{0, 0}, {resolution.x, resolution.y}},
      {
        {.image = color_image, .view = color_image_view}
      },
      {});

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.getVkPipeline());

    cmd_buf.bindVertexBuffers(0, {sphereVertexBuffer.buffer.get()}, {0});
    cmd_buf.bindIndexBuffer(sphereIndexBuffer.buffer.get(), 0, vk::IndexType::eUint32);

    {
      auto programInfo = etna::get_shader_program(PROGRAM_NAME);

      auto set = etna::create_descriptor_set(
        programInfo.getDescriptorLayoutId(0),
        cmd_buf,
        {
          etna::Binding{0, lights.genBinding()},
          etna::Binding{1, albedo_image.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
          etna::Binding{2, metallic_roughness_image.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
          etna::Binding{3, normals_image.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
          etna::Binding{4, depth_image.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
          etna::Binding{5, shadow_lights_viewproj_matrices.genBinding()},
        });

      cmd_buf.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, pipeline.getVkPipelineLayout(), 0, {set.getVkSet(), shadowmapSet.getVkSet()}, {});
    }

    PushConstants pushConst{
      .projView = proj_view,
      .wCamPos = glm::vec4(cam_pos, 1.0f),
      .resolution = resolution,
      .camNear = cam_near
    };
    cmd_buf.pushConstants<PushConstants>(
      pipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, {pushConst});
    
    cmd_buf.drawIndexed(
      uint32_t(sphereIndices.size()),
      lights_count,
      0,
      0,
      0);
  }
}
