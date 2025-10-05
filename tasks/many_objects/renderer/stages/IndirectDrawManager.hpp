#pragma once
#include "etna/Buffer.hpp"
#include "etna/GraphicsPipeline.hpp"
#include "etna/Image.hpp"
#include <etna/Vulkan.hpp>
#include <glm/fwd.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>
#include <glm/ext/matrix_float4x4.hpp>


class IndirectDrawManager
{
public:
  IndirectDrawManager() = default;

  IndirectDrawManager(const IndirectDrawManager&) = delete;
  IndirectDrawManager(IndirectDrawManager&&) = delete;
  IndirectDrawManager& operator=(const IndirectDrawManager&) = delete;
  IndirectDrawManager& operator=(IndirectDrawManager&&) = delete;

  void loadShader();

  void createPipeline(vk::Format swapchain_format, etna::VertexByteStreamFormatDescription vertex_format_description);

  void createDepthImage(glm::uvec2 resolution);

  // Synchronizes buffers by itself.
  void run(
    vk::CommandBuffer cmd_buf,

    etna::Buffer& draw_params,
    etna::Buffer& indirect_commands,
    etna::Buffer& draw_params_indices,
    etna::Buffer& commands_count,

    vk::Image target_image,
    vk::ImageView target_image_view,
    vk::Buffer vertex_buffer,
    vk::Buffer index_buffer,

    glm::uvec2 resolution,
    uint32_t relems_count,
    glm::mat4 proj_view);

private:
  struct PushConstants
  {
    glm::mat4x4 projView;
  };

  etna::Image mainViewDepth;
  const vk::Format DEPTH_ATTACHMENT_FORMAT = vk::Format::eD32Sfloat;

  etna::GraphicsPipeline pipeline;
  const char* PROGRAM_NAME = "static_mesh_program";
};
