#pragma once
#include "SynchronizedBuffer.hpp"
#include "etna/ComputePipeline.hpp"
#include <etna/Vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>
#include <glm/ext/matrix_float4x4.hpp>


class CullingManager
{
public:
  CullingManager() = default;

  CullingManager(const CullingManager&) = delete;
  CullingManager(CullingManager&&) = delete;
  CullingManager& operator=(const CullingManager&) = delete;
  CullingManager& operator=(CullingManager&&) = delete;

  void loadShader();

  void createPipeline();

  // Synchronizes buffers by itself.
  void run(
    vk::CommandBuffer cmd_buf,

    SynchronizedBuffer& draw_params,
    SynchronizedBuffer& aabbs,
    SynchronizedBuffer& indirect_commands,
    SynchronizedBuffer& draw_params_indices,
    SynchronizedBuffer& command_indices,

    uint32_t instance_count,
    glm::mat4 proj_view);

private:
  struct PushConstants
  {
    glm::mat4x4 projView;
  };

  etna::ComputePipeline pipeline;
  const char* PROGRAM_NAME = "cull_program";

  void zeroOutInstanceCountsInCommands(
    vk::CommandBuffer cmd_buf, SynchronizedBuffer& indirect_commands);
};
