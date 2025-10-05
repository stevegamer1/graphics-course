#pragma once
#include "etna/Buffer.hpp"
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

  struct Frustum {
    glm::vec4 near;
    glm::vec4 far;
    glm::vec4 left;
    glm::vec4 right;
    glm::vec4 top;
    glm::vec4 bottom;
  };

  // vfov is vertical in degrees, aspect is width/height.
  static Frustum getFrustum(float vfov, float near, float far, float aspect);

  // Synchronizes buffers by itself.
  void run(
    vk::CommandBuffer cmd_buf,

    etna::Buffer& draw_params,
    etna::Buffer& aabbs,
    etna::Buffer& indirect_commands,
    etna::Buffer& draw_params_indices,
    etna::Buffer& command_indices,

    uint32_t instance_count,
    const Frustum& camera_frustum);

private:
  struct PushConstants
  {
    Frustum camera_frustum;
    uint32_t instances_to_cull_count;
  };

  etna::ComputePipeline pipeline;
  const char* PROGRAM_NAME = "cull_program";
};
