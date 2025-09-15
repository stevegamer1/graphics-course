#pragma once
#include "SynchronizedBuffer.hpp"
#include "etna/ComputePipeline.hpp"
#include <etna/Vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>
#include <glm/ext/matrix_float4x4.hpp>


class AABBCalculator
{
public:
  AABBCalculator() = default;

  AABBCalculator(const AABBCalculator&) = delete;
  AABBCalculator(AABBCalculator&&) = delete;
  AABBCalculator& operator=(const AABBCalculator&) = delete;
  AABBCalculator& operator=(AABBCalculator&&) = delete;

  void loadShader();

  void createPipeline();

  // Synchronizes buffers by itself.
  void run(
    vk::CommandBuffer cmd_buf,

    SynchronizedBuffer& aabbs,
    SynchronizedBuffer& indirect_commands,

    etna::Buffer& vertex_buffer,
    etna::Buffer& index_buffer,

    uint32_t relem_count);

private:
  struct PushConstants
  {
    uint32_t abbs_count;
    uint32_t vertex_size_in_bytes;
  };

  etna::ComputePipeline pipeline;
  const char* PROGRAM_NAME = "aabb_program";
};
