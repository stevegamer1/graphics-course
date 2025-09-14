#pragma once
#include "etna/Buffer.hpp"
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>


struct BufferSyncUsage {
    vk::PipelineStageFlags2 stageFlags;
    vk::AccessFlags2 accessFlags;
};

class SynchronizedBuffer {
public:
    size_t current_size;
    etna::Buffer buffer;

    void syncBeforeUsage(BufferSyncUsage usage, vk::CommandBuffer cmd_buf);

    // Call after submission of command buffer.
    void resetAccumulatedUsage();
private:
    BufferSyncUsage previousUsages;
};
