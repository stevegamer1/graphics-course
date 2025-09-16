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

    static constexpr vk::AccessFlags2 ANY_HOST_ACCESS =
        vk::AccessFlagBits2::eHostWrite | vk::AccessFlagBits2::eHostRead;

    static constexpr vk::AccessFlags2 DEVICE_WRITE_ACCESS =
        vk::AccessFlagBits2::eMemoryWrite |
        vk::AccessFlagBits2::eShaderWrite |
        vk::AccessFlagBits2::eTransferWrite |
        vk::AccessFlagBits2::eColorAttachmentWrite |
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
        vk::AccessFlagBits2::eShaderStorageWrite;
};
