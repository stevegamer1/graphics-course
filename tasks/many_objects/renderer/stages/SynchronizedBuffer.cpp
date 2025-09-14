#include "SynchronizedBuffer.hpp"
#include "etna/GlobalContext.hpp"
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>


void SynchronizedBuffer::syncBeforeUsage(BufferSyncUsage usage, vk::CommandBuffer cmd_buf) {
    if (previousUsages.stageFlags) {
        vk::BufferMemoryBarrier2 barrier {
            .srcStageMask = previousUsages.stageFlags,
            .srcAccessMask = previousUsages.accessFlags,
            .dstStageMask = usage.stageFlags,
            .dstAccessMask = usage.accessFlags,
            .srcQueueFamilyIndex = etna::get_context().getQueueFamilyIdx(),
            .dstQueueFamilyIndex = etna::get_context().getQueueFamilyIdx(),
            .buffer = buffer.get(),
            .size = vk::WholeSize
        };
        vk::DependencyInfo depInfo {
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &barrier
        };
        cmd_buf.pipelineBarrier2(depInfo);
    } else {
        assert(!bool(previousUsages.accessFlags));
    }

    // todo: suboptimal. Need only synchronize with latest write accesses?
    previousUsages.accessFlags |= usage.accessFlags;
    previousUsages.stageFlags |= usage.stageFlags;
}

void SynchronizedBuffer::resetAccumulatedUsage() {
    previousUsages.accessFlags = vk::AccessFlags2{};
    previousUsages.stageFlags = vk::PipelineStageFlags2{};
}
