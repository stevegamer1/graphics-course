#pragma once
#include "etna/Buffer.hpp"
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>


struct BufferWithSize {
    etna::Buffer buffer;
    size_t size;
};
