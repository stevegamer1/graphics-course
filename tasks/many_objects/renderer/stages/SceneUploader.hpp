#pragma once
#include "BufferWithSize.hpp"
#include "etna/BlockingTransferHelper.hpp"
#include "etna/OneShotCmdMgr.hpp"
#include "scene/SceneManager.hpp"
#include <cstdint>
#include <glm/fwd.hpp>
#include <memory>
#include <vulkan/vulkan_structs.hpp>


enum class InstanceID : std::uint32_t { Invalid = ~std::uint32_t{0} };
enum class RelemID : std::uint32_t { Invalid = ~std::uint32_t{0} };

class SceneUploader {
public:
    SceneUploader();

    void updateScene(const SceneManager& scene_manager);
    
    // These are for binding when it's le time to draw le scene using le pbr or something
    struct PipelineInfo {
        uint32_t firstIndirectCommand;
        uint32_t commandCount;
    };

    struct MaterialInfo {
        constexpr static uint32_t ELEMS_COUNT = 3;
        std::array<TextureID, ELEMS_COUNT> textures;
        std::byte padding[16 - (sizeof(textures)) % 16];
        std::array<glm::vec4, ELEMS_COUNT> textures_factors;
    };

    const std::unordered_map<PipelineType, SceneUploader::PipelineInfo>& getPipelines() const;
    const BufferWithSize& getInstancesToCommandsBuffer() const;
    const BufferWithSize& getIndirectCommandsBuffer() const;
    const BufferWithSize& getMaterialsBuffer() const;
    const BufferWithSize& getMatricesBuffer() const;

private:
    std::unique_ptr<etna::OneShotCmdMgr> oneShotCmdMgr;
    etna::BlockingTransferHelper transferHelper;

    std::unordered_map<PipelineType, PipelineInfo> pipelines;

    BufferWithSize instancesToCommandsMap;
    BufferWithSize indirectCommands;
    BufferWithSize materials;
    BufferWithSize matrices;
};
