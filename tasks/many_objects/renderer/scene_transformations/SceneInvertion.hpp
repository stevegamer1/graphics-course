#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include "scene/SceneManager.hpp"


enum class InstanceID : std::uint32_t { Invalid = ~std::uint32_t{0} };
enum class RelemID : std::uint32_t { Invalid = ~std::uint32_t{0} };

struct InvertedPipeline {
    PipelineType type;
    std::vector<MaterialID> materials;  // TODO: store materials directly inside pipeline.
};

struct InvertedMaterial {
    std::vector<TextureID> textures;
    std::vector<glm::vec3> texturesScalarMultipliers;
    RelemID firstRelem;
    uint32_t relemCount;
};

struct InvertedRelem {
    InstanceID firstInstance;
    uint32_t instanceCount = 0;
    int32_t vertexOffset;
    uint32_t firstIndex;
    uint32_t indexCount = 0;
};

struct SingleRelemDrawParams {
  glm::mat4x4 model;
};

struct InvertedSceneView {
    std::vector<InvertedPipeline> pipelines;
    std::vector<InvertedMaterial> materials;
    std::vector<InvertedRelem> relems;
    std::vector<SingleRelemDrawParams> instances;
};

std::unique_ptr<InvertedSceneView> invert_scene(const SceneManager& scene_manager);
