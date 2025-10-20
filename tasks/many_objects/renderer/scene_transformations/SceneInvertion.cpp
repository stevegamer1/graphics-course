#include "SceneInvertion.hpp"
#include "scene/SceneManager.hpp"
#include <glm/fwd.hpp>
#include <memory>


std::unique_ptr<InvertedSceneView> invert_scene(const SceneManager& scene_manager) {
    std::unique_ptr<InvertedSceneView> result = std::make_unique<InvertedSceneView>();

    std::span<const Material> materials = scene_manager.getMaterials();
    std::span<const RenderElement> relems = scene_manager.getRenderElements();
    std::span<const Mesh> meshes = scene_manager.getMeshes();
    std::span<const uint32_t> instanceMeshes = scene_manager.getInstanceMeshes();
    std::span<const glm::mat4x4> instanceMatrices = scene_manager.getInstanceMatrices();

    result->pipelines = {InvertedPipeline{
        .type = PipelineType::PBR,
        .materials{}
    }};

    // result->materials.reserve(materials.size());

    // for (uint32_t i = 0; i < materials.size(); ++i) {
    //     result->pipelines[0].materials[i] = static_cast<MaterialID>(i);
    //     InvertedMaterial& invMat = result->materials.emplace_back();
    //     invMat.textures = materials[i].textures;
    // }

    // // TODO: create blank material here. Magenta and black checker board? :)

    // for (uint32_t i = 0; i < relems.size(); ++i) {
    //     assert(!materials.empty());
    //     MaterialID relemMaterial = relems[i].material != MaterialID::Invalid ? relems[i].material : MaterialID{0};
    //     result->materials[static_cast<uint32_t>(relemMaterial)].renderElements.push_back(RelemID{i});
    //     result->relems.emplace_back();
    // }

    // // result->instances.reserve(instanceMatrices.size());
    // // for (uint32_t i = 0; i < instanceCount; ++i) {
    // //     result->instances.push_back({instanceMatrices[i]});  // NOOOOO, TODO: remove unnecessary copy.
        
    // //     uint32_t begin = meshes[instanceMeshes[i]].firstRelem;
    // //     uint32_t end = begin + meshes[instanceMeshes[i]].relemCount;
    // //     for (uint32_t r = begin; r < end; ++r) {
    // //         result->relems[r].instances.push_back(InstanceID{i});
    // //     }
    // // }
    // std::vector<std::vector<InstanceID>> relemToInstances(result->relems.size());
    // result->instances.reserve(instanceMeshes.size());
    // for (uint32_t i = 0; i < instanceCount; ++i) {
    //     const uint32_t firstRelem = meshes[instanceMeshes[i]].firstRelem;
    //     const uint32_t relemCount = meshes[instanceMeshes[i]].relemCount;
    //     for (uint32_t r = firstRelem; r < relemCount; ++r) {
    //         relemToInstances[r].push_back(InstanceID{i});
    //     }
    // }

    // for (uint32_t r = 0; r < relems.size(); ++r) {
    //     result->relems[r].firstInstance = InstanceID{static_cast<uint32_t>(result->instances.size())};
    //     result->relems[r].instanceCount = static_cast<uint32_t>(relemToInstances[r].size());
    //     result->relems[r].firstIndex = relems[r].indexOffset;
    //     result->relems[r].indexCount = relems[r].indexCount;
    //     result->relems[r].vertexOffset = relems[r].vertexOffset;

    //     for (uint32_t i = 0; i < relemToInstances[r].size(); ++i) {
    //         uint32_t index = static_cast<uint32_t>(relemToInstances[r][i]);
    //         result->instances.push_back({instanceMatrices[index]});
    //     }
    // }


    // add all (1) pipelines

    // for each pipeline, add all materials that belong to it
    
    // for each material, add all render elements that belong to it

    // for each render element, add all instances that belong to it

    for (uint32_t m = 0; m < materials.size(); ++m) {
        const Material& material = materials[m];

        result->materials.emplace_back();
        result->materials.back().textures = material.textures;
        result->materials.back().texturesScalarMultipliers = material.texturesScalarMultipliers;
        result->pipelines[static_cast<uint32_t>(material.pipeline)].materials.push_back(MaterialID{m});
    }

    // TODO: create blank material here. Magenta and black checker board? :)
    std::vector<std::vector<RelemID>> materialToRelemMap(result->materials.size());
    for (uint32_t r = 0; r < relems.size(); ++r) {
        assert(!materials.empty());
        MaterialID relemMaterial = relems[r].material != MaterialID::Invalid ? relems[r].material : MaterialID{0};
        materialToRelemMap[static_cast<uint32_t>(relemMaterial)].push_back(RelemID{r});
    }

    std::vector<std::vector<InstanceID>> relemToInstanceMap(relems.size());
    for (uint32_t i = 0; i < instanceMeshes.size(); ++i) {
        const Mesh& mesh = meshes[instanceMeshes[i]];
        for (uint32_t r = mesh.firstRelem; r < mesh.firstRelem + mesh.relemCount; ++r) {
            relemToInstanceMap[r].push_back(InstanceID{i});
        }
    }

    for (uint32_t m = 0; m < materialToRelemMap.size(); ++m) {
        result->materials[m].firstRelem = RelemID{static_cast<uint32_t>(result->relems.size())};
        result->materials[m].relemCount = static_cast<uint32_t>(materialToRelemMap[m].size());
        
        for (uint32_t r = 0; r < materialToRelemMap[m].size(); ++r) {
            uint32_t oldRelemIndex = static_cast<uint32_t>(materialToRelemMap[m][r]);
            const RenderElement& relem = relems[oldRelemIndex];
            const std::vector<InstanceID>& relemInstances = relemToInstanceMap[oldRelemIndex];

            InvertedRelem& invRelem = result->relems.emplace_back();
            invRelem.firstIndex = relem.indexOffset;
            invRelem.indexCount = relem.indexCount;
            invRelem.vertexOffset = relem.vertexOffset;
            invRelem.firstInstance = InstanceID{static_cast<uint32_t>(result->instances.size())};
            invRelem.instanceCount = static_cast<uint32_t>(relemInstances.size());

            for(uint32_t i = 0; i < relemInstances.size(); ++i) {
                result->instances.emplace_back(
                    instanceMatrices[static_cast<uint32_t>(relemInstances[i])]
                );
            }
        }
    }

    return result;
}
