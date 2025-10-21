#include "SceneUploader.hpp"
#include "BufferWithSize.hpp"
#include "etna/BlockingTransferHelper.hpp"
#include "etna/Buffer.hpp"
#include "etna/GlobalContext.hpp"
#include "scene/SceneManager.hpp"
#include <cstdint>
#include <glm/fwd.hpp>
#include <span>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_structs.hpp>


SceneUploader::SceneUploader()
: oneShotCmdMgr{etna::get_context().createOneShotCmdMgr()}
, transferHelper{etna::BlockingTransferHelper::CreateInfo{ .stagingSize = 1024 }}
{}

void SceneUploader::updateScene(const SceneManager& scene_manager) {
    pipelines.clear();
    std::vector<vk::DrawIndexedIndirectCommand> indirectCommandsVector;
    std::vector<MaterialInfo> materialsVector;
    std::vector<glm::mat4x4> instanceMatricesVector;
    std::vector<uint32_t> instancesToCommandsMapCPU;

    {
        std::span<const Material> sceneMaterials = scene_manager.getMaterials();
        std::span<const RenderElement> sceneRelems = scene_manager.getRenderElements();
        std::span<const uint32_t> sceneInstanceMeshes = scene_manager.getInstanceMeshes();
        std::span<const glm::mat4x4> sceneInstanceMatrices = scene_manager.getInstanceMatrices();
        std::span<const Mesh> sceneMeshes = scene_manager.getMeshes();
        std::unordered_map<PipelineType, std::vector<MaterialID>> pipelineMaterials;
        std::vector<std::vector<RelemID>> materialRelems(sceneMaterials.size());
        std::vector<std::vector<InstanceID>> relemInstances(sceneRelems.size());
    
        for (uint32_t m = 0; m < sceneMaterials.size(); ++m) {
            pipelineMaterials[sceneMaterials[m].pipeline].push_back(MaterialID{m});
        }
    
        for (uint32_t r = 0; r < sceneRelems.size(); ++r) {
            materialRelems[static_cast<uint32_t>(sceneRelems[r].material)].push_back(RelemID{r});
        }
    
        for (uint32_t im = 0; im < sceneInstanceMeshes.size(); ++im) {
            uint32_t firstRelem = sceneMeshes[sceneInstanceMeshes[im]].firstRelem;
            uint32_t relemCount = sceneMeshes[sceneInstanceMeshes[im]].relemCount;
            for (uint32_t r = firstRelem; r < firstRelem + relemCount; ++r) {
                relemInstances[r].push_back(InstanceID{im});
            }
        }
        
        for (std::pair<const PipelineType, std::vector<MaterialID>>& pipeline : pipelineMaterials) {
            pipelines[pipeline.first].firstIndirectCommand = static_cast<uint32_t>(indirectCommandsVector.size());
    
            for (uint32_t m = 0; m < pipeline.second.size(); ++m) {
                uint32_t materialIndex = static_cast<uint32_t>(pipeline.second[m]);

                for (uint32_t r = 0; r < materialRelems[materialIndex].size(); ++r) {
                    uint32_t relemIndex = static_cast<uint32_t>(materialRelems[materialIndex][r]);

                    indirectCommandsVector.push_back(vk::DrawIndexedIndirectCommand{
                        .indexCount = sceneRelems[relemIndex].indexCount,
                        .instanceCount = static_cast<uint32_t>(relemInstances[relemIndex].size()),
                        .firstIndex = sceneRelems[relemIndex].indexOffset,
                        .vertexOffset = static_cast<int32_t>(sceneRelems[relemIndex].vertexOffset),
                        .firstInstance = static_cast<uint32_t>(instanceMatricesVector.size())
                    });
    
                    // add matrices
                    for (uint32_t i = 0; i < relemInstances[relemIndex].size(); ++i) {
                        instanceMatricesVector.push_back(sceneInstanceMatrices[i]);
                        instancesToCommandsMapCPU.push_back(static_cast<uint32_t>(indirectCommandsVector.size() - 1));
                    }
    
                    // write material
                    MaterialInfo newMaterial;
                    assert(sceneMaterials[materialIndex].textures.size() == sceneMaterials[materialIndex].texturesScalarMultipliers.size());
                    for (uint32_t e = 0; e < sceneMaterials[materialIndex].textures.size(); ++e) {
                        newMaterial.textures[e] = sceneMaterials[materialIndex].textures[e];
                        newMaterial.textures_factors[e] =
                            glm::vec4(sceneMaterials[materialIndex].texturesScalarMultipliers[e], 1.0f);
                    }
                    materialsVector.push_back(newMaterial);
                }
            }

            pipelines[pipeline.first].commandCount = static_cast<uint32_t>(indirectCommandsVector.size()) - pipelines[pipeline.first].firstIndirectCommand;
        }
    }

    size_t desiredIndirectCommandsSize = indirectCommandsVector.size() * sizeof(vk::DrawIndexedIndirectCommand);
    if (indirectCommands.size != desiredIndirectCommandsSize) {
        indirectCommands.buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
            .size = desiredIndirectCommandsSize,
            .bufferUsage = vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .name = "Indirect command buffer"
        });
        indirectCommands.size = desiredIndirectCommandsSize;
    }

    size_t desiredMaterialsSize = materialsVector.size() * sizeof(MaterialInfo);
    if (materials.size != desiredMaterialsSize) {
        materials.buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
            .size = desiredMaterialsSize,
            .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .name = "Materials buffer"
        });
        materials.size = desiredMaterialsSize;
    }

    size_t desiredMatricesSize = instanceMatricesVector.size() * sizeof(glm::mat4x4);
    if (matrices.size != desiredMatricesSize) {
        matrices.buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
            .size = desiredMatricesSize,
            .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .name = "Transformation matrices buffer"
        });
        matrices.size = desiredMatricesSize;
    }

    transferHelper.uploadBuffer(*oneShotCmdMgr, indirectCommands.buffer, 0, std::span<const vk::DrawIndexedIndirectCommand>{indirectCommandsVector});
    transferHelper.uploadBuffer(*oneShotCmdMgr, materials.buffer, 0, std::span<const MaterialInfo>{materialsVector});
    transferHelper.uploadBuffer(*oneShotCmdMgr, matrices.buffer, 0, std::span<const glm::mat4x4>{instanceMatricesVector});

    instancesToCommandsMap.buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
        .size = instancesToCommandsMapCPU.size() * sizeof(uint32_t),
        .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .name = "Instances to indirect commands map"
    });
    instancesToCommandsMap.size = instancesToCommandsMapCPU.size() * sizeof(uint32_t);
    transferHelper.uploadBuffer(*oneShotCmdMgr, instancesToCommandsMap.buffer, 0, std::span<const uint32_t>(instancesToCommandsMapCPU));
}

const std::unordered_map<PipelineType, SceneUploader::PipelineInfo>& SceneUploader::getPipelines() const {
    return pipelines;
}

const BufferWithSize& SceneUploader::getInstancesToCommandsBuffer() const {
    return instancesToCommandsMap;
}

const BufferWithSize& SceneUploader::getIndirectCommandsBuffer() const {
    return indirectCommands;
}

const BufferWithSize& SceneUploader::getMaterialsBuffer() const {
    return materials;
}

const BufferWithSize& SceneUploader::getMatricesBuffer() const {
    return matrices;
}
