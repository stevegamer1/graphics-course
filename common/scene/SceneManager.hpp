#pragma once

#include "etna/Image.hpp"
#include <filesystem>

#include <glm/glm.hpp>
#include <tiny_gltf.h>
#include <etna/Buffer.hpp>
#include <etna/BlockingTransferHelper.hpp>
#include <etna/VertexInput.hpp>
#include <vulkan/vulkan_structs.hpp>


enum class PipelineType {
  PBR
};

enum class TextureID : uint32_t { Invalid = ~uint32_t{0} };
enum class MaterialID : uint32_t { Invalid = ~uint32_t{0} };

struct Material {
  PipelineType pipeline = PipelineType::PBR;
  std::vector<TextureID> textures;
  std::vector<glm::vec3> texturesScalarMultipliers;
};

// A single render element (relem) corresponds to a single draw call
// of a certain pipeline with specific bindings (including material data)
struct RenderElement
{
  std::uint32_t vertexOffset;
  std::uint32_t indexOffset;
  std::uint32_t indexCount;

  MaterialID material = MaterialID::Invalid;
};

// A mesh is a collection of relems. A scene may have the same mesh
// located in several different places, so a scene consists of **instances**,
// not meshes.
struct Mesh
{
  std::uint32_t firstRelem;
  std::uint32_t relemCount;
};

class SceneManager
{
public:
  SceneManager();

  void selectScene(std::filesystem::path path);

  // Every instance is a mesh drawn with a certain transform
  // NOTE: maybe you can pass some additional data through unused matrix entries?
  std::span<const glm::mat4x4> getInstanceMatrices() const { return instanceMatrices; }
  std::span<const std::uint32_t> getInstanceMeshes() const { return instanceMeshes; }

  std::span<const Material> getMaterials() const { return materials; }
  std::span<const etna::Image> getImages() const { return images; }

  // Every mesh is a collection of relems
  std::span<const Mesh> getMeshes() const { return meshes; }

  // Every relem is a single draw call
  std::span<const RenderElement> getRenderElements() const { return renderElements; }

  vk::Buffer getVertexBuffer() { return unifiedVbuf.get(); }
  vk::Buffer getIndexBuffer() { return unifiedIbuf.get(); }

  etna::Buffer& getVertexBufferEtna() { return unifiedVbuf; }
  etna::Buffer& getIndexBufferEtna() { return unifiedIbuf; }

  etna::VertexByteStreamFormatDescription getVertexFormatDescription();

  struct Vertex
  {
    // First 3 floats are position, 4th float is a packed normal
    glm::vec4 positionAndNormal;
    // First 2 floats are tex coords, 3rd is a packed tangent, 4th is padding
    glm::vec4 texCoordAndTangentAndPadding;
  };

  static_assert(sizeof(Vertex) == sizeof(float) * 8);

private:
  std::optional<tinygltf::Model> loadModel(std::filesystem::path path);

  struct ProcessedImage {
    std::string_view name;
    vk::Extent3D extent;
    std::vector<std::byte> data;
  };
  struct ProcessedMaterials {
    std::vector<ProcessedImage> textures;
    std::vector<Material> materials;
  };
  ProcessedMaterials processMaterials(const tinygltf::Model& model) const;

  struct ProcessedInstances
  {
    std::vector<glm::mat4x4> matrices;
    std::vector<std::uint32_t> meshes;
  };
  ProcessedInstances processInstances(const tinygltf::Model& model) const;

  struct ProcessedMeshes
  {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<RenderElement> relems;
    std::vector<Mesh> meshes;
  };
  ProcessedMeshes processMeshes(const tinygltf::Model& model) const;
  void uploadData(std::span<const Vertex> vertices, std::span<const std::uint32_t>, std::span<ProcessedImage> processed_textures);

private:
  tinygltf::TinyGLTF loader;
  std::unique_ptr<etna::OneShotCmdMgr> oneShotCommands;
  etna::BlockingTransferHelper transferHelper;

  std::vector<etna::Image> images;
  std::vector<Material> materials;

  std::vector<RenderElement> renderElements;
  std::vector<Mesh> meshes;
  std::vector<glm::mat4x4> instanceMatrices;
  std::vector<std::uint32_t> instanceMeshes;

  etna::Buffer unifiedVbuf;
  etna::Buffer unifiedIbuf;
};
