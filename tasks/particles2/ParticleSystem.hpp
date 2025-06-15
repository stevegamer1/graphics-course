#pragma once
#include "etna/BlockingTransferHelper.hpp"
#include "etna/ComputePipeline.hpp"
#include "etna/GraphicsPipeline.hpp"
#include "etna/DescriptorSet.hpp"
#include <filesystem>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/vec3.hpp>
#include <etna/Vulkan.hpp>


struct ParticleSystem {
    static const uint32_t MAX_EMITTERS = 10;
    static const uint32_t PARTICLES_PER_EMITTER = 100;  // will be dynamic later (well, actually probably no)

    struct UpdatePushConsts {
        glm::vec4 deltaTimeRelativeAnd3Padding;
        glm::uvec4 firstParticleIndexAndTimeAndAliveCountAndCountToSpawn;
        glm::vec4 minStartVelocityAnd1Padding;
        glm::vec4 maxStartVelocityAnd1Padding;
        glm::vec4 emitterPosAnd1Padding;
    };

    struct SortPushConsts {
        glm::vec4 cameraPosAnd1Padding;
        glm::uvec4 firstParticleIndexAndParticlesNumberAnd2Padding;
    };

    struct RenderPushConsts {
        glm::vec4 camPos;
        glm::mat4x4 viewProj;
    };

    struct ParticleEmitter {
        // std::filesystem::path texture_path;
        // void loadTextureByPath();
        uint32_t whichPlaceOccupies;

        glm::vec3 pos;
        glm::vec3 startVelocityMax;
        glm::vec3 startVelocityMin;
        float accumulatedDesiredSpawn;  // When spawn frequency is low, this thing slowly accumulates.
        float spawnFrequency = 100.0f;
        float particleLifetime = 1.0f;
        float rotationSpeedMin = -1.0f;
        float rotationSpeedMax = 1.0f;
        glm::vec3 spawnZoneExtent = glm::vec3(1);
        glm::vec3 acceleration;
        glm::vec4 startColor{0.0f, 0.0f, 1.0f, 1.0f};
        glm::vec4 endColor{1.0f, 0.0f, 0.0f, 0.0f};
        // void update(glm::vec3 camera_pos, float delta_time);

        struct Particle {
            glm::vec4 posAndAngle;
            glm::vec4 velocityAndTimeToLiveRelative;
        };

        // void spawnParticles(int count);
        // void resetParticle(Particle& p);
        // static bool isParticleAlive(const Particle& p);

        // std::vector<Particle> particleParams = std::vector<Particle>(PARTICLES_PER_EMITTER);  // Will be deleted later.
    };

    std::vector<bool> particleBufferPartIsOccupied;
    uint32_t getFirstFreePBufferPart();

    std::unique_ptr<etna::OneShotCmdMgr> oneShotCommands;

    std::vector<ParticleEmitter> emitters;
    etna::Buffer particleParamsBuffer;
    etna::Buffer aliveCountBuffer;
    etna::DescriptorSet particleParamsDescriptorSet;
    etna::DescriptorSet aliveCountDescriptorSet;

    ParticleSystem();

    void setupPipelines(vk::Format swapchain_format);

    void memBarrierSortAndUpdate(vk::CommandBuffer cmd_buf);

    void update(glm::vec3 camera_pos, float delta_time, vk::CommandBuffer cmd_buf);
    void draw(glm::mat4x4 view_proj, vk::CommandBuffer cmd_buf);

    // For every emitter returns the number of alive particles;
    std::vector<uint32_t> sortParticles(vk::CommandBuffer cmd_buf, glm::vec3 cam_pos);

    void sortEmitters(glm::vec3 cam_pos);

    glm::vec3 cameraPosition;
    float deltaTime;
    float time = 0;

    const char* RENDER_SHADER_NAME = "particle_program";
    const char* UPDATE_SHADER_NAME = "update_particles_program";
    const char* SORT_SHADER_NAME = "sort_particles_program";
    etna::GraphicsPipeline renderPipeline;
    etna::ComputePipeline updatePipeline;
    etna::ComputePipeline sortPipeline;
    etna::BlockingTransferHelper transferHelper;
};
