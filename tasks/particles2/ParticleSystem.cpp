#define GLM_SWIZZLE
#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <etna/Profiling.hpp>
#include "ParticleSystem.hpp"
#include "etna/Etna.hpp"
#include "etna/GlobalContext.hpp"
#include "etna/PipelineManager.hpp"


// void ParticleSystem::ParticleEmitter::update([[maybe_unused]] glm::vec3 camera_pos, [[maybe_unused]] float delta_time) {
//     accumulatedDesiredSpawn += spawnFrequency * delta_time;
//     if (accumulatedDesiredSpawn >= 1.0f) {
//         int particlesToSpawn = int(floor(accumulatedDesiredSpawn));
//         spawnParticles(particlesToSpawn);
//         accumulatedDesiredSpawn -= particlesToSpawn;
//     }

//     for (size_t i = 0; i < PARTICLES_PER_EMITTER; ++i) {
//         auto& p = particleParams[i];

//         glm::vec3 velocity = p.velocityAndTimeToLiveRelative.xyz();
//         glm::vec3 newPos = p.posAndAngle.xyz() + velocity * delta_time + acceleration * delta_time * delta_time / 2.0f;
//         glm::vec3 newVel = velocity + acceleration * delta_time;
//         float newTimeToLiveRelative = p.velocityAndTimeToLiveRelative.w - delta_time / particleLifetime;

//         p.posAndAngle = glm::vec4(newPos, 0.0f);
//         p.velocityAndTimeToLiveRelative = glm::vec4(newVel, newTimeToLiveRelative);
//     }
// }

// void ParticleSystem::ParticleEmitter::sortParticles(glm::vec3 cam_pos) {
//     std::sort(
//         particlesVec.begin(),
//         particlesVec.end(), 
//         [cam_pos](const Particle& p1, const Particle& p2) {
//                 return (!isParticleAlive(p1) && isParticleAlive(p2)) ||
//                 ((p1.pos - cam_pos).length() > (p2.pos - cam_pos).length());
//         }
//     );
// }

// void ParticleSystem::ParticleEmitter::resetParticle(ParticleSystem::ParticleEmitter::Particle& p) {
//     auto r = [](){
//         return float(rand()) / RAND_MAX - 0.5;
//     };

//     auto r0 = [](){
//         return float(rand()) / RAND_MAX;
//     };

//     glm::vec3 pPos = pos + glm::vec3(r(), r(), r()) * spawnZoneExtent;
//     glm::vec3 velocity = glm::vec3(r0(), r0(), r0()) * (startVelocityMax - startVelocityMin) + startVelocityMin;
//     p.posAndAngle = glm::vec4(pPos, 0.0f);
//     p.velocityAndTimeToLiveRelative = glm::vec4(velocity, 1.0f);
// }

// void ParticleSystem::ParticleEmitter::spawnParticles(int count) {
//     for (size_t i = 0; i < PARTICLES_PER_EMITTER && count > 0; ++i) {
//         auto& p = particleParams[i];

//         if (!isParticleAlive(p)) {
//             resetParticle(p);
//             --count;
//         }
//     }
// }

// bool ParticleSystem::ParticleEmitter::isParticleAlive(const Particle& p) {
//     return p.velocityAndTimeToLiveRelative.w > 0;
// }

ParticleSystem::ParticleSystem()
    : oneShotCommands{etna::get_context().createOneShotCmdMgr()}
    , transferHelper{etna::BlockingTransferHelper::CreateInfo{.stagingSize = 65536}} {}

void ParticleSystem::setupPipelines(vk::Format swapchain_format) {
    
    etna::create_program(RENDER_SHADER_NAME, {
            PARTICLES2_SHADERS_ROOT "particle.vert.spv",
            PARTICLES2_SHADERS_ROOT "particle.frag.spv"
        });

    auto& pipelineManager = etna::get_context().getPipelineManager();

    renderPipeline = pipelineManager.createGraphicsPipeline(RENDER_SHADER_NAME, {
        .blendingConfig = {
            .attachments={
                vk::PipelineColorBlendAttachmentState{
                    .blendEnable = vk::True,
                    .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
                    .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
                    .colorBlendOp = vk::BlendOp::eAdd,
                    .srcAlphaBlendFactor = vk::BlendFactor::eOne,
                    .dstAlphaBlendFactor = vk::BlendFactor::eZero,
                    .alphaBlendOp = vk::BlendOp::eAdd,
                    .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
                },
            },
            .logicOpEnable = false,
            .logicOp = vk::LogicOp::eAnd,
            .blendConstants = {0, 0, 0, 0}
        },
        .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {swapchain_format},
          .depthAttachmentFormat = vk::Format::eD32Sfloat,
        }
    });

    etna::create_program(UPDATE_SHADER_NAME, {
        PARTICLES2_SHADERS_ROOT "update_particles.comp.spv"
    });

    updatePipeline = pipelineManager.createComputePipeline(UPDATE_SHADER_NAME, {});

    
    etna::create_program(SORT_SHADER_NAME, {
        PARTICLES2_SHADERS_ROOT "sort_particles.comp.spv"
    });

    sortPipeline = pipelineManager.createComputePipeline(SORT_SHADER_NAME, {});

    particleParamsBuffer = etna::get_context().createBuffer({
        .size = sizeof(ParticleEmitter::Particle) * PARTICLES_PER_EMITTER * MAX_EMITTERS,
        .bufferUsage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
        .name = "particles_params_buffer"
    });
    particleBufferPartIsOccupied = std::vector<bool>(MAX_EMITTERS, false);

    aliveCountBuffer = etna::get_context().createBuffer({
        .size = sizeof(uint32_t),
        .bufferUsage = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
        .name = "alive_particles_count_buffer"
    });
}

void ParticleSystem::sortEmitters(glm::vec3 cam_pos) {
    std::sort(emitters.begin(), emitters.end(), 
        [cam_pos](const ParticleEmitter& e1, const ParticleEmitter& e2){
            return (e1.pos - cam_pos).length() > (e2.pos - cam_pos).length();
        }
    );
}

void ParticleSystem::memBarrierSortAndUpdate(vk::CommandBuffer cmd_buf) {
    vk::MemoryBarrier bar{
        .sType = vk::StructureType::eMemoryBarrier,
        .pNext = nullptr,
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead
    };

    cmd_buf.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlags(0),
        1, &bar,
        0, nullptr,
        0, nullptr
    );
}

std::vector<uint32_t> ParticleSystem::sortParticles(vk::CommandBuffer cmd_buf, glm::vec3 cam_pos) {
    ZoneScopedN("sortParticles");

    std::vector<uint32_t> aliveCounts(emitters.size());

    {
        particleParamsDescriptorSet = etna::create_descriptor_set(
            etna::get_shader_program(SORT_SHADER_NAME).getDescriptorLayoutId(0), 
            cmd_buf,
            {
                etna::Binding{0, particleParamsBuffer.genBinding()}
            }
        );
    
        assert(particleParamsDescriptorSet.isValid());
    }

    {
        aliveCountDescriptorSet = etna::create_descriptor_set(
            etna::get_shader_program(SORT_SHADER_NAME).getDescriptorLayoutId(1),
            cmd_buf,
            {
                etna::Binding{0, aliveCountBuffer.genBinding()}
            }
        );

        assert(aliveCountDescriptorSet.isValid());
    }

    {
        vk::DescriptorSet vkSets[2] = {
            particleParamsDescriptorSet.getVkSet(),
            aliveCountDescriptorSet.getVkSet()
        };
        cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute, sortPipeline.getVkPipelineLayout(), 0, 2, vkSets, 0, nullptr);
    }

    for (uint32_t i = 0; i < emitters.size(); ++i)
    {
        auto& e = emitters[i];
        cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, sortPipeline.getVkPipeline());

        SortPushConsts pc{
            glm::vec4(cam_pos, 0),
            glm::uvec4(e.whichPlaceOccupies * PARTICLES_PER_EMITTER, PARTICLES_PER_EMITTER, 0, 0)
        };

        cmd_buf.pushConstants(
            updatePipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eCompute,
            0, sizeof(pc), &pc
        );

        cmd_buf.dispatch(1, 1, 1);

        memBarrierSortAndUpdate(cmd_buf);

        transferHelper.readbackBuffer(*oneShotCommands, std::span(aliveCounts.begin() + i, 1), aliveCountBuffer, 0);
    }

    return aliveCounts;
}

uint32_t ParticleSystem::getFirstFreePBufferPart() {
    for (uint32_t result = 0; result < particleBufferPartIsOccupied.size(); ++result) {
        if (!particleBufferPartIsOccupied[result]) {
            return result;
        }
    }

    return uint32_t(-1);
}

void ParticleSystem::update(glm::vec3 camera_pos, float delta_time, vk::CommandBuffer cmd_buf) {
    ZoneScopedN("updateParticles");

    // for (auto& e : emitters) {
    //     e.update(camera_pos, delta_time);
    // }

    std::vector<uint32_t> aliveCounts = sortParticles(cmd_buf, camera_pos);

    cameraPosition = camera_pos;
    deltaTime = delta_time;
    time += deltaTime;

    {
        ETNA_PROFILE_GPU(cmd_buf, bindSet);

        particleParamsDescriptorSet = etna::create_descriptor_set(
            etna::get_shader_program(UPDATE_SHADER_NAME).getDescriptorLayoutId(0), 
            cmd_buf,
            {
            etna::Binding{0, particleParamsBuffer.genBinding()}
            }
        );
    
        assert(particleParamsDescriptorSet.isValid());
    
        vk::DescriptorSet vkSet = particleParamsDescriptorSet.getVkSet();
        cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute, updatePipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, nullptr);
    }

    for (uint32_t i = 0; i < emitters.size(); ++i)
    {
        auto& e = emitters[i];
        cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, updatePipeline.getVkPipeline());

        e.accumulatedDesiredSpawn += delta_time * e.spawnFrequency;
        uint32_t countToSpawn = uint32_t(floor(e.accumulatedDesiredSpawn));
        e.accumulatedDesiredSpawn -= countToSpawn;

        UpdatePushConsts pc{
            glm::vec4(deltaTime / e.particleLifetime, 0, 0, 0),
            glm::uvec4(e.whichPlaceOccupies * PARTICLES_PER_EMITTER, uint32_t(time * 10000.0f), aliveCounts[i], countToSpawn),
            glm::vec4(e.startVelocityMin, 0),
            glm::vec4(e.startVelocityMax, 0),
            glm::vec4(e.pos, 0)
        };

        cmd_buf.pushConstants(
            updatePipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eCompute,
            0, sizeof(pc), &pc
        );

        cmd_buf.dispatch(PARTICLES_PER_EMITTER, 1, 1);

        memBarrierSortAndUpdate(cmd_buf);
    }
}

void ParticleSystem::draw(glm::mat4x4 view_proj, vk::CommandBuffer cmd_buf) {
    {
        ETNA_PROFILE_GPU(cmd_buf, bindSet);

        particleParamsDescriptorSet = etna::create_descriptor_set(
            etna::get_shader_program(RENDER_SHADER_NAME).getDescriptorLayoutId(0), 
            cmd_buf, 
            {
            etna::Binding{0, particleParamsBuffer.genBinding()}
            }
        );
    
        assert(particleParamsDescriptorSet.isValid());
    
        vk::DescriptorSet vkSet = particleParamsDescriptorSet.getVkSet();
        cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, renderPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, nullptr);
    }

    sortEmitters(cameraPosition);

    for (uint32_t i = 0; i < emitters.size(); ++i) {
        const ParticleEmitter& e = emitters[i];

        cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, renderPipeline.getVkPipeline());

        uint32_t firstInstance = e.whichPlaceOccupies * PARTICLES_PER_EMITTER;

        {
            RenderPushConsts pc{
                glm::vec4(cameraPosition, 1),
                view_proj
            };
    
            cmd_buf.pushConstants(
                renderPipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eVertex,
                0, sizeof(pc), &pc
            );
        }

        // transferHelper.uploadBuffer<ParticleEmitter::Particle>(*oneShotCommands, particleParamsBuffer, i * PARTICLES_PER_EMITTER * sizeof(ParticleEmitter::Particle), e.particleParams);

        cmd_buf.draw(6, PARTICLES_PER_EMITTER, 0, firstInstance);
    }
}

#undef GLM_SWIZZLE
