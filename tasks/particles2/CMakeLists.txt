add_executable(particles2
  main.cpp
  App.cpp
  Renderer.cpp
  WorldRenderer.cpp
  ParticleSystem.cpp
)

target_include_directories(particles2 PUBLIC baker)

target_link_libraries(particles2
  PRIVATE glfw etna glm::glm wsi gui scene render_utils)

target_add_shaders(particles2
  shaders/static_mesh.frag
  shaders/static_mesh.vert
  shaders/particle.frag
  shaders/particle.vert
  shaders/update_particles.comp
  shaders/sort_particles.comp
)
