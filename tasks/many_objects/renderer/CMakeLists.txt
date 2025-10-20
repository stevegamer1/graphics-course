
add_executable(many_objects_renderer
  main.cpp
  App.cpp
  Renderer.cpp
  WorldRenderer.cpp
  # scene_transformations/SceneInvertion.cpp
  stages/SceneUploader.cpp
  stages/CullingManager.cpp
  stages/IndirectDrawManager.cpp
  stages/GBufferDrawer.cpp
  stages/AABBCalculator.cpp
  stages/GBufferLightResolver.cpp
)

target_link_libraries(many_objects_renderer
  PRIVATE glfw etna glm::glm wsi gui scene render_utils)

target_add_shaders(many_objects_renderer
  shaders/static_mesh.frag
  shaders/static_mesh.vert
  shaders/gbuffer_create.frag
  shaders/gbuffer_create.vert
  shaders/light_from_gbuffer.frag
  shaders/light_from_gbuffer.vert
  shaders/cull.comp
  shaders/aabb_calculator.comp
)
