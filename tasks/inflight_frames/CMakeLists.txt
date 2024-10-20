add_executable(inflight_frames
  main.cpp
  App.cpp
)

target_link_libraries(inflight_frames
  PRIVATE glfw etna glm::glm wsi gui)

target_add_shaders(inflight_frames
  shaders/toy.vert
  shaders/toy.frag
  shaders/texture.frag
)

target_link_libraries(inflight_frames PUBLIC render_utils scene)

add_compile_definitions(inflight_frames PRIVATE TEXTURES_ROOT="${PROJECT_SOURCE_DIR}/resources/textures/")
