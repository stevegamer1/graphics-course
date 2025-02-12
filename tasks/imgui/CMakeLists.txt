add_subdirectory(baker)

add_executable(imgui_renderer
  main.cpp
  App.cpp
  Renderer.cpp
  WorldRenderer.cpp
  baker/baker.cpp
)

target_include_directories(imgui_renderer PUBLIC baker)

target_link_libraries(imgui_renderer
  PRIVATE glfw etna glm::glm wsi gui scene render_utils)

target_add_shaders(imgui_renderer
  shaders/static_mesh.frag
  shaders/static_mesh.vert
)
