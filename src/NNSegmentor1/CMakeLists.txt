
include_directories(
  model
)

add_executable(NNSegmentor1 NNSegmentor1.cpp)
target_link_libraries(NNSegmentor1 ${LIBS})

