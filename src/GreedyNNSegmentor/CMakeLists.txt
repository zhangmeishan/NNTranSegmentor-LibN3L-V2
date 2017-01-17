
include_directories(
  model
)

add_executable(GreedyNNSegmentor GreedyNNSegmentor.cpp)
target_link_libraries(GreedyNNSegmentor ${LIBS})
