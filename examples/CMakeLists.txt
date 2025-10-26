# Examples demonstrating neural network usage

add_executable(demo_nn_mcts
  demo_nn_mcts.cpp
)

target_link_libraries(demo_nn_mcts
  PRIVATE
    golomb_core
    golomb_nn
    golomb_mcts
)
