set (BLOCK_NAME imageLib)
include_directories(
  ${SOURCE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${NVTX_INCLUDE_DIR}
  ${NVML_INCLUDE_DIR}
  ${FREEGLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  ${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
  ${TOP_DIR}/thirdparty/libpng
  ${TOP_DIR}/thirdparty/libz
  ${TOP_DIR}/thirdparty/include
)

#######################################
#######################################
file(GLOB HEADERS *.h *.cuh)
source_group("Headers" FILES ${HEADERS})

add_library( ${BLOCK_NAME} 
${HEADERS}
Convolve.cpp
Convert.cpp
Image.cpp
ImageIO.cpp
RefCntMem.cpp
ImageIOpng.cpp
)

set_target_properties(${BLOCK_NAME} PROPERTIES OUTPUT_NAME_DEBUG ${BLOCK_NAME}d)
set_target_properties(${BLOCK_NAME} PROPERTIES OUTPUT_NAME_RELEASE ${BLOCK_NAME})
