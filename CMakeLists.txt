cmake_minimum_required(VERSION 3.21)
project(MNIST VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#Found this on the Cmake forums from a Mike Jackson.
#Didn't do what it promised but now the exe is in its own folder so I guess that's nice.
# ---------- Setup output Directories -------------------------
if(NOT DEFINED LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_NAME}
    CACHE PATH
    "Single Directory for all Libraries"
    )
endif()

# --------- Setup the Executable output Directory -------------
if(NOT DEFINED RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_NAME}
    CACHE PATH
    "Single Directory for all Executables."
    )
endif()

# --------- Setup the Executable output Directory -------------
if(NOT DEFINED ARCHIVE_OUTPUT_DIRECTORY)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    ${PROJECT_NAME}
    CACHE PATH
    "Single Directory for all static libraries."
    )
endif()

set(DLLS "${PROJECT_SOURCE_DIR}/lib/OpenBLAS/libopenblas.dll")

set(Extras
	src/DatasetProcessing/Reader.cpp
	src/DatasetProcessing/Utility.cpp
	src/Neural/Network.cpp
	)

add_executable(${PROJECT_NAME} src/main.cpp ${Extras})

target_include_directories(${PROJECT_NAME} PUBLIC lib/armadillo)
target_link_libraries(${PROJECT_NAME} ${DLLS})

#Put those darn DLLs into the same folder as the exe. CMAKE DOCUMENTATIONS IS FRUSTRATINGLY HARD TO SEARCH.
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:${PROJECT_NAME}> ${DLLS}
  COMMAND_EXPAND_LISTS
)
