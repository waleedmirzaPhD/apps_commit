#------------------------------------------------------------
# CMakeLists for one hiperlife App
#------------------------------------------------------------

#Name
get_filename_component(PROGRAMME_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(PROGRAMME_NAME "hl${PROGRAMME_NAME}")

#Info
message(STATUS "Programme name: ${PROGRAMME_NAME} ")

#Sources
file(GLOB PROGRAM_CPP *.cpp)

#Add executable
include_directories(${CMAKE_CURRENT_LIST_DIR})
add_executable(${PROGRAMME_NAME} ${PROGRAM_CPP})

#Link libs
target_link_libraries(${PROGRAMME_NAME} ${hiperlife_LIBRARIES})
target_link_libraries(${PROGRAMME_NAME} ${Parmmg_LIB})

#Install
install(TARGETS ${PROGRAMME_NAME} DESTINATION ${PROJECT_INSTALL_PATH})
