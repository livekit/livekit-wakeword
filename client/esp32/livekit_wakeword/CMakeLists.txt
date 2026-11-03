idf_component_register(
    SRCS src/livekit_wakeword.cc src/lkww_frontend.c
    INCLUDE_DIRS include
    PRIV_INCLUDE_DIRS src
    REQUIRES espressif__esp_codec_dev esp_driver_gpio
    PRIV_REQUIRES espressif__esp-tflite-micro esp_driver_i2s esp_timer heap log freertos)

target_compile_options(${COMPONENT_LIB} PRIVATE -O2)

if(CONFIG_LKWW_MODEL_PATH)
    idf_build_get_property(project_dir PROJECT_DIR)
    get_filename_component(LKWW_MODEL "${CONFIG_LKWW_MODEL_PATH}" ABSOLUTE BASE_DIR "${project_dir}")
    if(NOT EXISTS "${LKWW_MODEL}")
        message(FATAL_ERROR "livekit-wakeword: model '${LKWW_MODEL}' does not exist (LKWW_MODEL_PATH).")
    endif()
    set(LKWW_MODEL_COPY "${CMAKE_CURRENT_BINARY_DIR}/lkww_model.tflite")
    configure_file("${LKWW_MODEL}" "${LKWW_MODEL_COPY}" COPYONLY)
    target_add_binary_data(${COMPONENT_LIB} "${LKWW_MODEL_COPY}" BINARY)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC LKWW_HAS_EMBEDDED_MODEL=1)
endif()
