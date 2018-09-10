find_package( PythonInterp REQUIRED )


file( GLOB test_scripts_list "test_*.py" )

foreach( t IN LISTS test_scripts_list )
        get_filename_component( test_script_name "${t}" NAME_WE )
        get_filename_component( test_script_file "${t}" NAME )
        add_test( NAME "${test_script_name}"
                  COMMAND ${PYTHON_EXECUTABLE} "${test_script_file}"
                  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/test" )
        set_tests_properties( "${test_script_name}" 
		PROPERTIES ENVIRONMENT LMBSPECIALOPS_LIB=$<TARGET_FILE:lmbspecialops> )
        message( STATUS "found test ${test_script_name}" )
endforeach()

