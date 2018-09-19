find_package( PythonInterp REQUIRED )

add_custom_target(doc 
	COMMAND ${PYTHON_EXECUTABLE} generate_doc.py $<TARGET_FILE:lmbspecialops>
	COMMENT "Building documentation"
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/doc
	)
add_dependencies(doc lmbspecialops)

