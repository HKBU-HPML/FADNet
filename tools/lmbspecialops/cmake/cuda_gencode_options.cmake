#
# Provide options to configure the gencode string for the nvcc compiler
#

set( CUDA_GENCODE_STRING "")

option( GENERATE_PTX30_CODE "This will generate PTX 3.0 code" OFF )
if( GENERATE_PTX30_CODE )
        list( APPEND CUDA_GENCODE_STRING "-gencode=arch=compute_30,code=compute_30" )
endif()

option( GENERATE_KEPLER_SM30_CODE "This will generate code for the Kepler GPU architecture (sm_30)" OFF )
if( GENERATE_KEPLER_SM30_CODE )
        list( APPEND CUDA_GENCODE_STRING "-gencode=arch=compute_30,code=sm_30" )
endif()

option( GENERATE_KEPLER_SM35_CODE "This will generate code for the Kepler GPU architecture (sm_35)" ON )
if( GENERATE_KEPLER_SM35_CODE )
        list( APPEND CUDA_GENCODE_STRING "-gencode=arch=compute_35,code=sm_35" )
endif()

option( GENERATE_KEPLER_SM37_CODE "This will generate code for the Kepler GPU architecture (sm_37)" ON )
if( GENERATE_KEPLER_SM37_CODE )
        list( APPEND CUDA_GENCODE_STRING "-gencode=arch=compute_37,code=sm_37" )
endif()

option( GENERATE_MAXWELL_SM50_CODE "This will generate code for the Maxwell GPU architecture (sm_50)" OFF )
if( GENERATE_MAXWELL_SM50_CODE )
        list( APPEND CUDA_GENCODE_STRING "-gencode=arch=compute_50,code=sm_50" )
endif()

option( GENERATE_MAXWELL_SM52_CODE "This will generate code for the Maxwell GPU architecture (sm_52)" ON )
if( GENERATE_MAXWELL_SM52_CODE )
        list( APPEND CUDA_GENCODE_STRING "-gencode=arch=compute_52,code=sm_52" )
endif()

option( GENERATE_PASCAL_SM60_CODE "This will generate code for the Pascal GPU architecture (sm_60)" ON )
if( GENERATE_PASCAL_SM60_CODE )
        list( APPEND CUDA_GENCODE_STRING "-gencode=arch=compute_60,code=sm_60" )
endif()

option( GENERATE_PTX60_CODE "This will generate PTX 6.0 code" OFF )
if( GENERATE_PTX60_CODE )
        list( APPEND CUDA_GENCODE_STRING "-gencode=arch=compute_60,code=compute_60" )
endif()

option( GENERATE_PASCAL_SM61_CODE "This will generate code for the Pascal GPU architecture (sm_61)" ON )
if( GENERATE_PASCAL_SM61_CODE )
        list( APPEND CUDA_GENCODE_STRING "-gencode=arch=compute_61,code=sm_61" )
endif()

option( GENERATE_PTX61_CODE "This will generate PTX 6.1 code" OFF )
if( GENERATE_PTX61_CODE )
        list( APPEND CUDA_GENCODE_STRING "-gencode=arch=compute_61,code=compute_61" )
endif()

option( GENERATE_VOLTA_SM70_CODE "This will generate code for the Pascal GPU architecture (sm_70)" OFF )
if( GENERATE_VOLTA_SM70_CODE )
        list( APPEND CUDA_GENCODE_STRING "-gencode=arch=compute_70,code=sm_70" )
endif()

option( GENERATE_PTX70_CODE "This will generate PTX 7.0 code" OFF )
if( GENERATE_PTX70_CODE )
        list( APPEND CUDA_GENCODE_STRING "-gencode=arch=compute_70,code=compute_70" )
endif()

