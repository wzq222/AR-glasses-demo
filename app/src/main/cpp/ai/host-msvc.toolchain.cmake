# host 端 vulkan-shaders-gen 工具链：脱离 vcvars 环境直接用 MSVC
# （MinGW gcc 的 fork() 模拟在多线程 shader 生成时会死锁，必须用 MSVC）
# 路径为 DOS 短路径（无空格、正斜杠），换机需同步修改
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_C_COMPILER   "D:/MICROS~1/2022/COMMUN~1/VC/Tools/MSVC/1444~1.352/bin/Hostx64/x64/cl.exe")
set(CMAKE_CXX_COMPILER "D:/MICROS~1/2022/COMMUN~1/VC/Tools/MSVC/1444~1.352/bin/Hostx64/x64/cl.exe")
set(CMAKE_RC_COMPILER "C:/PROGRA~2/WI3CF2~1/10/bin/10.0.26100.0/x64/rc.exe" CACHE FILEPATH "" FORCE)
set(CMAKE_MT "C:/PROGRA~2/WI3CF2~1/10/bin/10.0.26100.0/x64/rc.exe" CACHE FILEPATH "" FORCE)
set(_MSVC_INC "D:/MICROS~1/2022/COMMUN~1/VC/Tools/MSVC/1444~1.352/include" "C:/PROGRA~2/WI3CF2~1/10/include/10.0.26100.0/ucrt" "C:/PROGRA~2/WI3CF2~1/10/include/10.0.26100.0/um" "C:/PROGRA~2/WI3CF2~1/10/include/10.0.26100.0/shared")
set(_MSVC_LIB "D:/MICROS~1/2022/COMMUN~1/VC/Tools/MSVC/1444~1.352/lib/x64" "C:/PROGRA~2/WI3CF2~1/10/lib/10.0.26100.0/ucrt/x64" "C:/PROGRA~2/WI3CF2~1/10/lib/10.0.26100.0/um/x64")
foreach(_d ${_MSVC_INC})
    string(APPEND CMAKE_C_FLAGS_INIT   " /I${_d}")
    string(APPEND CMAKE_CXX_FLAGS_INIT " /I${_d}")
endforeach()
foreach(_d ${_MSVC_LIB})
    string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " /LIBPATH:${_d}")
endforeach()
