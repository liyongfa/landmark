#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#


cmake_minimum_required(VERSION 2.8.4)

PROJECT(examples)

include(../dlib/cmake)

# Tell CMake to compile a program.  We do this with the ADD_EXECUTABLE()
# statement which takes the name of the output executable and then a list of
# .cpp files to compile.  Here each example consists of only one .cpp file but
# in general you will make programs that const of many .cpp files.
# Then we tell it to link with dlib.




# Since there are a lot of examples I'm going to use a macro to simply this
# CMakeLists.txt file.  However, usually you will create only one executable in
# your cmake projects and use the syntax shown above.


#here we apply our macros 
#add_example(main)


find_package(OpenCV)
if (OpenCV_FOUND)
   include_directories(${OpenCV_INCLUDE_DIRS})

   #ADD_EXECUTABLE(main main.cpp)
FILE(GLOB files
	"*.h"
	"*.cpp"
)
ADD_EXECUTABLE(examples ${files})

   TARGET_LINK_LIBRARIES(examples dlib ${OpenCV_LIBS} )
endif()


#if (DLIB_LINK_WITH_SQLITE3)
#   add_example(sqlite_ex)
#endif()


