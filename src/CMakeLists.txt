cmake_minimum_required( VERSION 2.4.6 )
PROJECT( ObjectCategory_proj )

set(HOMEBREW_INCLUDE_DIR /usr/local/include)
set(HOMEBREW_LIBRARY_DIR /usr/local/lib)
set(BOOST_FILESYSTEM_LIBS boost_filesystem-mt boost_system-mt)

FIND_PACKAGE( OpenCV REQUIRED )

include_directories(${HOMEBREW_INCLUDE_DIR})
link_directories(${HOMEBREW_LIBRARY_DIR})
ADD_EXECUTABLE( ObjectCategory main.cpp ObjectCategory.cpp)
TARGET_LINK_LIBRARIES( ObjectCategory ${OpenCV_LIBS} ${BOOST_FILESYSTEM_LIBS})
