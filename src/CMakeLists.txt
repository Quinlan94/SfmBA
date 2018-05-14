add_library( mysfm SHARED
        CalculateCameraMatrix.cpp
        Common.cpp
        FeatureMatching.cpp
        Triangulation.cpp
        )
#target_link_libraries(mysfm GL GLU glut)