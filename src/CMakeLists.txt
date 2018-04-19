add_library( mysfm SHARED
        CalculateCameraMatrix.cpp
        Common.cpp
        FeatureMatching.cpp
        fed.cpp
        nldiffusion_functions.cpp
        SaveXYZimages.cpp
        Triangulation.cpp
        utils.cpp
        )
#target_link_libraries(mysfm GL GLU glut)