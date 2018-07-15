//
// Created by quinlan on 18-5-26.
//

#ifndef SFM_PROJECTION_H
#define SFM_PROJECTION_H

#include "rotation.h"

// camera : 9 dims array with
// [0-2] : angle-axis rotation
// [3-5] : translateion

// point : 3D location.
// predictions : 2D predictions with center of the image plane.

template<typename T>
inline bool CamProjectionWithDistortion(const T* pose, const T* camera,const T* point, T* predictions){
    // Rodrigues' formula
    T p[3];
    AngleAxisRotatePoint(pose, point, p);
    // camera[3,4,5] are the translation
    p[0] += pose[3]; p[1] += pose[4]; p[2] += pose[5];

    //像素平面，为什么有的加负号？
    /*
     * 这个取决于你的重投影图像是在相机前还是相机后，很明显这应该是同侧
     */
    T xp = p[0]/p[2];
    T yp = p[1]/p[2];


    const T& cx = camera[1];
    const T& cy = camera[2];

//    T r2 = xp*xp + yp*yp;
//    T distortion = T(1.0) + r2 * (l1 + l2 * r2);

    const T& focal = camera[0];
    predictions[0] = focal *  xp + cx;
    predictions[1] = focal *  yp + cy;

    return true;
}



#endif //SFM_PROJECTION_H
