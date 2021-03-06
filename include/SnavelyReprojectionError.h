//
// Created by quinlan on 18-5-25.
//

#ifndef SFM_SNAVELYREPROJECTIONERROR_H
#define SFM_SNAVELYREPROJECTIONERROR_H


#include <iostream>
#include "ceres/ceres.h"


#include "rotation.h"
#include "projection.h"
int i = 0;
class SnavelyReprojectionError
{
public:
    SnavelyReprojectionError(double observation_x, double observation_y):observed_x(observation_x),observed_y(observation_y){}

    template<typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals)const{
        // camera[0,1,2] are the angle-axis rotation
        T predictions[2];
        CamProjectionWithDistortion(camera, point, predictions);
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);

        //std::cout<<"我被执行了"<<i<<"次"<<endl;
        //i++;

        return true;
    }

    static ceres::CostFunction* Create(const double observed_x, const double observed_y){
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError,2,9,3>(
                new SnavelyReprojectionError(observed_x,observed_y)));
    }


private:
    double observed_x;
    double observed_y;
};


#endif //SFM_SNAVELYREPROJECTIONERROR_H
