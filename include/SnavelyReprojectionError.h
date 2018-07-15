//
// Created by quinlan on 18-5-25.
//

#ifndef SFM_SNAVELYREPROJECTIONERROR_H
#define SFM_SNAVELYREPROJECTIONERROR_H



#include "ceres/ceres.h"


#include "rotation.h"
#include "projection.h"
#include "Common.h"

class SnavelyReprojectionError
{
public:
    SnavelyReprojectionError(const Eigen::Vector2d point2d_1,
                             const Eigen::Vector2d point2d_2,
                             const Eigen::Vector3d r_vec,
                             const Eigen::Vector3d t_vec,
                             const Eigen::Vector3d k_vec)

            : observed_x_1(point2d_1(0)),
              observed_y_1(point2d_1(1)),
              observed_x_2(point2d_2(0)),
              observed_y_2(point2d_2(1)),
              const_r_0(r_vec(0)),
              const_r_1(r_vec(1)),
              const_r_2(r_vec(2)),
              const_t_0(t_vec(0)),
              const_t_1(t_vec(1)),
              const_t_2(t_vec(2)),
              focal_length(k_vec(0)),
              cx(k_vec(1)),
              cy(k_vec(2)) {}

    template<typename T>
    bool operator()(const T* const pose_2,
                    const T* const point,
                    T* residuals)const{

        T predictions_1[2];
        T predictions_2[2];
        const  T camera[3] = {T(focal_length),T(cx),T(cy)};
        const  T pose_1[6] = {T(const_r_0),T(const_r_1),T(const_r_2),
                              T(const_t_0),T(const_t_1),T(const_t_2)};
        CamProjectionWithDistortion(pose_1,camera, point, predictions_1);
        CamProjectionWithDistortion(pose_2,camera, point, predictions_2);

        residuals[0] = predictions_1[0] - T(observed_x_1);
        residuals[1] = predictions_1[1] - T(observed_y_1);
        residuals[2] = predictions_2[0] - T(observed_x_2);
        residuals[3] = predictions_2[1] - T(observed_y_2);



        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector2d point2d_1,
                                       const Eigen::Vector2d point2d_2,
                                       const Eigen::Vector3d r_vec,
                                       const Eigen::Vector3d t_vec,
                                       const Eigen::Vector3d k_vec){
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError,4,6,3>(
                new SnavelyReprojectionError(point2d_1,point2d_2,r_vec,t_vec,k_vec)));
    }


private:
    double observed_x_1;
    double observed_y_1;
    double observed_x_2;
    double observed_y_2;
    double focal_length;
    double cx;
    double cy;
    double const_r_0;
    double const_r_1;
    double const_r_2;
    double const_t_0;
    double const_t_1;
    double const_t_2;



};


#endif //SFM_SNAVELYREPROJECTIONERROR_H
